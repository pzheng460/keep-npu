"""Multi-device Ascend keepalive coordination."""

from __future__ import annotations

import threading
from typing import List, Optional, Union

from keep_npu.single_npu_controller.ascend_npu_controller import (
    AscendNPUController,
)
from keep_npu.single_npu_controller.workload import validate_workload_vram
from keep_npu.utilities.logger import setup_logger
from keep_npu.utilities.platform_manager import visible_torch_device_count
from keep_npu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    DEFAULT_WORKLOAD,
    validate_busy_threshold,
    validate_interval,
    validate_npu_ids,
    validate_workload,
)

logger = setup_logger(__name__)


class ControllerStartupUnavailable(Exception):
    """Expected Ascend hardware/runtime unavailability during startup."""


class NoNPUAvailableError(ControllerStartupUnavailable, ValueError):
    pass


class InvalidVisibleNPUSelectionError(ValueError):
    pass


def _resolve_visible_npu_ids(npu_ids: Optional[List[int]]) -> List[int]:
    try:
        count = visible_torch_device_count()
    except Exception as exc:
        raise NoNPUAvailableError(str(exc)) from exc
    if count <= 0:
        raise NoNPUAvailableError("No NPUs available for GlobalNPUController")
    if npu_ids is None:
        return list(range(count))
    invalid = [npu_id for npu_id in npu_ids if npu_id >= count]
    if invalid:
        raise InvalidVisibleNPUSelectionError(
            f"npu_ids must be visible device ordinals less than {count}; got {invalid}"
        )
    return npu_ids


class GlobalNPUController:
    def __init__(
        self,
        npu_ids: Optional[List[int]] = None,
        interval: Union[int, float] = 300,
        vram_to_keep: Union[int, str] = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
        workload: str = DEFAULT_WORKLOAD,
    ):
        self.interval = validate_interval(interval)
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        self.workload = validate_workload(workload)
        validate_workload_vram(self.workload, vram_to_keep)
        self.vram_to_keep = vram_to_keep
        npu_ids = validate_npu_ids(npu_ids)
        self.npu_ids = _resolve_visible_npu_ids(npu_ids)
        self.controllers = [
            AscendNPUController(
                rank=rank,
                interval=self.interval,
                vram_to_keep=self.vram_to_keep,
                busy_threshold=self.busy_threshold,
                workload=self.workload,
            )
            for rank in self.npu_ids
        ]

    def keep(self) -> None:
        started = []
        for controller in self.controllers:
            try:
                controller.keep()
            except BaseException:
                rollback_errors = []
                for candidate in [controller, *reversed(started)]:
                    try:
                        candidate.release()
                    except Exception as release_exc:
                        rollback_errors.append((candidate.rank, release_exc))
                if rollback_errors:
                    detail = "; ".join(
                        f"rank {rank}: {exc}" for rank, exc in rollback_errors
                    )
                    logger.error("NPU startup rollback release failures: %s", detail)
                raise
            started.append(controller)

    def release(self) -> None:
        errors = []
        lock = threading.Lock()

        def release_one(controller) -> None:
            try:
                controller.release()
            except Exception as exc:
                with lock:
                    errors.append((controller.rank, exc))

        threads = [
            threading.Thread(target=release_one, args=(controller,))
            for controller in self.controllers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            detail = "; ".join(f"rank {rank}: {exc}" for rank, exc in errors)
            raise RuntimeError(f"Failed to release NPU controllers: {detail}")

    def runtime_error(self) -> Optional[Exception]:
        for controller in self.controllers:
            error = controller.allocation_status()
            if error is not None:
                return error
        return None

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
