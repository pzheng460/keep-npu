"""Single-device Ascend keepalive controller."""

from __future__ import annotations

import threading
import time
from typing import Optional

from keep_npu.single_npu_controller.base_npu_controller import BaseNPUController
from keep_npu.utilities.logger import setup_logger
from keep_npu.utilities.npu_monitor import get_npu_utilization
from keep_npu.utilities.platform_manager import (
    load_torch_npu,
    visible_torch_device_count,
)
from keep_npu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    validate_busy_threshold,
    validate_positive_integer,
    validate_rank_type,
    validate_visible_rank,
)

logger = setup_logger(__name__)
MAX_CHUNK_ELEMENTS = 1 << 30


class AscendNPUController(BaseNPUController):
    def __init__(
        self,
        *,
        rank: int,
        interval: float = 1.0,
        iterations: int = 1,
        vram_to_keep: str | int = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
    ):
        rank = validate_rank_type(rank)
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        self.iterations = validate_positive_integer(iterations, "iterations")
        self.rank = validate_visible_rank(rank, visible_torch_device_count())
        self._torch = load_torch_npu()
        self.device = self._torch.device(f"npu:{rank}")
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._failure_exc: Optional[Exception] = None
        self._num_elements: Optional[int] = None

    def keep(self) -> None:
        if self._thread and self._thread.is_alive():
            if self._stop_evt is not None and self._stop_evt.is_set():
                raise RuntimeError(
                    f"rank {self.rank}: previous keep thread startup did not complete"
                )
            logger.warning("rank %s: keep thread already running", self.rank)
            return
        self._failure_exc = None
        self._num_elements = int(self.vram_to_keep)
        self._stop_evt = threading.Event()
        startup_evt = threading.Event()
        startup_errors: list[Exception] = []
        self._thread = threading.Thread(
            target=self._keep_loop,
            args=(startup_evt, startup_errors),
            name=f"npu-keeper-ascend-{self.rank}",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._stop_evt = None
            raise
        startup_timeout = 5.0
        if not startup_evt.wait(startup_timeout):
            self._stop_evt.set()
            self._thread.join(timeout=1.0)
            raise RuntimeError(
                f"rank {self.rank}: keep thread did not complete startup within "
                f"{startup_timeout:.1f}s"
            )
        if startup_errors:
            self._thread.join(timeout=1.0)
            self._thread = None
            self._stop_evt = None
            raise startup_errors[0]

    def release(self) -> None:
        thread = self._thread
        if not (thread and thread.is_alive()):
            if thread is not None:
                self._torch.npu.empty_cache()
                self._thread = None
                self._stop_evt = None
            return
        stop_evt = self._stop_evt
        if stop_evt is None:
            raise RuntimeError(f"rank {self.rank}: stop event missing")
        stop_evt.set()
        join_timeout = max(2.0, min(float(self.interval) + 2.0, 30.0))
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            raise TimeoutError(
                f"rank {self.rank}: keep thread did not stop within {join_timeout:.1f}s"
            )
        self._torch.npu.empty_cache()
        self._thread = None
        self._stop_evt = None

    def _allocate(self, num_elements: int):
        chunks = []
        remaining = num_elements
        while remaining:
            chunk_size = min(remaining, MAX_CHUNK_ELEMENTS)
            chunks.append(
                self._torch.rand(
                    chunk_size,
                    device=self.device,
                    dtype=self._torch.float32,
                    requires_grad=False,
                )
            )
            remaining -= chunk_size
        return chunks

    def _keep_loop(
        self,
        startup_evt: Optional[threading.Event] = None,
        startup_errors: Optional[list[Exception]] = None,
    ) -> None:
        startup_confirmed = startup_evt is None

        def confirm_startup() -> None:
            nonlocal startup_confirmed
            if not startup_confirmed:
                startup_confirmed = True
                assert startup_evt is not None
                startup_evt.set()

        def record_failure(exc: Exception) -> None:
            wrapped = RuntimeError(
                f"rank {self.rank}: unexpected Ascend keep worker failure: {exc}"
            )
            if not startup_confirmed and startup_errors is not None:
                startup_errors.append(exc)
            else:
                self._failure_exc = wrapped
            confirm_startup()

        stop_evt = self._stop_evt
        if stop_evt is None:
            record_failure(RuntimeError("stop event not initialized"))
            return
        try:
            self._torch.npu.set_device(self.rank)
        except Exception as exc:
            record_failure(exc)
            return
        tensors = None
        while not stop_evt.is_set():
            try:
                utilization = self._monitor_utilization(self.rank)
                if not self._should_run_batch(utilization, self.busy_threshold):
                    confirm_startup()
                    if stop_evt.wait(self.interval):
                        return
                    continue
                tensors = self._allocate(int(self._num_elements or 0))
                confirm_startup()
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self._torch.npu.empty_cache()
                    confirm_startup()
                    if stop_evt.wait(self.interval):
                        return
                    continue
                record_failure(exc)
                return
            except Exception as exc:
                record_failure(exc)
                return
        if tensors is None:
            confirm_startup()
            return
        while not stop_evt.is_set():
            try:
                utilization = self._monitor_utilization(self.rank)
                if self._should_run_batch(utilization, self.busy_threshold):
                    self._run_batch(tensors)
                if stop_evt.wait(self.interval):
                    break
            except Exception as exc:
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected Ascend keep worker failure: {exc}"
                )
                return

    def _run_batch(self, tensors) -> None:
        started = time.monotonic()
        for _ in range(self.iterations):
            for tensor in tensors:
                self._torch.relu_(tensor)
            if self._stop_evt is not None and self._stop_evt.is_set():
                break
        self._torch.npu.synchronize()
        logger.debug(
            "rank %s: keepalive batch completed in %.2f ms",
            self.rank,
            (time.monotonic() - started) * 1000,
        )

    @staticmethod
    def _monitor_utilization(rank: int) -> Optional[int]:
        return get_npu_utilization(rank)

    def allocation_status(self) -> Optional[Exception]:
        return self._failure_exc

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
