"""Single-device Ascend keepalive controller."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from keep_npu.single_npu_controller.base_npu_controller import BaseNPUController
from keep_npu.single_npu_controller.random_workload import RandomPhaseScheduler
from keep_npu.single_npu_controller.workload import (
    AICORE_BATCH_ITERATIONS,
    AICorePlan,
    MixedPlan,
    RandomPlan,
    plan_aicore_workload,
    plan_mixed_workload,
    plan_random_workload,
)
from keep_npu.utilities.logger import setup_logger
from keep_npu.utilities.npu_monitor import get_npu_utilization
from keep_npu.utilities.platform_manager import (
    load_torch_npu,
    visible_torch_device_count,
)
from keep_npu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    DEFAULT_WORKLOAD,
    validate_busy_threshold,
    validate_positive_integer,
    validate_rank_type,
    validate_visible_rank,
    validate_workload,
)

logger = setup_logger(__name__)
MAX_CHUNK_ELEMENTS = 1 << 30
VECTOR_SYNC_INTERVAL = 32
MAX_MIXED_ACTIVE_VECTOR_BYTES = 160 * 1024**2
MIXED_BATCH_SECONDS = 1.0
MIXED_UNCONDITIONAL_BATCH_SECONDS = 60.0
MIXED_FEEDER_JOIN_TIMEOUT = 5.0
MIXED_CUBE_CHUNK = 4
MIXED_VECTOR_CHUNK = 64
RANDOM_QUANTUM_SECONDS = 0.1
RANDOM_CUBE_CHUNK = 1
RANDOM_VECTOR_CHUNK = 8
RANDOM_HBM_CHUNK = 8
RANDOM_FEEDER_JOIN_TIMEOUT = 5.0
ASCEND_STARTUP_TIMEOUT_SECONDS = 30.0


@dataclass
class AICoreAllocation:
    """Preallocated matrices and filler for one AI Core worker."""

    left: Any
    right: Any
    output: Any
    fillers: List[Any]
    plan: AICorePlan


@dataclass
class MixedAllocation:
    """Disjoint tensors and streams for concurrent Cube and Vector pressure."""

    left: Any
    right: Any
    output: Any
    vectors: List[Any]
    reserve_vectors: List[Any]
    plan: MixedPlan
    cube_stream: Any
    vector_stream: Any


@dataclass
class RandomAllocation:
    """Disjoint tensors and streams for random Cube, Vector, and HBM pressure."""

    left: Any
    right: Any
    output: Any
    vector: Any
    hbm_source: Any
    hbm_target: Any
    reserves: List[Any]
    plan: RandomPlan
    cube_stream: Any
    vector_stream: Any
    hbm_stream: Any


def _raise_random_session_error(
    *,
    failures: list[tuple[str, Exception]],
    coordinator_failure: Optional[Exception],
    stuck: list[str],
) -> None:
    """Raise the original runtime error before any cleanup-only timeout."""
    if failures:
        engine, exc = failures[0]
        raise RuntimeError(f"{engine} feeder failed: {exc}") from exc
    if coordinator_failure is not None:
        raise coordinator_failure
    if stuck:
        raise TimeoutError(f"random feeder threads did not stop: {', '.join(stuck)}")


class AscendNPUController(BaseNPUController):
    def __init__(
        self,
        *,
        rank: int,
        interval: float = 1.0,
        iterations: int = 5000,
        vram_to_keep: str | int = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
        workload: str = DEFAULT_WORKLOAD,
    ):
        rank = validate_rank_type(rank)
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        self.workload = validate_workload(workload)
        if self.workload == "mixed":
            plan_mixed_workload(self.vram_to_keep)
        elif self.workload == "random":
            plan_random_workload(self.vram_to_keep)
        elif self.workload == "aicore":
            plan_aicore_workload(self.vram_to_keep)
        self.iterations = validate_positive_integer(iterations, "iterations")
        self.rank = validate_visible_rank(rank, visible_torch_device_count())
        self._torch = load_torch_npu()
        self.device = self._torch.device(f"npu:{rank}")
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._failure_exc: Optional[Exception] = None
        self._num_elements: Optional[int] = None
        self._startup_timeout_seconds = ASCEND_STARTUP_TIMEOUT_SECONDS
        self._random_scheduler_factory = RandomPhaseScheduler

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
        startup_timeout = self._startup_timeout_seconds
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

    def clear_cache(self) -> None:
        """Release process-wide unused Ascend allocator blocks."""
        self._torch.npu.empty_cache()

    def release(self, *, clear_cache: bool = True) -> None:
        thread = self._thread
        if not (thread and thread.is_alive()):
            if thread is not None:
                if clear_cache:
                    self.clear_cache()
                self._thread = None
                self._stop_evt = None
            return
        stop_evt = self._stop_evt
        if stop_evt is None:
            raise RuntimeError(f"rank {self.rank}: stop event missing")
        stop_evt.set()
        join_timeout = max(
            MIXED_FEEDER_JOIN_TIMEOUT + 1.0,
            RANDOM_FEEDER_JOIN_TIMEOUT + 1.0,
            min(float(self.interval) + 2.0, 30.0),
        )
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            raise TimeoutError(
                f"rank {self.rank}: keep thread did not stop within {join_timeout:.1f}s"
            )
        if clear_cache:
            self.clear_cache()
        self._thread = None
        self._stop_evt = None

    def _allocate_vector(self, num_elements: int) -> List[Any]:
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

    def _allocate_reserve(self, num_elements: int) -> List[Any]:
        chunks = []
        remaining = num_elements
        while remaining:
            chunk_size = min(remaining, MAX_CHUNK_ELEMENTS)
            chunks.append(
                self._torch.empty(
                    chunk_size,
                    device=self.device,
                    dtype=self._torch.float32,
                    requires_grad=False,
                )
            )
            remaining -= chunk_size
        return chunks

    def _allocate_aicore(self, num_elements: int) -> AICoreAllocation:
        plan = plan_aicore_workload(num_elements)
        shape = (plan.matrix_dim, plan.matrix_dim)
        common = {
            "device": self.device,
            "dtype": self._torch.float16,
            "requires_grad": False,
        }
        left = self._torch.rand(shape, **common)
        right = self._torch.rand(shape, **common)
        output = self._torch.empty(shape, **common)
        fillers = self._allocate_vector(plan.filler_elements)
        return AICoreAllocation(left, right, output, fillers, plan)

    def _allocate_mixed(self, num_elements: int) -> MixedAllocation:
        plan = plan_mixed_workload(num_elements)
        shape = (plan.matrix_dim, plan.matrix_dim)
        common = {
            "device": self.device,
            "dtype": self._torch.float16,
            "requires_grad": False,
        }
        left = self._torch.rand(shape, **common)
        right = self._torch.rand(shape, **common)
        output = self._torch.empty(shape, **common)
        max_active_elements = MAX_MIXED_ACTIVE_VECTOR_BYTES // 4
        active_vector_elements = min(plan.vector_elements, max_active_elements)
        reserve_vector_elements = plan.vector_elements - active_vector_elements
        vectors = self._allocate_vector(active_vector_elements)
        reserve_vectors = self._allocate_reserve(reserve_vector_elements)
        cube_stream = self._torch.npu.Stream(device=self.device)
        vector_stream = self._torch.npu.Stream(device=self.device)
        return MixedAllocation(
            left=left,
            right=right,
            output=output,
            vectors=vectors,
            reserve_vectors=reserve_vectors,
            plan=plan,
            cube_stream=cube_stream,
            vector_stream=vector_stream,
        )

    def _allocate_random(self, num_elements: int) -> RandomAllocation:
        plan = plan_random_workload(num_elements)
        shape = (plan.matrix_dim, plan.matrix_dim)
        common = {
            "device": self.device,
            "dtype": self._torch.float16,
            "requires_grad": False,
        }
        left = self._torch.rand(shape, **common)
        right = self._torch.rand(shape, **common)
        output = self._torch.empty(shape, **common)
        vector = self._allocate_vector(plan.vector_elements)[0]
        hbm_source = self._allocate_vector(plan.hbm_buffer_elements)[0]
        hbm_target = self._allocate_reserve(plan.hbm_buffer_elements)[0]
        reserves = self._allocate_reserve(plan.reserve_elements)
        return RandomAllocation(
            left=left,
            right=right,
            output=output,
            vector=vector,
            hbm_source=hbm_source,
            hbm_target=hbm_target,
            reserves=reserves,
            plan=plan,
            cube_stream=self._torch.npu.Stream(device=self.device),
            vector_stream=self._torch.npu.Stream(device=self.device),
            hbm_stream=self._torch.npu.Stream(device=self.device),
        )

    def _allocate_workload(self, num_elements: int) -> Any:
        if self.workload == "mixed":
            return self._allocate_mixed(num_elements)
        if self.workload == "random":
            return self._allocate_random(num_elements)
        if self.workload == "aicore":
            return self._allocate_aicore(num_elements)
        return self._allocate_vector(num_elements)

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
                utilization = self._current_utilization()
                if not self._should_run_batch(utilization, self.busy_threshold):
                    confirm_startup()
                    if stop_evt.wait(self.interval):
                        return
                    continue
                tensors = self._allocate_workload(int(self._num_elements or 0))
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
                utilization = self._current_utilization()
                if self._should_run_batch(utilization, self.busy_threshold):
                    self._run_batch(tensors)
                if self._wait_for_next_check(stop_evt):
                    break
            except Exception as exc:
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected Ascend keep worker failure: {exc}"
                )
                return

    def _run_vector_batch(self, tensors: List[Any]) -> None:
        started = time.monotonic()
        pending_iterations = 0
        for _ in range(self.iterations):
            for tensor in tensors:
                self._torch.relu_(tensor)
            pending_iterations += 1
            if pending_iterations >= VECTOR_SYNC_INTERVAL:
                self._torch.npu.synchronize()
                pending_iterations = 0
            if self._stop_evt is not None and self._stop_evt.is_set():
                break
        if pending_iterations:
            self._torch.npu.synchronize()
        logger.debug(
            "rank %s: keepalive batch completed in %.2f ms",
            self.rank,
            (time.monotonic() - started) * 1000,
        )

    def _run_aicore_batch(self, allocation: AICoreAllocation) -> None:
        started = time.monotonic()
        for _ in range(AICORE_BATCH_ITERATIONS):
            self._torch.matmul(
                allocation.left,
                allocation.right,
                out=allocation.output,
            )
            if self._stop_evt is not None and self._stop_evt.is_set():
                break
        self._torch.npu.synchronize()
        logger.debug(
            "rank %s: AI Core keepalive batch completed in %.2f ms",
            self.rank,
            (time.monotonic() - started) * 1000,
        )

    def _mixed_batch_seconds(self) -> float:
        if self.busy_threshold == -1:
            return MIXED_UNCONDITIONAL_BATCH_SECONDS
        return MIXED_BATCH_SECONDS

    def _run_mixed_batch(self, allocation: MixedAllocation) -> None:
        deadline = time.monotonic() + self._mixed_batch_seconds()
        cancel = threading.Event()
        failures: list[tuple[str, Exception]] = []
        failure_lock = threading.Lock()

        def should_stop() -> bool:
            return (
                cancel.is_set()
                or (self._stop_evt is not None and self._stop_evt.is_set())
                or time.monotonic() >= deadline
            )

        def record_failure(engine: str, exc: Exception) -> None:
            with failure_lock:
                if not failures:
                    failures.append((engine, exc))
            cancel.set()

        def cube_feeder() -> None:
            try:
                self._torch.npu.set_device(self.rank)
                with self._torch.npu.stream(allocation.cube_stream):
                    while not should_stop():
                        for _ in range(MIXED_CUBE_CHUNK):
                            self._torch.matmul(
                                allocation.left,
                                allocation.right,
                                out=allocation.output,
                            )
                            if should_stop():
                                break
                        allocation.cube_stream.synchronize()
            except Exception as exc:
                record_failure("cube", exc)

        def vector_feeder() -> None:
            try:
                self._torch.npu.set_device(self.rank)
                with self._torch.npu.stream(allocation.vector_stream):
                    while not should_stop():
                        for _ in range(MIXED_VECTOR_CHUNK):
                            for tensor in allocation.vectors:
                                self._torch.relu_(tensor)
                            if should_stop():
                                break
                        allocation.vector_stream.synchronize()
            except Exception as exc:
                record_failure("vector", exc)

        feeders = [
            threading.Thread(
                target=cube_feeder,
                name=f"npu-cube-{self.rank}",
                daemon=True,
            ),
            threading.Thread(
                target=vector_feeder,
                name=f"npu-vector-{self.rank}",
                daemon=True,
            ),
        ]
        for feeder in feeders:
            feeder.start()
        while not should_stop() and any(feeder.is_alive() for feeder in feeders):
            remaining = max(0.0, deadline - time.monotonic())
            cancel.wait(timeout=min(0.05, remaining))
        join_deadline = time.monotonic() + MIXED_FEEDER_JOIN_TIMEOUT
        for feeder in feeders:
            feeder.join(timeout=max(0.0, join_deadline - time.monotonic()))
        stuck = [feeder.name for feeder in feeders if feeder.is_alive()]
        if stuck:
            cancel.set()
            raise TimeoutError(f"mixed feeder threads did not stop: {', '.join(stuck)}")
        if failures:
            engine, exc = failures[0]
            raise RuntimeError(f"{engine} feeder failed: {exc}") from exc

    def _run_random_session(self, allocation: RandomAllocation) -> None:
        stop_evt = self._stop_evt
        if stop_evt is None:
            raise RuntimeError("stop event not initialized")
        scheduler = self._random_scheduler_factory()
        cancel = threading.Event()
        failures: list[tuple[str, Exception]] = []
        failure_lock = threading.Lock()
        state_lock = threading.Lock()
        state: dict[str, Any] = {"enabled": False, "snapshot": None}

        def should_stop() -> bool:
            return cancel.is_set() or stop_evt.is_set()

        def record_failure(engine: str, exc: Exception) -> None:
            with failure_lock:
                if not failures:
                    failures.append((engine, exc))
            cancel.set()

        def current_duty(engine: str) -> float:
            with state_lock:
                snapshot = state["snapshot"]
                if not state["enabled"] or snapshot is None:
                    return 0.0
                return float(getattr(snapshot, f"{engine}_duty"))

        def run_duty_cycle(engine: str, operation: Any, stream: Any) -> None:
            try:
                self._torch.npu.set_device(self.rank)
                with self._torch.npu.stream(stream):
                    while not should_stop():
                        quantum_started = time.monotonic()
                        active_deadline = quantum_started + (
                            RANDOM_QUANTUM_SECONDS * current_duty(engine)
                        )
                        while not should_stop() and time.monotonic() < active_deadline:
                            operation()
                            # Include device execution in the active budget. Without
                            # this synchronization, asynchronous kernels can queue
                            # beyond the intended duty-cycle window.
                            stream.synchronize()
                        stream.synchronize()
                        remaining = RANDOM_QUANTUM_SECONDS - (
                            time.monotonic() - quantum_started
                        )
                        if remaining > 0:
                            cancel.wait(remaining)
            except Exception as exc:
                record_failure(engine, exc)

        def cube_operation() -> None:
            for _ in range(RANDOM_CUBE_CHUNK):
                if should_stop():
                    break
                self._torch.matmul(
                    allocation.left,
                    allocation.right,
                    out=allocation.output,
                )

        def vector_operation() -> None:
            for _ in range(RANDOM_VECTOR_CHUNK):
                if should_stop():
                    break
                self._torch.sin(allocation.vector, out=allocation.vector)

        hbm_direction = [False]

        def hbm_operation() -> None:
            for _ in range(RANDOM_HBM_CHUNK):
                if should_stop():
                    break
                if hbm_direction[0]:
                    allocation.hbm_source.copy_(
                        allocation.hbm_target, non_blocking=True
                    )
                else:
                    allocation.hbm_target.copy_(
                        allocation.hbm_source, non_blocking=True
                    )
                hbm_direction[0] = not hbm_direction[0]

        feeders = [
            threading.Thread(
                target=run_duty_cycle,
                args=("cube", cube_operation, allocation.cube_stream),
                name=f"npu-random-cube-{self.rank}",
                daemon=True,
            ),
            threading.Thread(
                target=run_duty_cycle,
                args=("vector", vector_operation, allocation.vector_stream),
                name=f"npu-random-vector-{self.rank}",
                daemon=True,
            ),
            threading.Thread(
                target=run_duty_cycle,
                args=("hbm", hbm_operation, allocation.hbm_stream),
                name=f"npu-random-hbm-{self.rank}",
                daemon=True,
            ),
        ]
        for feeder in feeders:
            feeder.start()

        coordinator_failure: Optional[Exception] = None
        try:
            last_update = time.monotonic()
            next_probe = last_update
            enabled = self.busy_threshold < 0
            previous_profile: Optional[str] = None
            while not should_stop():
                now = time.monotonic()
                if self.busy_threshold >= 0 and now >= next_probe:
                    utilization = self._monitor_utilization(self.rank)
                    enabled = self._should_run_batch(utilization, self.busy_threshold)
                    next_probe = now + self.interval
                active_seconds = max(0.0, now - last_update) if enabled else 0.0
                snapshot = scheduler.advance(active_seconds)
                last_update = now
                with state_lock:
                    state["enabled"] = enabled
                    state["snapshot"] = snapshot
                if snapshot.profile != previous_profile:
                    logger.debug(
                        "rank %s: random workload phase %s "
                        "(Cube %.0f%%, Vector %.0f%%, HBM %.0f%%)",
                        self.rank,
                        snapshot.profile,
                        snapshot.target_cube_duty * 100,
                        snapshot.target_vector_duty * 100,
                        snapshot.target_hbm_duty * 100,
                    )
                    previous_profile = snapshot.profile
                cancel.wait(RANDOM_QUANTUM_SECONDS)
        except Exception as exc:
            coordinator_failure = exc
        finally:
            cancel.set()

        join_deadline = time.monotonic() + RANDOM_FEEDER_JOIN_TIMEOUT
        for feeder in feeders:
            feeder.join(timeout=max(0.0, join_deadline - time.monotonic()))
        stuck = [feeder.name for feeder in feeders if feeder.is_alive()]
        _raise_random_session_error(
            failures=failures,
            coordinator_failure=coordinator_failure,
            stuck=stuck,
        )

    def _run_batch(self, allocation: Any) -> None:
        if self.workload == "mixed":
            self._run_mixed_batch(allocation)
            return
        if self.workload == "random":
            self._run_random_session(allocation)
            return
        if self.workload == "aicore":
            self._run_aicore_batch(allocation)
        else:
            self._run_vector_batch(allocation)

    @staticmethod
    def _monitor_utilization(rank: int) -> Optional[int]:
        return get_npu_utilization(rank)

    def _current_utilization(self) -> Optional[int]:
        if self.busy_threshold < 0:
            return None
        return self._monitor_utilization(self.rank)

    def _wait_for_next_check(self, stop_evt: threading.Event) -> bool:
        if self.busy_threshold < 0:
            return stop_evt.is_set()
        return stop_evt.wait(self.interval)

    def allocation_status(self) -> Optional[Exception]:
        return self._failure_exc

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
