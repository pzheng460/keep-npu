import threading
import time

import pytest


class FakeStream:
    def __init__(self, fake, device=None):
        self.fake = fake
        self.device = device
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class FakeStreamContext:
    def __init__(self, fake, stream):
        self.fake = fake
        self.stream = stream
        self.previous = None

    def __enter__(self):
        self.previous = getattr(self.fake.stream_state, "active", None)
        self.fake.stream_state.active = self.stream

    def __exit__(self, exc_type, exc, tb):
        self.fake.stream_state.active = self.previous


class FakeNPU:
    def __init__(self, count=2, fake=None):
        self.count = count
        self.fake = fake
        self.current = 0
        self.empty_cache_calls = 0
        self.sync_calls = 0
        self.streams = []

    def is_available(self):
        return self.count > 0

    def device_count(self):
        return self.count

    def current_device(self):
        return self.current

    def set_device(self, rank):
        if not 0 <= rank < self.count:
            raise RuntimeError("invalid device")
        self.current = rank

    def empty_cache(self):
        self.empty_cache_calls += 1

    def synchronize(self):
        self.sync_calls += 1

    def Stream(self, device=None):
        stream = FakeStream(self.fake, device)
        self.streams.append(stream)
        return stream

    def stream(self, stream):
        return FakeStreamContext(self.fake, stream)

    def mem_get_info(self, rank=None):
        return 6 * 1024**3, 8 * 1024**3

    def get_device_name(self, rank):
        return f"Ascend Fake {rank}"


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, count=2):
        self.npu = FakeNPU(count, self)
        self.allocations = []
        self.matmul_calls = 0
        self.relu_calls = 0
        self.on_matmul = None
        self.on_relu = None
        self.stream_state = threading.local()
        self.operation_lock = threading.Lock()
        self.matmul_streams = []
        self.relu_streams = []

    def device(self, value):
        return value

    def rand(self, *shape, **kwargs):
        tensor = {"shape": shape, **kwargs}
        if len(shape) == 1 and isinstance(shape[0], int):
            tensor["elements"] = shape[0]
        self.allocations.append(tensor)
        return tensor

    def empty(self, *shape, **kwargs):
        tensor = {"shape": shape, **kwargs}
        self.allocations.append(tensor)
        return tensor

    def matmul(self, left, right, *, out):
        with self.operation_lock:
            self.matmul_calls += 1
            calls = self.matmul_calls
            self.matmul_streams.append(getattr(self.stream_state, "active", None))
        if self.on_matmul is not None:
            self.on_matmul(calls)
        return out

    def relu_(self, tensor):
        with self.operation_lock:
            self.relu_calls += 1
            calls = self.relu_calls
            self.relu_streams.append(getattr(self.stream_state, "active", None))
        if self.on_relu is not None:
            self.on_relu(calls)
        return tensor


def test_visible_count_uses_torch_npu(monkeypatch):
    from keep_npu.utilities import platform_manager

    fake = FakeTorch(count=3)
    monkeypatch.setattr(platform_manager, "load_torch_npu", lambda: fake)

    assert platform_manager.visible_torch_device_count() == 3


def test_visible_count_wraps_enumeration_failure(monkeypatch):
    from keep_npu.utilities import platform_manager

    fake = FakeTorch()
    fake.npu.device_count = lambda: (_ for _ in ()).throw(RuntimeError("driver down"))
    monkeypatch.setattr(platform_manager, "load_torch_npu", lambda: fake)

    with pytest.raises(
        platform_manager.DeviceEnumerationUnavailableError,
        match="Unable to enumerate visible NPUs: driver down",
    ):
        platform_manager.visible_torch_device_count()


def test_controller_rejects_invalid_rank_before_backend_probe(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    monkeypatch.setattr(
        module,
        "visible_torch_device_count",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe")),
    )

    with pytest.raises(TypeError, match="rank must be an integer"):
        module.AscendNPUController(rank="0", vram_to_keep=4)


def test_controller_defaults_to_mixed_workload(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)

    controller = module.AscendNPUController(rank=0, vram_to_keep="1GiB")

    assert controller.workload == "mixed"


def test_controller_unknown_utilization_defers_allocation(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    monkeypatch.setattr(module, "get_npu_utilization", lambda rank: None)
    controller = module.AscendNPUController(
        rank=0,
        interval=0.01,
        iterations=1,
        vram_to_keep=1536,
        busy_threshold=25,
        workload="aicore",
    )

    controller.keep()
    try:
        assert controller._thread is not None
        assert controller._thread.is_alive()
        assert fake.allocations == []
        assert controller.allocation_status() is None
    finally:
        controller.release()


def test_controller_unconditional_mode_allocates_runs_and_releases(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    monkeypatch.setattr(
        module,
        "get_npu_utilization",
        lambda rank: (_ for _ in ()).throw(
            AssertionError("unconditional mode must skip telemetry")
        ),
    )
    controller = module.AscendNPUController(
        rank=0,
        interval=0.01,
        iterations=2,
        vram_to_keep=8,
        busy_threshold=-1,
        workload="vector",
    )

    controller.keep()
    controller.release()

    assert fake.allocations[0]["elements"] == 2
    assert fake.relu_calls >= 1
    assert fake.npu.sync_calls >= 1
    assert fake.npu.empty_cache_calls == 1
    assert controller._thread is None


def test_controller_default_workload_runs_matmul_and_relu(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    monkeypatch.setattr(module, "get_npu_utilization", lambda rank: 0)
    controller = module.AscendNPUController(
        rank=0,
        interval=0.01,
        vram_to_keep="1GiB",
        busy_threshold=-1,
    )
    cube_started = threading.Event()
    vector_started = threading.Event()
    fake.on_matmul = lambda _calls: cube_started.set()
    fake.on_relu = lambda _calls: vector_started.set()

    controller.keep()
    try:
        assert cube_started.wait(timeout=1.0)
        assert vector_started.wait(timeout=1.0)
    finally:
        controller.release()

    assert fake.matmul_calls >= 1
    assert fake.relu_calls >= 1
    assert fake.matmul_streams
    assert fake.relu_streams
    assert set(fake.matmul_streams).isdisjoint(fake.relu_streams)


def test_mixed_allocation_uses_one_budget_and_two_streams(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(rank=0, vram_to_keep="1GiB")

    allocation = controller._allocate_mixed(controller.vram_to_keep)

    assert allocation.left["device"] == "npu:0"
    assert allocation.right["device"] == "npu:0"
    assert allocation.output["device"] == "npu:0"
    assert allocation.vectors
    assert allocation.cube_stream is not allocation.vector_stream
    assert allocation.plan.allocated_bytes == 1024**3
    assert sum(item["elements"] for item in allocation.vectors) == (
        allocation.plan.vector_elements
    )


def test_mixed_batch_routes_operations_to_distinct_streams(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(rank=0, vram_to_keep="1GiB")
    allocation = controller._allocate_mixed(controller.vram_to_keep)
    controller._stop_evt = threading.Event()
    cube_started = threading.Event()
    vector_started = threading.Event()

    def on_matmul(calls):
        cube_started.set()
        vector_started.wait(timeout=1.0)
        if calls == 2:
            controller._stop_evt.set()

    fake.on_matmul = on_matmul
    fake.on_relu = lambda _calls: vector_started.set()

    controller._run_mixed_batch(allocation)

    assert cube_started.is_set()
    assert vector_started.is_set()
    assert fake.matmul_streams
    assert fake.relu_streams
    assert set(fake.matmul_streams) == {allocation.cube_stream}
    assert set(fake.relu_streams) == {allocation.vector_stream}


@pytest.mark.parametrize(
    ("failed_engine", "message"),
    [
        ("cube", "cube feeder failed: cube stream failed"),
        ("vector", "vector feeder failed: vector stream failed"),
    ],
)
def test_mixed_feeder_failure_stops_batch_and_preserves_engine(
    monkeypatch, failed_engine, message
):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(rank=0, vram_to_keep="1GiB")
    allocation = controller._allocate_mixed(controller.vram_to_keep)
    controller._stop_evt = threading.Event()
    if failed_engine == "cube":
        fake.on_matmul = lambda _calls: (_ for _ in ()).throw(
            RuntimeError("cube stream failed")
        )
    else:
        fake.on_relu = lambda _calls: (_ for _ in ()).throw(
            RuntimeError("vector stream failed")
        )

    with pytest.raises(RuntimeError, match=message):
        controller._run_mixed_batch(allocation)

    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name in {"npu-cube-0", "npu-vector-0"}
    ]


def test_mixed_runtime_failure_reaches_allocation_status(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    fake.on_matmul = lambda _calls: (_ for _ in ()).throw(
        RuntimeError("cube runtime failed")
    )
    controller = module.AscendNPUController(
        rank=0,
        vram_to_keep="1GiB",
        busy_threshold=-1,
    )

    controller.keep()
    try:
        deadline = time.monotonic() + 1.0
        while controller.allocation_status() is None and time.monotonic() < deadline:
            time.sleep(0.001)

        assert str(controller.allocation_status()) == (
            "rank 0: unexpected Ascend keep worker failure: "
            "cube feeder failed: cube runtime failed"
        )
    finally:
        controller.release()


def test_aicore_allocation_uses_selected_device_and_budget(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=2)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 2)
    controller = module.AscendNPUController(
        rank=1, vram_to_keep="1MiB", workload="aicore"
    )

    allocation = controller._allocate_aicore(controller.vram_to_keep)

    assert {tensor["device"] for tensor in fake.allocations} == {"npu:1"}
    assert 1024**2 - 3 <= allocation.plan.allocated_bytes <= 1024**2


def test_aicore_batch_observes_stop_event(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0, vram_to_keep="1MiB", workload="aicore"
    )
    controller._stop_evt = threading.Event()
    fake.on_matmul = lambda calls: controller._stop_evt.set() if calls == 2 else None

    controller._run_aicore_batch(controller._allocate_aicore(controller.vram_to_keep))

    assert fake.matmul_calls == 2


def test_unconditional_mode_does_not_sleep_between_compute_batches(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0,
        interval=60,
        vram_to_keep="1MiB",
        busy_threshold=-1,
        workload="aicore",
    )

    class RecordingEvent:
        def __init__(self):
            self.wait_calls = []

        def is_set(self):
            return False

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            return False

    stop_evt = RecordingEvent()

    assert controller._wait_for_next_check(stop_evt) is False
    assert stop_evt.wait_calls == []


def test_vector_batch_synchronizes_in_bounded_chunks(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0,
        iterations=65,
        vram_to_keep=4,
        workload="vector",
    )
    controller._stop_evt = threading.Event()

    controller._run_vector_batch([{"tensor": 0}])

    assert fake.relu_calls == 65
    assert fake.npu.sync_calls == 3


def test_controller_surfaces_startup_device_failure(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    fake.npu.set_device = lambda rank: (_ for _ in ()).throw(RuntimeError("NPU lost"))
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0, vram_to_keep=1536, busy_threshold=-1, workload="aicore"
    )

    with pytest.raises(RuntimeError, match="NPU lost"):
        controller.keep()

    assert controller._thread is None
    assert controller._stop_evt is None


def test_controller_rejects_retry_while_worker_is_stopping(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0, vram_to_keep=1536, workload="aicore"
    )

    class AliveThread:
        def is_alive(self):
            return True

    controller._thread = AliveThread()
    controller._stop_evt = threading.Event()
    controller._stop_evt.set()

    with pytest.raises(RuntimeError, match="startup did not complete"):
        controller.keep()
