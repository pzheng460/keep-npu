import threading

import pytest


class FakeNPU:
    def __init__(self, count=2):
        self.count = count
        self.current = 0
        self.empty_cache_calls = 0
        self.sync_calls = 0

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

    def mem_get_info(self, rank=None):
        return 6 * 1024**3, 8 * 1024**3

    def get_device_name(self, rank):
        return f"Ascend Fake {rank}"


class FakeTorch:
    float32 = "float32"

    def __init__(self, count=2):
        self.npu = FakeNPU(count)
        self.allocations = []
        self.relu_calls = 0

    def device(self, value):
        return value

    def rand(self, elements, **kwargs):
        tensor = {"elements": elements, **kwargs}
        self.allocations.append(tensor)
        return tensor

    def relu_(self, tensor):
        self.relu_calls += 1
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
        vram_to_keep=4,
        busy_threshold=25,
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
    monkeypatch.setattr(module, "get_npu_utilization", lambda rank: 100)
    controller = module.AscendNPUController(
        rank=0,
        interval=0.01,
        iterations=2,
        vram_to_keep=8,
        busy_threshold=-1,
    )

    controller.keep()
    controller.release()

    assert fake.allocations[0]["elements"] == 2
    assert fake.relu_calls >= 1
    assert fake.npu.sync_calls >= 1
    assert fake.npu.empty_cache_calls == 1
    assert controller._thread is None


def test_controller_surfaces_startup_device_failure(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    fake.npu.set_device = lambda rank: (_ for _ in ()).throw(RuntimeError("NPU lost"))
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0, vram_to_keep=4, busy_threshold=-1
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
    controller = module.AscendNPUController(rank=0, vram_to_keep=4)

    class AliveThread:
        def is_alive(self):
            return True

    controller._thread = AliveThread()
    controller._stop_evt = threading.Event()
    controller._stop_evt.set()

    with pytest.raises(RuntimeError, match="startup did not complete"):
        controller.keep()

