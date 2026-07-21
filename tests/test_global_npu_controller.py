import pytest


class FakeController:
    instances = []
    fail_rank = None

    def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
        self.rank = rank
        self.keep_calls = 0
        self.release_calls = 0
        self._thread = None
        self._stop_evt = None
        self.instances.append(self)

    def keep(self):
        self.keep_calls += 1
        self._thread = object()
        if self.rank == self.fail_rank:
            raise RuntimeError(f"rank {self.rank} failed")

    def release(self):
        self.release_calls += 1

    def allocation_status(self):
        return None


def test_global_defaults_to_all_visible_npus(monkeypatch):
    from keep_npu.global_npu_controller import global_npu_controller as module

    FakeController.instances = []
    FakeController.fail_rank = None
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 3)
    monkeypatch.setattr(module, "AscendNPUController", FakeController)

    controller = module.GlobalNPUController()

    assert controller.npu_ids == [0, 1, 2]


def test_global_rejects_out_of_range_visible_npu(monkeypatch):
    from keep_npu.global_npu_controller import global_npu_controller as module

    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 2)

    with pytest.raises(ValueError, match="less than 2"):
        module.GlobalNPUController(npu_ids=[2])


def test_global_rolls_back_partial_start(monkeypatch):
    from keep_npu.global_npu_controller import global_npu_controller as module

    FakeController.instances = []
    FakeController.fail_rank = 1
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 2)
    monkeypatch.setattr(module, "AscendNPUController", FakeController)
    controller = module.GlobalNPUController(npu_ids=[0, 1])

    with pytest.raises(RuntimeError, match="rank 1 failed"):
        controller.keep()

    assert FakeController.instances[0].release_calls == 1
    assert FakeController.instances[1].release_calls == 1

