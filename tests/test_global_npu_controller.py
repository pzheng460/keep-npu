import pytest


class FakeController:
    instances = []
    fail_rank = None

    def __init__(self, *, rank, interval, vram_to_keep, busy_threshold, workload):
        self.rank = rank
        self.workload = workload
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


def test_global_rejects_small_aicore_budget_before_hardware_enumeration(monkeypatch):
    from keep_npu.global_npu_controller import global_npu_controller as module

    monkeypatch.setattr(
        module,
        "visible_torch_device_count",
        lambda: (_ for _ in ()).throw(AssertionError("hardware should not be queried")),
    )

    with pytest.raises(ValueError, match="at least 1536 bytes"):
        module.GlobalNPUController(npu_ids=[0], vram_to_keep=4)


def test_global_controller_passes_workload_to_each_device(monkeypatch):
    from keep_npu.global_npu_controller import global_npu_controller as module

    FakeController.instances = []
    FakeController.fail_rank = None
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 2)
    monkeypatch.setattr(module, "AscendNPUController", FakeController)

    controller = module.GlobalNPUController(npu_ids=[0, 1], workload="vector")

    assert [item.workload for item in controller.controllers] == ["vector", "vector"]


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


def test_global_rollback_attempts_every_release_when_one_release_fails(monkeypatch):
    from keep_npu.global_npu_controller import global_npu_controller as module

    class ReleaseFailureController(FakeController):
        fail_rank = 2

        def release(self):
            super().release()
            if self.rank == 1:
                raise RuntimeError("rank 1 release failed")

    ReleaseFailureController.instances = []
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 3)
    monkeypatch.setattr(module, "AscendNPUController", ReleaseFailureController)
    controller = module.GlobalNPUController(npu_ids=[0, 1, 2])

    with pytest.raises(RuntimeError, match="rank 2 failed"):
        controller.keep()

    assert [item.release_calls for item in ReleaseFailureController.instances] == [
        1,
        1,
        1,
    ]
