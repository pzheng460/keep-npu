import pytest

from keep_npu.single_npu_controller.workload import (
    AICorePlan,
    plan_aicore_workload,
    validate_workload_vram,
)


def test_minimum_aicore_plan_is_three_aligned_fp16_matrices():
    plan = plan_aicore_workload(1536 // 4)

    assert plan == AICorePlan(
        matrix_dim=16,
        filler_elements=0,
        allocated_bytes=1536,
    )


def test_aicore_plan_rejects_budget_below_minimum():
    with pytest.raises(
        ValueError,
        match="aicore workload requires --vram of at least 1536 bytes",
    ):
        plan_aicore_workload((1536 // 4) - 1)


def test_workload_vram_validation_applies_workload_specific_minimum():
    assert validate_workload_vram("aicore", 1536) == 1536 // 4
    assert validate_workload_vram("vector", 4) == 1

    with pytest.raises(
        ValueError,
        match="aicore workload requires --vram of at least 1536 bytes",
    ):
        validate_workload_vram("aicore", 4)


def test_aicore_plan_is_aligned_capped_and_inside_budget():
    budget_elements = 1024**3 // 4

    plan = plan_aicore_workload(budget_elements)

    assert plan.matrix_dim == 8192
    assert plan.matrix_dim % 16 == 0
    assert budget_elements * 4 - 3 <= plan.allocated_bytes <= budget_elements * 4
    assert (
        plan.allocated_bytes
        == 3 * 8192 * 8192 * 2 + plan.filler_elements * 4
    )
