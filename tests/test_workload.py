import pytest

from keep_npu.single_npu_controller.workload import (
    FP16_BYTES,
    MATRIX_COUNT,
    MAX_MIXED_MATRIX_DIM,
    MIN_MIXED_BYTES,
    MIN_MIXED_VECTOR_BYTES,
    MIN_RANDOM_BYTES,
    RANDOM_HBM_BYTES,
    RANDOM_VECTOR_BYTES,
    AICorePlan,
    plan_aicore_workload,
    plan_mixed_workload,
    plan_random_workload,
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
    assert plan.allocated_bytes == 3 * 8192 * 8192 * 2 + plan.filler_elements * 4


def test_minimum_mixed_plan_reserves_aligned_cube_and_vector_memory():
    plan = plan_mixed_workload(MIN_MIXED_BYTES // 4)

    assert plan.matrix_dim == 16
    assert plan.vector_elements == MIN_MIXED_VECTOR_BYTES // 4
    assert plan.allocated_bytes == MIN_MIXED_BYTES


def test_mixed_plan_rejects_budget_below_minimum():
    with pytest.raises(
        ValueError,
        match=f"mixed workload requires --vram of at least {MIN_MIXED_BYTES} bytes",
    ):
        plan_mixed_workload((MIN_MIXED_BYTES // 4) - 1)


def test_default_mixed_plan_caps_cube_and_assigns_remainder_to_vector():
    budget_elements = 1024**3 // 4
    plan = plan_mixed_workload(budget_elements)
    matrix_bytes = 3 * MAX_MIXED_MATRIX_DIM**2 * 2

    assert MAX_MIXED_MATRIX_DIM == 12288
    assert plan.matrix_dim == MAX_MIXED_MATRIX_DIM
    assert plan.vector_elements == (1024**3 - matrix_bytes) // 4
    assert plan.allocated_bytes == 1024**3


def test_workload_vram_validation_applies_mixed_minimum():
    assert validate_workload_vram("mixed", MIN_MIXED_BYTES) == MIN_MIXED_BYTES // 4
    with pytest.raises(ValueError, match="mixed workload requires --vram"):
        validate_workload_vram("mixed", MIN_MIXED_BYTES - 4)


def test_minimum_random_plan_reserves_all_three_engines():
    plan = plan_random_workload(MIN_RANDOM_BYTES // 4)

    assert plan.matrix_dim == 16
    assert plan.vector_elements == RANDOM_VECTOR_BYTES // 4
    assert plan.hbm_buffer_elements == (RANDOM_HBM_BYTES // 2) // 4
    assert plan.reserve_elements == 0
    assert plan.allocated_bytes == MIN_RANDOM_BYTES


def test_random_plan_rejects_budget_below_minimum():
    with pytest.raises(
        ValueError,
        match=f"random workload requires --vram of at least {MIN_RANDOM_BYTES} bytes",
    ):
        plan_random_workload((MIN_RANDOM_BYTES // 4) - 1)


def test_one_gib_random_plan_accounts_for_every_byte():
    plan = plan_random_workload(1024**3 // 4)
    matrix_bytes = MATRIX_COUNT * plan.matrix_dim**2 * FP16_BYTES
    active_bytes = plan.vector_elements * 4 + plan.hbm_buffer_elements * 4 * 2

    assert plan.matrix_dim % 16 == 0
    assert plan.matrix_dim <= MAX_MIXED_MATRIX_DIM
    assert active_bytes == RANDOM_VECTOR_BYTES + RANDOM_HBM_BYTES
    assert matrix_bytes + active_bytes + plan.reserve_elements * 4 == 1024**3
    assert plan.allocated_bytes == 1024**3


def test_workload_vram_validation_applies_random_minimum():
    assert validate_workload_vram("random", MIN_RANDOM_BYTES) == MIN_RANDOM_BYTES // 4
    with pytest.raises(ValueError, match="random workload requires --vram"):
        validate_workload_vram("random", MIN_RANDOM_BYTES - 4)
