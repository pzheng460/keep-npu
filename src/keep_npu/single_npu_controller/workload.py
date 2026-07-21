"""Pure planning helpers for Ascend keepalive workloads."""

from dataclasses import dataclass
from math import isqrt

FP16_BYTES = 2
MATRIX_COUNT = 3
MATRIX_ALIGNMENT = 16
MAX_MATRIX_DIM = 12288
AICORE_BATCH_ITERATIONS = 32
MIN_AICORE_BYTES = MATRIX_COUNT * MATRIX_ALIGNMENT**2 * FP16_BYTES


@dataclass(frozen=True)
class AICorePlan:
    """Allocation dimensions that fit inside a public VRAM budget."""

    matrix_dim: int
    filler_elements: int
    allocated_bytes: int


def plan_aicore_workload(float32_elements: int) -> AICorePlan:
    """Plan three aligned FP16 matrices and float32 filler within a budget."""
    budget_bytes = float32_elements * 4
    if budget_bytes < MIN_AICORE_BYTES:
        raise ValueError(
            f"aicore workload requires --vram of at least {MIN_AICORE_BYTES} bytes"
        )
    raw_dim = isqrt(budget_bytes // (MATRIX_COUNT * FP16_BYTES))
    matrix_dim = min(MAX_MATRIX_DIM, raw_dim)
    matrix_dim -= matrix_dim % MATRIX_ALIGNMENT
    matrix_bytes = MATRIX_COUNT * matrix_dim**2 * FP16_BYTES
    filler_elements = (budget_bytes - matrix_bytes) // 4
    allocated_bytes = matrix_bytes + filler_elements * 4
    return AICorePlan(matrix_dim, filler_elements, allocated_bytes)
