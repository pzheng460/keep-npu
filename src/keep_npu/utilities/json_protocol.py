"""Strict JSON helpers for public protocol boundaries."""

import json
import math
from typing import Any


def _reject_nonstandard_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON constant: {constant}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"invalid JSON number: {value}")
    return parsed


def strict_json_loads(data: Any) -> Any:
    """Decode standard JSON while rejecting non-finite numeric values."""
    return json.loads(
        data,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )
