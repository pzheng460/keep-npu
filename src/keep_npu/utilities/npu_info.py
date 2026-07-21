"""Ascend NPU inventory and best-effort ``npu-smi`` telemetry."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from keep_npu.utilities.logger import setup_logger
from keep_npu.utilities.platform_manager import (
    DeviceEnumerationUnavailableError,
    load_torch_npu,
    visible_torch_device_count,
)
from keep_npu.utilities.session_config import (
    normalize_memory_byte_pair,
    normalize_utilization_percent,
)

logger = setup_logger(__name__)

_TABLE_ROW = re.compile(
    r"^\|\s*(?P<id>[0-9]+)\s+(?P<name>Ascend\s+[^|\s]+).*?"
    r"\|[^|]*\|\s*(?P<util>[0-9]+)\s+"
    r"(?P<used>[0-9]+)\s*/\s*(?P<total>[0-9]+)\s*\|?\s*$",
    re.IGNORECASE,
)
_ASCEND_25_DEVICE_ROW = re.compile(
    r"^\|\s*(?P<id>[0-9]+)\s+(?P<name>[^|]+?)\s*\|\s*[A-Za-z]+\s*\|"
)
_ASCEND_25_TELEMETRY_ROW = re.compile(
    r"^\|\s*[0-9]+\s*\|\s*[^|]+\|\s*(?P<util>[0-9]+)\s+"
    r"[0-9]+\s*/\s*[0-9]+\s+(?P<used>[0-9]+)\s*/\s*(?P<total>[0-9]+)\s*\|"
)


def _telemetry_record(
    physical_id: int, name: str, utilization_mb: str, used_mb: str, total_mb: str
) -> Dict[str, Any]:
    total = int(total_mb) * 1024**2
    used = int(used_mb) * 1024**2
    total, used = normalize_memory_byte_pair(total, used)
    utilization = normalize_utilization_percent(int(utilization_mb))
    return {
        "physical_id": physical_id,
        "name": name.strip(),
        "memory_total": total,
        "memory_used": used,
        "utilization": int(utilization) if utilization is not None else None,
    }


def parse_npu_smi_output(output: str) -> List[Dict[str, Any]]:
    """Parse one-line and Ascend 25.x two-line ``npu-smi info`` tables."""
    records: List[Dict[str, Any]] = []
    pending_device: Optional[tuple[int, str]] = None
    for line in output.splitlines():
        stripped = line.strip()
        match = _TABLE_ROW.match(stripped)
        if match is not None:
            records.append(
                _telemetry_record(
                    int(match.group("id")),
                    match.group("name"),
                    match.group("util"),
                    match.group("used"),
                    match.group("total"),
                )
            )
            pending_device = None
            continue
        header = _ASCEND_25_DEVICE_ROW.match(stripped)
        if header is not None:
            pending_device = (int(header.group("id")), header.group("name").strip())
            continue
        telemetry = _ASCEND_25_TELEMETRY_ROW.match(stripped)
        if telemetry is not None and pending_device is not None:
            records.append(
                _telemetry_record(
                    pending_device[0],
                    pending_device[1],
                    telemetry.group("util"),
                    telemetry.group("used"),
                    telemetry.group("total"),
                )
            )
            pending_device = None
        elif stripped.startswith("+"):
            pending_device = None
    return records


def _run_npu_smi(timeout: float = 3.0) -> List[Dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("npu-smi unavailable: %s", exc)
        return []
    if completed.returncode != 0:
        logger.debug("npu-smi failed: %s", completed.stderr.strip())
        return []
    return parse_npu_smi_output(completed.stdout)


def _physical_ids_for_visible_count(count: int) -> Optional[List[int]]:
    raw = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
    if raw is None:
        return list(range(count))
    tokens = [token.strip() for token in raw.split(",")]
    if (
        len(tokens) != count
        or any(not token.isascii() or not token.isdigit() for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        return None
    return [int(token) for token in tokens]


def _torch_memory_info(torch, rank: int) -> tuple[Optional[int], Optional[int]]:
    try:
        free, total = torch.npu.mem_get_info(rank)
        total = int(total)
        used = total - int(free)
        return normalize_memory_byte_pair(total, used)
    except Exception:
        return None, None


def list_npus() -> List[Dict[str, Any]]:
    """Return start-compatible records for every selectable visible ordinal."""
    torch = load_torch_npu()
    count = visible_torch_device_count()
    if count <= 0:
        return []
    physical_ids = _physical_ids_for_visible_count(count)
    smi_by_id = {record["physical_id"]: record for record in _run_npu_smi()}
    current = None
    try:
        current = int(torch.npu.current_device())
    except Exception:
        pass
    records: List[Dict[str, Any]] = []
    try:
        for visible_id in range(count):
            try:
                torch.npu.set_device(visible_id)
            except Exception as exc:
                logger.debug("NPU ordinal %s is not selectable: %s", visible_id, exc)
                continue
            physical_id = (
                physical_ids[visible_id] if physical_ids is not None else None
            )
            smi = smi_by_id.get(physical_id, {})
            total, used = _torch_memory_info(torch, visible_id)
            if smi.get("memory_total") is not None:
                total = smi["memory_total"]
                used = smi.get("memory_used")
            try:
                fallback_name = str(torch.npu.get_device_name(visible_id))
            except Exception:
                fallback_name = f"npu:{visible_id}"
            record: Dict[str, Any] = {
                "id": visible_id,
                "visible_id": visible_id,
                "platform": "ascend",
                "name": smi.get("name") or fallback_name,
                "memory_total": total,
                "memory_used": used,
                "utilization": smi.get("utilization"),
            }
            if physical_id is not None:
                record["physical_id"] = physical_id
            records.append(record)
    except DeviceEnumerationUnavailableError:
        raise
    finally:
        if current is not None:
            try:
                torch.npu.set_device(current)
            except Exception:
                pass
    return records


def get_npu_utilization(rank: int) -> Optional[int]:
    for record in list_npus():
        if record["visible_id"] == rank:
            value = record.get("utilization")
            return int(value) if value is not None else None
    return None


def get_npu_info() -> List[Dict[str, Any]]:
    """Backward-compatible internal name used by the shared service."""
    return list_npus()
