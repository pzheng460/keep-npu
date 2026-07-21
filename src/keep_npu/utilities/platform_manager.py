"""Ascend runtime discovery with lazy hardware imports."""

from __future__ import annotations

from enum import Enum


class ComputingPlatform(Enum):
    CPU = "cpu"
    ASCEND = "ascend"
    # Internal aliases keep the ported service contract tests importable while
    # the public product exposes only the Ascend backend.
    CUDA = "ascend"
    ROCM = "ascend"
    MACM = "ascend"


class NPUBackendUnavailableError(RuntimeError):
    """PyTorch Ascend is not importable or did not register ``torch.npu``."""


class DeviceEnumerationUnavailableError(RuntimeError):
    """Visible torch NPU ordinals could not be enumerated."""


def load_torch_npu():
    """Import the vendor runtime lazily and return its patched torch module."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise NPUBackendUnavailableError(
            "Failed to import torch/torch_npu. Install Ascend PyTorch and load "
            "the CANN environment first."
        ) from exc
    if not hasattr(torch, "npu"):
        raise NPUBackendUnavailableError(
            "torch_npu imported but torch.npu is unavailable"
        )
    return torch


def visible_torch_device_count() -> int:
    """Return the number of torch-visible Ascend device ordinals."""
    try:
        return int(load_torch_npu().npu.device_count())
    except NPUBackendUnavailableError:
        raise
    except Exception as exc:
        raise DeviceEnumerationUnavailableError(
            f"Unable to enumerate visible NPUs: {exc}"
        ) from exc


_cached_platform: ComputingPlatform | None = None


def get_platform() -> ComputingPlatform:
    global _cached_platform
    if _cached_platform is not None:
        return _cached_platform
    try:
        torch = load_torch_npu()
        if bool(torch.npu.is_available()):
            _cached_platform = ComputingPlatform.ASCEND
            return _cached_platform
    except NPUBackendUnavailableError:
        pass
    _cached_platform = ComputingPlatform.CPU
    return _cached_platform
