#!/usr/bin/env python3
"""Compatibility command for the original ``keep_npu_alive.py`` interface."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


@dataclass(frozen=True)
class KeepAliveConfig:
    device: str
    interval: float
    size: int
    dtype_name: str
    log_every: int
    warmup: int
    once: bool


class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def request_stop(self, signum: int, _frame: object) -> None:
        print(f"\nreceived signal {signum}, stopping...", flush=True)
        self.stop = True


def normalize_device(device: str) -> str:
    value = device.strip().lower()
    if value.isdigit():
        return f"npu:{value}"
    if value.startswith("npu:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return value
    raise argparse.ArgumentTypeError("device must be like 'npu:0' or '0'")


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def estimate_tensor_bytes(size: int, dtype_name: str) -> int:
    return size * size * DTYPE_BYTES[dtype_name] * 3


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep a Huawei Ascend NPU lightly active by periodically running "
            "a tiny torch_npu matrix multiplication."
        )
    )
    parser.add_argument(
        "-d",
        "--device",
        type=normalize_device,
        default="npu:0",
        help="NPU device to use, for example 'npu:0' or '0'. Default: npu:0",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=positive_float,
        default=5.0,
        help="Seconds to sleep between keepalive operations. Default: 5",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=positive_int,
        default=256,
        help="Square matrix size for the keepalive matmul. Default: 256",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BYTES),
        default="float16",
        help="Tensor dtype. Default: float16",
    )
    parser.add_argument(
        "--log-every",
        type=positive_int,
        default=12,
        help="Print one heartbeat every N iterations. Default: 12",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup matmuls before entering the loop. Default: 3",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run warmup plus one keepalive operation, then exit.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> KeepAliveConfig:
    if args.warmup < 0:
        raise SystemExit("--warmup must be greater than or equal to 0")
    return KeepAliveConfig(
        device=args.device,
        interval=args.interval,
        size=args.size,
        dtype_name=args.dtype,
        log_every=args.log_every,
        warmup=args.warmup,
        once=args.once,
    )


def import_torch_npu():
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Failed to import torch/torch_npu. Install Ascend PyTorch and run "
            "the CANN environment setup first, for example:\n"
            "  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"
        ) from exc
    return torch


def resolve_dtype(torch, dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def create_workload(torch, config: KeepAliveConfig):
    dtype = resolve_dtype(torch, config.dtype_name)
    torch.npu.set_device(config.device)
    a = torch.randn((config.size, config.size), device=config.device, dtype=dtype)
    b = torch.randn((config.size, config.size), device=config.device, dtype=dtype)
    c = torch.empty((config.size, config.size), device=config.device, dtype=dtype)
    torch.npu.synchronize()
    return a, b, c


def run_one_step(torch, a, b, out) -> None:
    torch.matmul(a, b, out=out)
    torch.npu.synchronize()


def log_start(config: KeepAliveConfig) -> None:
    memory = format_bytes(estimate_tensor_bytes(config.size, config.dtype_name))
    print("keep-npu-alive starting", flush=True)
    print(f"  device:   {config.device}", flush=True)
    print(f"  interval: {config.interval:.3f}s", flush=True)
    print(f"  matrix:   {config.size} x {config.size}", flush=True)
    print(f"  dtype:    {config.dtype_name}", flush=True)
    print(f"  tensors:  about {memory}", flush=True)
    print("press Ctrl+C to stop", flush=True)


def controller_kwargs(config: KeepAliveConfig) -> dict:
    """Translate legacy matrix flags into the new controller's public inputs."""
    return {
        "npu_ids": [int(config.device.split(":", 1)[1])],
        "interval": config.interval,
        "vram_to_keep": estimate_tensor_bytes(config.size, config.dtype_name),
        "busy_threshold": -1,
        "workload": "aicore",
    }


def run_keepalive(config: KeepAliveConfig) -> int:
    from keep_npu.global_npu_controller import GlobalNPUController

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request_stop)
    signal.signal(signal.SIGTERM, stop_flag.request_stop)

    log_start(config)
    with GlobalNPUController(**controller_kwargs(config)):
        if config.once:
            return 0
        iteration = 0
        while not stop_flag.stop:
            iteration += 1
            if iteration == 1 or iteration % config.log_every == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{now}] keepalive #{iteration} on {config.device}",
                    flush=True,
                )
            time.sleep(config.interval)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    return run_keepalive(config)


if __name__ == "__main__":
    sys.exit(main())
