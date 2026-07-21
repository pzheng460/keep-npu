#!/usr/bin/env python3
"""Compatibility wrapper for the packaged KeepNPU legacy command."""

from keep_npu.legacy import *  # noqa: F403
from keep_npu.legacy import main

if __name__ == "__main__":
    raise SystemExit(main())
