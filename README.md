# keep-npu

Keep a Huawei Ascend NPU lightly active with periodic `torch_npu` work.

The script allocates a small pair of tensors on the selected NPU and runs a
tiny matrix multiplication at a fixed interval. This mirrors the common
keep-GPU-awake pattern while keeping the default workload intentionally low.

## Requirements

- Huawei Ascend NPU runtime and CANN environment
- Python 3.10+
- PyTorch with `torch_npu`

Before running, load the Ascend toolkit environment:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

## Usage

Run with defaults:

```bash
python keep_npu_alive.py
```

Use a specific device:

```bash
python keep_npu_alive.py --device npu:0
python keep_npu_alive.py --device 0
```

Tune the keepalive interval and matrix size:

```bash
python keep_npu_alive.py --interval 5 --size 256
```

Run one keepalive operation and exit:

```bash
python keep_npu_alive.py --once
```

## Test

The included tests only cover argument parsing and helper behavior, so they do
not require a real NPU:

```bash
python -m unittest tests/test_keep_npu_alive.py -v
```
