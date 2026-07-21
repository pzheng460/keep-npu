# KeepNPU

KeepNPU is a small, polite Huawei Ascend NPU keeper for shared machines. It
matches KeepGPU's CLI, service, REST/JSON-RPC, MCP, and dashboard workflows,
using `torch_npu` and `npu-smi` for Ascend devices.

It allocates only when utilization backoff permits, keeps the requested device
memory signal lightweight, and releases cleanly on exit.

## Requirements

- Python 3.9–3.13
- Huawei Ascend driver and CANN runtime
- A mutually compatible PyTorch and `torch_npu` installation

Load the CANN environment before using KeepNPU when your system does not do so
automatically:

```console
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

## Install

```console
python -m pip install keep-npu
```

`torch` and `torch_npu` are deliberately not installed as package dependencies:
their versions must match the server's CANN and driver stack.

To install the latest unreleased revision directly from GitHub:

```console
python -m pip install 'keep-npu @ git+https://github.com/pzheng460/keep-npu.git'
```

## Blocking mode

Keep all visible NPUs, or select torch-visible ordinals with `--npu-ids`:

```console
keep-npu --vram 1GiB --interval 60
keep-npu --npu-ids 0,2 --vram 512MiB --interval 30
```

Press `Ctrl+C` to release memory. The default utilization threshold is 25%;
KeepNPU backs off while a device is busier than that or telemetry is unknown.
Use `--busy-threshold -1` only when you intentionally want to disable backoff.

## Service and dashboard

Run the local HTTP service in the foreground:

```console
keep-npu serve --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` for the dashboard. Non-blocking CLI workflows use
the same service and auto-start it on localhost by default:

```console
keep-npu list-npus
keep-npu start --npu-ids 0 --vram 1GiB --interval 60
keep-npu status
keep-npu stop --job-id JOB_ID
keep-npu stop --all
keep-npu service-stop
```

The service exposes `GET /health`, `GET /api/npus`, and session CRUD under
`/api/sessions`. JSON-RPC is available at the exact `/rpc` endpoint, including
MCP-shaped `tools/list` and `tools/call` messages. As in KeepGPU 1.0, HTTP mode
is not a Streamable HTTP MCP endpoint; standards-based MCP transport is stdio.

## MCP server

For a local MCP client using stdio:

```console
keep-npu-mcp-server --mode stdio
```

For HTTP transport:

```console
keep-npu-mcp-server --mode http --host 127.0.0.1 --port 8765
```

The MCP tools are `start_keep`, `stop_keep`, `status`, and `list_npus`.

## Ascend device semantics

`--npu-ids` always addresses the visible ordinal used by `torch.npu`, after
`ASCEND_RT_VISIBLE_DEVICES` filtering. `list-npus` additionally reports the
physical ID when it can be derived safely. Memory comes from `torch.npu` with
best-effort `npu-smi` telemetry; unavailable values are returned as `null`
instead of being guessed.

See [KeepGPU compatibility](https://github.com/pzheng460/keep-npu/blob/main/docs/compatibility.md)
for the exact public-name mapping and intentional Ascend differences.

## Development

```console
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m pytest
cd web/dashboard && npm ci && npm test -- --run && npm run build
```

Hardware tests use the `ascend` pytest marker and conservative memory sizes.
The original standalone `keep_npu_alive.py` entry point remains available for
backward compatibility.
