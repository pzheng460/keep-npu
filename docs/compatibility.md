# KeepGPU compatibility

KeepNPU 1.0 follows KeepGPU 1.0's interaction model while translating the
device backend from CUDA/ROCm/MPS to Huawei Ascend.

| KeepGPU public contract | KeepNPU equivalent |
| --- | --- |
| `keep-gpu` | `keep-npu` |
| `keep-gpu-mcp-server` | `keep-npu-mcp-server` |
| `--gpu-ids` | `--npu-ids` |
| `list-gpus` / `/api/gpus` | `list-npus` / `/api/npus` |
| `gpu_ids` / `gpus` payload fields | `npu_ids` / `npus` |
| `~/.keepgpu` | `~/.keepnpu` |
| CUDA, ROCm, or MPS backend | Ascend `torch_npu` backend |

The blocking CLI, background service commands, job lifecycle, REST resources,
JSON-RPC methods, MCP tools, validation limits, dashboard flow, exit handling,
and utilization-backoff policy otherwise retain the KeepGPU behavior.

Intentional backend differences:

- `torch_npu` is imported lazily so help, validation, and service health remain
  usable on a machine without the Ascend runtime.
- Device selectors are torch-visible ordinals. This avoids confusing a physical
  card/chip ID with the post-filter ordinal accepted by `torch.npu.set_device`.
- `npu-smi` formats differ across Ascend products and driver versions. KeepNPU
  treats telemetry as best effort and returns nullable metrics when a value
  cannot be established safely.
- KeepNPU defaults to a mixed Ascend workload: preallocated FP16 matrix
  multiplication drives AI Core/Cube while independent Vector operations drive
  AI Vector and HBM bandwidth concurrently. `--vram` remains one capacity
  budget shared by both engines, not a requested bandwidth percentage. The
  explicit `--workload aicore` and `--workload vector` modes retain the
  compatibility paths. The opt-in `--workload random` mode requires roughly
  224 MiB at minimum and varies Cube-, Vector-, HBM-intensive, and balanced
  pressure phases every 10–60 seconds while keeping the requested allocation
  resident. Public workload values are `mixed`, `aicore`, `vector`, and
  `random`. Busy-threshold decisions use Ascend's total NPU utilization,
  including AI Vector work.
- Hardware/vendor setup is not declared as a PyPI dependency because PyTorch,
  `torch_npu`, CANN, and the driver must be installed as a compatible set.
