# Default Mixed Cube, Vector, and HBM Workload Design

## Goal

KeepNPU's default workload will concurrently drive Ascend Cube/AI Core,
AI Vector, and HBM bandwidth. A normal command will not require a workload
flag:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --busy-threshold -1
```

On the validated Ascend 910B2 host, repeated samples from
`npu-smi info -t usages` must report at least 90% AICore/Cube utilization,
90% AI Vector utilization, and 90% HBM bandwidth utilization on every selected
device after warm-up. Validation must use multiple samples so a transient peak
does not count as success.

## Public Behavior

The public workload values will be:

- `mixed`: concurrent Cube, Vector, and HBM pressure; this becomes the default.
- `aicore`: Cube-focused compatibility mode.
- `vector`: Vector/HBM-focused compatibility mode.

Omitting `--workload` normalizes to `mixed` through the blocking CLI, service
CLI, REST API, JSON-RPC, MCP schema, Python controllers, status records, and
dashboard. Existing explicit `aicore` and `vector` requests retain their
current behavior.

`--vram` remains a per-NPU total allocation budget. Mixed mode does not
allocate the requested amount independently for each engine. Its Cube matrices
and Vector buffers must fit together inside the same budget.

`--busy-threshold -1` remains the documented maximum-pressure setting. A
non-negative busy threshold continues to pause the whole mixed workload when
the device is already busy or telemetry is unavailable.

## Alternatives Considered

### Two independent streams with bounded feeders

This is the selected approach. One stream continuously submits FP16 matrix
multiplication and another continuously submits memory-intensive Vector work.
Independent streams allow the Ascend scheduler to overlap work on separate
engines while bounded submission and synchronization preserve responsive
shutdown.

### One stream alternating matrix and Vector operations

This is simpler, but operations on one stream are ordered. It can create
alternating utilization peaks, not sustained simultaneous Cube and Vector
pressure, so it does not meet the acceptance criterion.

### Two independent KeepNPU processes on the same device

This could overlap engines, but duplicates device contexts, makes the HBM
budget ambiguous, complicates ownership and cleanup, and can leave one process
running after the other fails. It is not suitable as the default.

## Workload Planning and Allocation

Mixed mode will reuse the existing `AICoreAllocation` shape: three FP16
matrices for Cube work and float32 filler tensors for Vector/HBM work.
The planner will explicitly guarantee a non-empty Vector region rather than
treating all residual bytes as incidental.

The planner follows these rules:

1. The full allocation remains within the parsed `--vram` byte budget.
2. The Cube matrix dimension remains aligned to the existing hardware
   alignment and capped at the validated maximum.
3. The plan reserves enough Vector bytes to issue meaningful memory traffic.
4. If the budget cannot satisfy both minimum allocations, mixed mode fails
   before probing hardware with a clear workload-specific error.
5. Matrix and Vector tensors are disjoint, so concurrent in-place Vector
   operations cannot race with matrix multiplication.

The default `1GiB` budget leaves enough room for the validated 8192-dimension
Cube matrices and a large Vector buffer. Larger budgets increase the Vector
region after the Cube dimension reaches its cap.

## Runtime Architecture

Each `AscendNPUController` continues to own one lifecycle worker and one stop
event. In mixed mode the lifecycle worker allocates tensors, creates two
device-local streams, and starts two bounded feeder loops:

- the Cube feeder submits a bounded group of `torch.matmul(..., out=...)`
  calls to the Cube stream;
- the Vector feeder submits bounded in-place elementwise operations across
  the Vector buffers on the Vector stream.

Each feeder synchronizes only its own stream after a bounded group. This keeps
queues finite, surfaces asynchronous CANN failures close to their source, and
allows the stop event to be observed between groups. The lifecycle worker
waits for both feeders during release.

The implementation must not globally synchronize after each individual
operation because that would serialize the two engines. A global device
synchronization is allowed only during final cleanup after both feeders have
stopped.

Explicit `aicore` and `vector` modes keep their existing single-workload paths.

## Failure and Shutdown Semantics

Both feeders share a thread-safe first-failure channel. The first exception:

1. records the original engine name and exception text;
2. sets the shared stop event;
3. lets the sibling feeder finish its bounded group and stop;
4. joins both feeders; and
5. exposes one `runtime_error()` through the existing global controller.

The blocking CLI's worker supervision then releases every selected NPU,
prints the rank and original CANN error without a Rich traceback, and exits
non-zero. Service sessions transition to `runtime_failed` through the existing
runtime health path. Later cleanup errors must not replace the first workload
failure, though they may be logged separately.

SIGINT and SIGTERM continue to request an orderly stop. Release must remain
bounded even if one feeder is blocked in stream synchronization; the existing
controller timeout behavior remains the final safeguard.

## Interfaces and Compatibility

The normalized `mixed` value must be propagated through:

- shared workload validation and `DEFAULT_WORKLOAD`;
- single- and global-controller constructors;
- blocking and service CLI options and help;
- REST, JSON-RPC, and MCP input schemas;
- service status records;
- dashboard form validation, payloads, and labels;
- README examples and workload descriptions.

Machine-readable clients receive `workload: "mixed"` when they omit the
field. This is an intentional default behavior change for the next release.
Explicit requests remain backward compatible.

## Testing

Pure planning tests will verify:

- `mixed` is the default and all three public values validate;
- mixed matrix and Vector allocations are non-empty, aligned, disjoint, and
  within budget;
- too-small mixed budgets fail before hardware enumeration.

Controller tests with a fake torch backend will verify:

- mixed mode creates two distinct streams;
- Cube operations execute in the Cube stream;
- Vector operations execute in the Vector stream;
- both feeders begin work before release;
- a failure in either feeder stops the sibling and reaches
  `allocation_status()`;
- release joins both feeders and clears cached HBM;
- explicit `aicore` and `vector` behavior remains unchanged.

CLI, protocol, service, and dashboard tests will verify the new default,
accepted enum, normalized status payload, help text, and malformed-value
handling.

The full existing suite, Ruff, and formatting checks must pass.

## Hardware Validation

Build a wheel from the exact candidate commit and install it into an isolated
virtual environment on the 910B2 host. Use idle devices only and run the normal
command without `--workload`.

After warm-up, collect at least ten usage samples per selected device. Every
selected device must sustain at least 90% for AICore/Cube, AI Vector, and HBM
bandwidth in the same sampling window. Record the raw samples and candidate
commit in `docs/validation/ascend-remote-results.md`.

Finally, send SIGTERM and verify:

- the command exits within the controller's bounded shutdown period;
- HBM returns to its pre-test range;
- no KeepNPU process remains on the selected devices; and
- no runtime worker failure was reported.
