# Default AI Core Workload Design

## Goal

Change KeepNPU's default keepalive operation so a normal `keep-npu` invocation
drives Ascend AI Core/Cube utilization, which is the value displayed as `UTL`
by `nputop`. Users must not need a new workload flag. Existing public CLI,
service, REST, JSON-RPC, MCP, and dashboard parameters remain unchanged.

The target command remains:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --interval 0.001 --busy-threshold -1
```

On supported Ascend hardware, this command should continuously drive the
selected devices' `nputop` UTL toward 100% while holding approximately the
requested memory on each device.

## Chosen Approach

The single-device controller will replace its elementwise ReLU workload with a
preallocated FP16 matrix-multiplication workload. Ascend executes matrix
multiplication on the Cube/AI Core pipeline, whereas ReLU primarily exercises
the AI Vector pipeline and can leave `nputop` UTL at zero.

No `--workload` option will be added. The matrix workload is the default and
only public behavior. This keeps the user interface aligned with KeepGPU while
making the backend operation appropriate for the metric Ascend users observe.

Alternatives rejected:

- Keeping ReLU as the default and adding `--workload aicore` would preserve the
  old implementation but would make the desired NPU behavior opt-in and add an
  NPU-only public option.
- Running a separate synthetic benchmark process would complicate lifecycle,
  device isolation, error propagation, and cleanup.
- Using a fixed matrix size without accounting for `--vram` could exceed small
  memory requests or under-allocate large ones.

## Controller Architecture

The controller will keep allocation planning separate from execution:

1. Convert the existing per-device `--vram` value to a byte budget.
2. Choose the largest practical square FP16 matrix dimension, aligned to 16
   elements for Cube execution and initially capped at 4096 to prevent one
   operation from becoming unresponsive.
3. Allocate two input matrices and one reusable output matrix on only the
   controller's selected `npu:<visible ordinal>`.
4. Allocate inert filler tensors for the remainder of the requested memory.
5. Repeatedly call `torch.matmul(left, right, out=output)` and synchronize at
   bounded batch boundaries.

The three matrix buffers are part of the public memory target, not additional
unaccounted memory. Filler allocation continues to use bounded chunks. Small
rounding differences caused by tensor element size are permitted only up to one
allocation element; the controller must never intentionally allocate a second
full copy of the requested budget.

If the requested budget is smaller than 1536 bytes (three 16-by-16 FP16
matrices), startup fails with a concise actionable error. It must not silently
fall back to the vector workload because that would again report zero `nputop`
UTL.

## Runtime Semantics

Existing utilization-backoff behavior remains intact:

- `--busy-threshold -1` skips utilization gating and runs the matrix workload
  continuously, subject only to bounded synchronization and stop checks.
- A threshold from 0 through 100 queries total NPU utilization before work. If
  the device is at or above the threshold, or telemetry is unavailable, the
  worker waits for `--interval` before checking again.
- Once allowed to run, an initial batch of 32 matrix multiplications avoids
  per-operation telemetry subprocesses and minimizes idle gaps. The next
  utilization decision occurs at the existing controller-loop boundary.

`--interval` remains the backoff/check interval. It is not redefined as a delay
between every matrix multiply. This distinction is necessary for sustained
UTL. Stop requests are checked inside each bounded compute burst, so release
latency does not depend on completing an unbounded queue of operations.

The existing default busy threshold remains unchanged for KeepGPU-compatible
safe coexistence. Users explicitly requesting maximum sustained UTL use
`--busy-threshold -1`.

## Device Isolation and Lifecycle

Each worker calls `torch.npu.set_device` only for its assigned visible ordinal,
and every tensor is allocated with that device object. Enumeration and
telemetry must not activate unselected devices. Starting controllers for
ordinals 4, 5, 6, and 7 must therefore create KeepNPU processes/contexts only
on those four visible devices.

Allocation remains transactional across devices. An allocation or runtime
failure is surfaced through the existing controller health path; already
started controllers are released during startup rollback. Normal stop,
SIGINT, and SIGTERM synchronize as needed, drop all matrix and filler
references, empty the backend cache, and terminate worker threads.

Out-of-memory handling retains the current retry behavior for transient
pressure. Non-OOM allocation errors and matrix-execution failures become clean
runtime errors instead of raw Rich tracebacks.

## Public Compatibility

There is no schema change. Blocking CLI, `start`, service session records,
REST, JSON-RPC, MCP, and dashboard continue to accept and report:

- `npu_ids`
- `vram`
- `interval`
- `busy_threshold`
- optional job identifiers where already supported

Help text and documentation will clarify that the keepalive workload uses AI
Core/Cube matrix multiplication and that `--busy-threshold -1` is the mode for
unconditional maximum utilization. The documented command for devices 4-7
will not contain an Ascend workload selector.

## Testing

Local tests will be written before implementation and will cover:

- default batches invoke `torch.matmul`, not `torch.relu_`;
- matrix and filler allocation stay within the requested byte budget;
- dimensions are aligned and capped;
- too-small budgets fail clearly;
- stop requests bound the number of queued operations;
- busy-threshold gating and unconditional mode retain their semantics;
- only the selected device is set and used;
- allocation, execution, and release failures retain existing lifecycle rules;
- CLI and service contracts do not gain or require a workload parameter.

The full existing test suite, package build, and installed-entry-point smoke
tests must pass.

## Real-Hardware Acceptance

Testing will use the previously validated Ascend 910B2 environment through SSH
without changing its shared Python environment. A unique temporary deployment
will be installed and removed after testing.

For at least one idle device and then devices 4, 5, 6, and 7 together, record
before/during/after evidence from `npu-smi` and the metric source used by
`nputop`. Acceptance requires:

1. Selected devices sustain `nputop` UTL near 100% after warmup. Because
   sampling windows fluctuate, success means repeated samples predominantly at
   least 95%, with any lower sample documented.
2. Selected devices hold approximately the requested memory, allowing runtime
   and allocator overhead.
3. Unselected devices gain no KeepNPU process or memory allocation.
4. Ctrl+C/SIGTERM and normal stop remove KeepNPU processes and return memory to
   its pre-test range.
5. Default safe-threshold mode still backs off when total utilization is at or
   above its threshold, while `-1` runs unconditionally.

If a 1 GiB budget cannot sustain the acceptance target with the initial
4096-element matrix cap or 32-operation batch, those internal constants may be
tuned using the same tests. No new public flag will be introduced for that
tuning.

## Delivery

After local and hardware verification, update the package patch version,
documentation, changelog/release notes, and validation record. Publish the
wheel and source distribution so a user can upgrade with plain pip and run the
unchanged command directly.
