# Default AI Core Workload Design

## Goal

Change KeepNPU's default keepalive operation so a normal `keep-npu` invocation
drives Ascend AI Core/Cube utilization, which is the value displayed as `UTL`
by `nputop`. Users must not need a workload flag for this default. An optional
`workload` setting exposes the old Vector/ReLU behavior only when explicitly
selected.

The target command remains:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --interval 0.001 --busy-threshold -1
```

On supported Ascend hardware, this command should continuously drive the
selected devices' `nputop` UTL toward 100% while holding approximately the
requested memory on each device.

## Chosen Approach

The single-device controller will remove its elementwise ReLU workload and use
a preallocated FP16 matrix-multiplication workload. Ascend executes matrix
multiplication on the Cube/AI Core pipeline. ReLU primarily exercises the AI
Vector pipeline, which common NPU utilization views do not count as AI Core
utilization and which can leave `nputop` UTL at zero; it is therefore not a
meaningful KeepNPU workload.

The matrix workload is the default. ReLU is retained only as the explicit
`--workload vector` compatibility mode and is never selected as an automatic
fallback. Users do not write `aicore` in the normal command. This keeps the
ordinary user flow aligned with KeepGPU while making the default backend
operation appropriate for the metric Ascend users observe.

Alternatives rejected:

- Keeping ReLU as the default and requiring `--workload aicore` would preserve
  the old implementation but would make the desired NPU behavior opt-in.
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
matrices), AI Core startup fails with a concise actionable error. It must not
silently fall back to the vector workload because that would again report zero
`nputop` UTL. Explicit vector mode retains the existing four-byte minimum.

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

Blocking CLI and `start` accept `--workload` with exactly `aicore` and `vector`
as valid values; it defaults to `aicore`. Service calls, REST, JSON-RPC, and MCP
accept the matching optional `workload` field with the same validation and
default. Session records report the normalized value. The dashboard provides a
selector whose initial value is AI Core. Existing callers that omit the field
therefore receive the new AI Core behavior without changing their request.

The interfaces accept and report:

- `npu_ids`
- `vram`
- `interval`
- `busy_threshold`
- `workload`
- optional job identifiers where already supported

Help text and documentation will clarify that the default keepalive workload
uses AI Core/Cube matrix multiplication, `--workload vector` is a lightweight
opt-in whose activity may not appear in `nputop` UTL, and
`--busy-threshold -1` is the mode for unconditional maximum utilization. The
documented maximum-UTL command for devices 4-7 will not contain a workload
selector.

## Testing

Local tests will be written before implementation and will cover:

- default batches invoke `torch.matmul`, not `torch.relu_`;
- explicit vector batches retain the existing ReLU behavior;
- invalid workload values fail consistently through every public interface;
- matrix and filler allocation stay within the requested byte budget;
- dimensions are aligned and capped;
- too-small budgets fail clearly;
- stop requests bound the number of queued operations;
- busy-threshold gating and unconditional mode retain their semantics;
- only the selected device is set and used;
- allocation, execution, and release failures retain existing lifecycle rules;
- omitted workload values normalize to `aicore` through every public interface.

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

Explicit vector mode will also receive a short smoke test confirming that it
runs and releases, but its AI Core/`nputop` UTL is not an acceptance metric.

If a 1 GiB budget cannot sustain the acceptance target with the initial
4096-element matrix cap or 32-operation batch, those internal constants may be
tuned using the same tests. No new public flag will be introduced for that
tuning.

## Delivery

After local and hardware verification, update the package patch version,
documentation, changelog/release notes, and validation record. Publish the
wheel and source distribution so a user can upgrade with plain pip and run the
unchanged command directly.
