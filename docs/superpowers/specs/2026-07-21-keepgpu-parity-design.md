# KeepNPU–KeepGPU 1.0 Compatibility Design

## Background

The current repository contains a single foreground `torch_npu` matrix-multiply
loop. It demonstrates the basic keepalive idea but does not implement the public
product surface of KeepGPU 1.0.0: package entry points, multi-device resource
targets, utilization backoff, service-managed sessions, telemetry, REST,
JSON-RPC, MCP, or the dashboard.

This project will turn the prototype into an Ascend-native KeepGPU port. The
reference contract is KeepGPU 1.0.0 plus compatibility-relevant fixes present on
the upstream `main` branch at commit `691720383f5325cffb9fc5960304541caec31444`.

## Product and Compatibility Boundary

KeepNPU uses NPU-native public names while preserving the corresponding
KeepGPU behavior:

| KeepGPU surface | KeepNPU surface |
| --- | --- |
| `keep-gpu` | `keep-npu` |
| `keep-gpu-mcp-server` | `keep-npu-mcp-server` |
| `--gpu-ids` | `--npu-ids` |
| `list-gpus` | `list-npus` |
| `gpu_ids` | `npu_ids` |
| `gpus` | `npus` |
| `~/.keepgpu` | `~/.keepnpu` |

Apart from this explicit naming map and unavoidable Ascend-specific device or
runtime diagnostics, the following behavior must remain equivalent:

- CLI hierarchy, defaults, exit status, help structure, and validation limits.
- Blocking and service-driven workflows.
- Session lifecycle and JSON result shapes.
- REST, JSON-RPC, and MCP method semantics.
- Service auto-start, ownership verification, and shutdown behavior.
- Dashboard layout, controls, refresh behavior, and lifecycle presentation.
- Multi-device rollback, failure retention, and cleanup behavior.

The service listens on `127.0.0.1:8765` by default. Machine interfaces use
NPU-native field names only; mixed GPU/NPU aliases are deliberately excluded to
avoid maintaining two public protocols.

## Architecture

The implementation follows the KeepGPU 1.0 component boundaries so upstream
behavior can be audited and ported without inventing a second architecture:

1. `keep_npu.utilities` owns validation, size parsing, endpoint validation,
   strict JSON decoding, platform discovery, visible-device mapping, telemetry,
   and logging.
2. `keep_npu.single_npu_controller` owns one Ascend device's allocation,
   backoff, keepalive worker, runtime health, and release lifecycle.
3. `keep_npu.global_npu_controller` validates the public selection and
   coordinates multiple controllers with rollback and parallel release.
4. `keep_npu.cli` exposes blocking mode and the `serve`, `start`, `status`,
   `stop`, `list-npus`, and `service-stop` commands.
5. `keep_npu.mcp.server` owns the shared session registry and exposes the same
   operations through stdio MCP, HTTP JSON-RPC, REST, and the dashboard.
6. `web/dashboard` contains the editable React source; built assets are shipped
   under `keep_npu.mcp.static`.

The legacy `keep_npu_alive.py` remains as a compatibility wrapper for one
release cycle. Its old flags translate into the new blocking controller when
possible and its help points users to `keep-npu`.

## Controller and Data Flow

### Blocking mode

The CLI validates all local input before importing hardware libraries. It then
resolves visible NPU ordinals, creates one controller per target, and starts the
workers transactionally. Each worker queries utilization before allocating or
computing. With a non-negative busy threshold, utilization above the threshold
or unavailable telemetry causes conservative backoff. `-1` explicitly enables
unconditional keepalive work. Ctrl+C or SIGTERM requests a coordinated release.

### Workload and memory target

`--vram` is a per-NPU byte target and accepts KeepGPU-compatible human sizes,
including `512MB`, `1GiB`, and bare bytes. The controller converts bytes to
float32 element counts, rounds up when required, and splits allocations into
bounded chunks so a single tensor does not exceed backend indexing limits. It
uses preallocated, low-cost tensor operations and explicit `torch.npu`
synchronization. Internal workspace memory is not counted as part of the public
target.

### Service mode

`keep-npu start` validates locally, auto-starts the local service when allowed,
and calls `start_keep`. The service records `starting` before controller
creation so concurrent status calls do not mistake an in-progress start for an
absent session. Tracked states are `starting`, `active`, `stopping`,
`runtime_failed`, and `stop_failed`. Terminal runtime and release failures retain
the session record and `last_error` until the user stops or retries it.

Dashboard, REST, direct JSON-RPC, standard MCP `tools/call`, and the CLI service
commands all invoke the same service object and validators.

## Ascend Discovery and Telemetry

`torch_npu` is the source of truth for whether a visible ordinal can be selected.
The implementation uses the supported `torch.npu.device_count()`, device
selection, synchronization, and memory APIs available in the installed Ascend
PyTorch version.

`npu-smi info` is the preferred source for physical identity, model name,
utilization, total memory, and used memory. Because output varies across CANN
and driver releases, parsing is isolated behind a monitor interface and backed
by fixture tests from each remote environment. Visible ordinal remains the
public identifier; physical chip/card identifiers are metadata only.

Missing commands, unsupported output, inconsistent counters, and out-of-range
utilization produce `null` telemetry. They never produce an idle-looking zero.
When non-negative utilization backoff is active, unknown utilization prevents
allocation and compute.

## Public Interfaces

### CLI

Blocking mode accepts `--interval`, `--npu-ids`, `--vram`,
`--busy-threshold`/`--util-threshold`, and the deprecated `--threshold` alias.
The service commands and endpoint flags match KeepGPU's hierarchy and defaults.
`status`, `stop`, and `list-npus` write exactly one plain JSON object without
ANSI decoration, including error results after CLI parsing succeeds.

### Service APIs

The direct methods and MCP tools are:

- `start_keep(npu_ids?, vram?, interval?, busy_threshold?, job_id?)`
- `stop_keep(job_id?)`
- `status(job_id?)`
- `list_npus()`

REST endpoints are:

- `GET /api/npus`
- `GET /api/sessions`
- `GET /api/sessions/{job_id}`
- `POST /api/sessions`
- `DELETE /api/sessions`
- `DELETE /api/sessions/{job_id}`

`POST /` and `POST /rpc` implement JSON-RPC. Stdio implements MCP lifecycle
methods including `initialize`, `notifications/initialized`, `ping`,
`tools/list`, and `tools/call`, while retaining direct-method compatibility.

### Dashboard

The dashboard retains the KeepGPU layout and responsive behavior with NPU
terminology. It shows detected NPUs, tracked sessions, average utilization, a
start form for NPU IDs/VRAM/interval/busy threshold, per-session release and
release-all controls, telemetry cards, manual refresh, opt-in auto refresh, and
operation messages. Unknown telemetry displays `n/a` and no utilization fill.

## Validation and Errors

Validation rejects non-finite or non-positive intervals, invalid or duplicate
visible ordinals, signed-zero and non-ASCII numeric spellings, invalid byte
sizes, thresholds outside `-1` or `0..100`, unsafe job IDs, invalid hosts, and
ports outside `1..65535`. Public JSON parsers reject `NaN`, infinities, and
oversized numbers that decode to non-finite floats.

Expected absence of Ascend hardware, unusable visible ordinals, missing
`torch_npu`, and failed enumeration are startup-unavailable errors. Unexpected
controller or protocol faults are internal errors. Partial multi-NPU startup
rolls back every controller already started. Release runs concurrently and
reports every failed rank.

The auto-started daemon is stored under `~/.keepnpu`. Shutdown signals a process
only after PID, UID, process start identity, endpoint, and command line match
the ownership record. A foreground service remains under terminal control.

## Testing and Acceptance

### Local tests

- Unit tests cover validation, byte conversion, visibility, telemetry parsing,
  controller timing, lifecycle, rollback, health, and release.
- Contract tests port KeepGPU inputs and expected results through the explicit
  GPU-to-NPU naming map.
- CLI tests cover help, exit codes, blocking options, service commands, JSON
  purity, daemon ownership, and malformed service responses.
- Server tests cover REST routing, strict JSON, JSON-RPC, MCP lifecycle and
  tools, concurrency, status retention, and static asset behavior.
- Dashboard tests cover payload creation, formatting, refresh rules, lifecycle
  rendering, and production asset synchronization.
- Hardware-facing code is dependency-injected so the full local suite runs
  without Ascend hardware.

### Remote Ascend tests

The SSH configuration is discovered read-only. Each reachable NPU host is
preflighted for `npu-smi`, Python, CANN environment, `torch`, `torch_npu`,
device count, and current memory pressure. Code is copied to a unique temporary
directory. No system package, shared environment, or persistent service is
modified.

On at least two environments, or on multiple visible NPUs if only one host is
available, validation covers:

1. Short blocking and one-shot work.
2. Requested memory allocation and release recovery.
3. Utilization telemetry and conservative backoff.
4. Multi-NPU selection and invalid selection rejection.
5. `start`, `status`, `stop`, `list-npus`, and `service-stop`.
6. REST, JSON-RPC, MCP, and dashboard API responses.
7. SIGINT/SIGTERM, repeated stop, startup rollback, and cleanup.

Every remote command has a timeout. Tests record versions, commands, selected
devices, before/during/after telemetry, exit status, and relevant logs. Any
temporary daemon is stopped and the temporary deployment is removed after
results are collected.

### Definition of done

The port is complete when:

- all local unit, contract, service, and dashboard tests pass;
- package build and installed console entry points pass smoke tests;
- the KeepGPU compatibility matrix has no unexplained behavioral differences;
- remote core lifecycle tests pass on the available Ascend environments;
- memory returns to its pre-test range and no test service or workload remains;
- Ascend-imposed differences are documented with reproducible evidence.

## Delivery Sequence

1. Package structure, validators, discovery, telemetry, and controller.
2. Blocking CLI and multi-NPU lifecycle.
3. Service registry and service CLI commands.
4. REST, JSON-RPC, and MCP.
5. Dashboard and packaged assets.
6. Full contract suite, packaging, and documentation.
7. SSH preflight, remote validation, hardware-specific corrections, and final
   compatibility report.
