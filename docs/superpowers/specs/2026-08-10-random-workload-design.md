# Random Workload Design

## Goal

KeepNPU will add an explicit `random` workload that makes Ascend utilization
look like a changing application rather than a constant synthetic saturation
test. It will vary Cube, Vector, and HBM pressure through recognizable workload
phases while keeping the requested HBM allocation resident.

The feature is opt-in:

```console
keep-npu --workload random --npu-ids 0,1 --vram 1GiB \
  --interval 60 --busy-threshold -1
```

The default remains `mixed`. Existing `mixed`, `aicore`, and `vector` behavior
does not change.

## Public Surface

`random` becomes the fourth accepted workload value everywhere workload is
public:

- blocking CLI;
- `keep-npu start` and the service JSON protocol;
- MCP tools and their schemas;
- Dashboard workload selection;
- shared validation and serialized session state.

Unknown workload values retain the existing validation failure style. Random
mode requires enough HBM for its three active regions. Validation happens
before probing Ascend hardware and reports a workload-specific minimum. The
minimum is three aligned FP16 matrices plus 224 MiB of active storage, comprising
64 MiB for Vector and 160 MiB for HBM traffic.

No public random-seed option is added. Production uses a system-seeded random
generator, while tests inject a deterministic generator through an internal
interface.

## Allocation Plan

Random mode allocates one fixed tensor plan at startup and retains it across
all phase transitions:

1. Three aligned FP16 matrices drive Cube through matrix multiplication.
2. A 64 MiB float32 working set drives a compute-oriented Vector operator.
3. Two HBM buffers with a combined size of 160 MiB drive independent memory
   traffic.
4. Any remaining requested HBM is held in empty reserve tensors.

The matrix dimension is calculated from the remaining budget after reserving
the Vector and HBM regions. It uses the existing alignment and bounded planning
style. The complete plan never exceeds the requested public HBM budget, apart
from vendor-runtime workspace outside KeepNPU's tensor accounting.

Reserve tensors remain resident but are not traversed on every operation. Phase
changes update scheduler targets only; they never reallocate tensors or create
new streams.

## Runtime Architecture

Random mode uses three long-lived feeder threads and three distinct Ascend
streams:

- the Cube feeder submits FP16 matrix multiplication;
- the Vector feeder submits a compute-oriented elementwise Vector operator on
  the 64 MiB working set;
- the HBM feeder moves data between its two buffers to create bandwidth
  pressure independently of the Vector working set.

A coordinator owns the current phase, phase deadline, target duty cycles, and
ramp state. Feeders read immutable scheduler snapshots. They do not mutate the
coordinator or call telemetry themselves.

Duty-cycle control uses short bounded scheduling quanta. During each quantum a
feeder submits work for its active share and waits interruptibly for the
remainder. Queue depth remains bounded so SIGINT and SIGTERM cannot leave a
large asynchronous backlog.

This is open-loop workload shaping. Target percentages describe feeder duty
cycles and trends, not guaranteed one-to-one `npu-smi` readings. Hardware
generation, health, power limits, temperature, and operator implementation may
change the observed values.

## Phase Model

Each phase lasts a uniformly randomized 10 to 60 seconds. Four profiles are
selected with equal probability:

| Profile | Cube duty cycle | Vector duty cycle | HBM duty cycle |
| --- | ---: | ---: | ---: |
| Cube intensive | 85–100% | 20–45% | 35–65% |
| Vector intensive | 25–55% | 80–100% | 40–70% |
| HBM intensive | 20–50% | 35–65% | 85–100% |
| Balanced | 55–85% | 55–85% | 55–85% |

The same profile is not selected for two consecutive phases. Upon entering a
profile, the coordinator independently samples a target inside each range.
Targets transition from the prior phase over a randomized 2 to 5 seconds,
bounded to no more than one quarter of the new phase duration. Linear
interpolation provides a stable, explainable ramp without abrupt utilization
steps.

The first phase starts from zero duty cycle and uses the same ramp behavior.
This avoids a startup spike while leaving the full requested HBM allocation
resident as soon as allocation completes.

## Busy Threshold and Interval Semantics

`--busy-threshold -1` disables utilization backoff and runs the random phase
scheduler continuously.

With a non-negative threshold, the existing total-NPU utilization check remains
authoritative. When external utilization is above the threshold or telemetry is
unavailable, all three random feeders pause together. The current phase clock
also pauses, so an externally busy period does not consume or skip phases. When
work resumes, the scheduler continues the same phase and ramp position.

`--interval` remains the utilization-check and backoff interval. It does not
control the 10–60 second random phase duration.

## Shutdown and Failure Handling

Random mode reuses the current bounded controller lifecycle:

1. SIGINT, SIGTERM, release, a feeder failure, or a runtime failure sets the
   shared stop/cancel event.
2. All waits are interruptible and all feeder queues are bounded.
3. Feeders share one shutdown deadline rather than receiving cumulative join
   timeouts.
4. Multi-device release stops workers in parallel and performs one process-wide
   allocator cache flush.

The first feeder failure records its engine name (`cube`, `vector`, or `hbm`),
cancels the other feeders, and reaches the existing allocation-status/runtime
error channel. Cleanup failures do not overwrite the original workload error.

Random scheduling does not write samples, profiles, or telemetry to disk.
Phase transitions are DEBUG-only logs, so normal long-running use does not grow
stdout or service log files from per-phase messages.

## Validation

Unit and integration tests cover:

- accepting `random` and rejecting unknown workloads through every public
  interface;
- rejecting insufficient random-mode HBM before hardware enumeration;
- exact allocation accounting for matrices, Vector, HBM buffers, and reserve;
- one-time allocation with no tensor or stream creation during phase changes;
- deterministic profile selection through an injected random generator;
- 10–60 second phase durations, 2–5 second bounded ramps, target ranges, equal
  profile weights, and no immediately repeated profile;
- three distinct streams and correct operation routing;
- pausing all feeders and the phase clock under busy-threshold backoff;
- bounded queueing, feeder failure propagation, SIGTERM, and multi-device
  release;
- CLI, service, MCP, Dashboard, README, and help-text parity.

Remote Ascend validation uses `npu-smi info -t usages` across several phase
transitions. Acceptance requires:

- the requested HBM allocation remains resident across all phases;
- each named intensive phase makes its corresponding metric visibly dominant
  relative to at least one other phase on the same device;
- balanced phases exercise all three resources concurrently;
- no unexpected allocation, process, or utilization appears on unselected
  devices;
- SIGTERM returns selected devices to their pre-test memory range with no
  KeepNPU process left behind.

Remote validation records observed counters rather than requiring every profile
to hit exact percentages.

## Non-goals

- Changing the default `mixed` workload.
- Mimicking a specific training model or trace.
- Closed-loop control of exact `npu-smi` utilization percentages.
- Persisting random schedules or replaying a seed through the public CLI.
- Adding adaptive temperature or power-limit control.
