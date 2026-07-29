# Ascend remote validation — 2026-07-21

## Scope

Validation used SSH aliases from the local SSH configuration with bounded,
non-interactive commands. `npu0` and `npu1` resolve to the same 8-device 910B2
host, while `npu2` and `npu3` are separate 8-device 910B1 hosts. `npu4` was not
reachable during the run. No shared Python environment or system package was
modified.

The usable hardware environment was:

- `npu-smi 25.3.rc1`, eight Ascend 910B2 devices with 64 GiB HBM each
- Python 3.11 environment at an existing project-local path
- PyTorch 2.7.1 and `torch_npu` 2.7.1.post2
- eight devices reported by `torch.npu.device_count()`

The two 910B1 hosts ran `npu-smi 25.5.0`. Their login Python environments did
not include PyTorch; `npu2` containers exposed `torch_npu` 2.9 and 2.10, but no
production container was modified for this test.

## Hardware discrepancy and correction

All three physical hosts emit Ascend 25.x's two-line device records: the first
line contains NPU ID/model/health, and the second contains chip, AICore, and HBM
fields. The original one-line parser returned no records. The captured format
was converted into the regression
`test_parse_npu_smi_output_accepts_ascend_25_two_line_records` before the parser
was changed.

After correction, live output produced all eight ordered physical IDs on every
host:

| SSH alias | Model | Parsed devices | Parsed utilization |
| --- | --- | ---: | --- |
| `npu0` | 910B2 | 8 | eight numeric values |
| `npu2` | 910B1 | 8 | eight numeric values, including an active device |
| `npu3` | 910B1 | 8 | eight numeric values |

On 910B2, `list_npus()` also returned eight selectable torch-visible ordinals,
physical IDs 0–7, 64 GiB totals, live HBM usage, and numeric utilization.

## Lifecycle results

Tests deliberately avoided devices 0–3 because existing workloads occupied
them.

- Multi-device controller: IDs 4, 5, and 6, 16 MiB per device, 0.2 s interval,
  unconditional backoff override, two-second run. All three workers started
  with no allocation error, reported no runtime error, and released with no
  worker thread remaining.
- Blocking CLI: ID 7, 16 MiB, 0.2 s interval. An injected SIGINT ended the
  foreground command with signal status 130; subsequent `npu-smi` showed no
  KeepNPU process on the device.
- REST/dashboard: `/health` returned `{"ok": true}`, `/api/npus` returned all
  eight Ascend records, and `/` returned the packaged KeepNPU dashboard.
- REST session lifecycle: job `ssh-smoke` started on ID 7, reached `active`, and
  `DELETE /api/sessions/ssh-smoke` reported it in `stopped` with empty
  `timed_out`, `failed`, and `errors` fields.
- JSON-RPC: direct `list_npus` at `/rpc` returned eight records; MCP-shaped
  `tools/call` for `list_npus` returned `isError: false`; invalid ordinal 99 was
  rejected with JSON-RPC code `-32602` before controller startup.

## Cleanup

The post-test audit found and terminated two foreground HTTP test processes
whose initial shell wrappers had exited without forwarding their PID. A second
audit confirmed:

- no process command line referenced either unique test directory;
- no `keep-npu` process appeared in `npu-smi`;
- test ports 18765, 18766, and 18767 had no listener;
- devices 4–7 had no running process in `npu-smi`; and
- both `/tmp/keep-npu-real.RCXhc2` and
  `/tmp/keep-npu-protocol.IhxqLU` were removed.

This cleanup issue belonged to the ad-hoc SSH shell wrapper, not KeepNPU's
tracked daemon workflow. The final protocol check captured the Python PID
directly and exited cleanly.

## Post-review hardware recheck

After final review reduced the Ascend default workload to one pass, added
coordinated SIGTERM handling, and hardened HTTP request headers, the affected
paths were redeployed once more:

- a default-iteration controller on device 7 allocated 16 MiB, ran without a
  worker error, and released;
- `kill -TERM` during blocking startup was caught, logged as an interruption,
  released through the controller context, and exited with status 0; and
- a cross-origin `text/plain` POST to `/api/sessions` was rejected with HTTP
  415 and did not create a session.

The `/tmp/keep-npu-final.Q5XXLo` deployment was removed after `pgrep`,
`npu-smi`, and the test-port audit found no residual KeepNPU process or
listener.

## Busy-threshold alignment recheck

The KeepGPU-compatible 5,000-pass ReLU batch was restored and tested on NPU 7
with a 1 GiB allocation. During the batch, `npu-smi info -t usages -i 7`
reported 99% AI Vector usage, 100% HBM bandwidth usage, and 100% total NPU
utilization while the summary table's `AICore(%)` remained zero.

KeepNPU now reads the total NPU utilization metric for backoff. The deployed
monitor returned 100%; a busy threshold of 25 rejected another keep-alive
batch, while the explicit `-1` mode allowed it. The test process was terminated
with SIGTERM and its temporary deployment directory was removed.

## Default AI Core workload validation

Version 1.0.3 was installed from its locally built wheel into an isolated
`--system-site-packages` virtual environment on the 910B2 host. The shared
Ascend Python environment was not modified. Testing used the public command
without an explicit workload option:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --interval 0.001 --busy-threshold -1
```

Two hardware-specific bottlenecks were found and corrected. Unconditional mode
had still queried `npu-smi` before every batch, and the controller had slept for
one millisecond after every completed batch. Skipping telemetry and inter-batch
sleep when the threshold is `-1` raised repeated AI Core readings from zero,
then 87%, to the final result below. A matrix-size sweep found 8192 to be the
best tested cap: 4096 reached 94–95%, while 12288 regressed to 92–94%.

Five consecutive samples from the final four-device run were:

| NPU | AI Core utilization | Total NPU utilization |
| --- | --- | --- |
| 4 | 96, 96, 96, 96, 96% | 100% in every sample |
| 5 | 95, 95, 95, 95, 95% | 100% in every sample |
| 6 | 96, 96, 96, 96, 96% | 100% in every sample |
| 7 | 96, 96, 96, 97, 96% | 100% in every sample |

`npu-smi info` associated the KeepNPU PID only with devices 4–7, with one
roughly 1180 MiB context on each selected device. Devices 0–3 retained only
their pre-existing Ray/VLLM processes; KeepNPU created no process or allocation
on them.

The explicit `--workload vector` smoke test produced 99% AI Vector and 100%
total utilization. Vector execution now synchronizes its asynchronous queue in
bounded groups, so SIGTERM released the 1 GiB allocation cleanly without the
previous two-second worker shutdown error. A final post-stop sample showed zero
AI Core, AI Vector, and total utilization on devices 4–7.

## Default mixed Cube, Vector, and HBM validation — 2026-07-29

The release-candidate source for 1.0.4 was built before the version bump as a
local 1.0.3 wheel (`sha256
9691959f7102d2f296b54710877f994302189533d2cda76aca528ded89affd`). That wheel
was installed into isolated `--system-site-packages` virtual environments.
Neither shared Ascend Python environment was modified. The tested command
omitted `--workload`, so it exercised the public default:

```console
keep-npu --npu-ids 0 --vram 1GiB --interval 60 --busy-threshold -1
```

On the healthy 910B2 host, the mixed planner used 12288-by-12288 FP16 matrices,
160 MiB of Vector storage, one queued Cube operation, and 64 queued Vector
operations. Unconditional mode retained its feeder threads for 60 seconds,
avoiding the utilization gap caused by rebuilding them every second. Ten
consecutive post-warm-up samples from the installed wheel were:

| Metric | Ten consecutive readings |
| --- | --- |
| AI Core | 99, 99, 99, 99, 99, 99, 99, 99, 99, 99% |
| AI Vector | 95, 95, 95, 95, 94, 95, 95, 95, 95, 95% |
| HBM bandwidth | 99, 99, 99, 99, 99, 99, 99, 99, 99, 99% |
| Total NPU utilization | 100% in every sample |

The same wheel also started successfully on four idle 910B1 devices 4–7.
Across ten samples per device it sustained 94–95% AI Core, 99–100% HBM
bandwidth, 100% total utilization, and 84–90% AI Vector. The lower Vector
counter was specific to the 910B1 host, whose devices reported Warning health;
the strict simultaneous 90% acceptance was therefore established on the
healthy 910B2 target.

The first 910B1 deployment exposed a separate startup bug: fresh
`torch.npu.set_device` context creation took 8.68 seconds while mixed tensor
allocation took 0.03 seconds, so the previous fixed five-second startup wait
rejected normal initialization. The bounded startup wait is now 30 seconds.

SIGTERM stopped both final runs cleanly. Post-stop telemetry reported zero AI
Core, AI Vector, HBM bandwidth, and total utilization on every selected device,
with HBM usage back at the five-percent baseline. All uniquely named wheel and
virtual-environment directories were removed from both hosts and the test
container; no KeepNPU test process remained.

## Large mixed allocation and release regression — 2026-07-29

The large-memory regression used the public default workload and the exact
reported command shape:

```console
keep-npu --npu-ids 0,1,4,5,6,7 --vram 56GiB \
  --interval 0.001 --busy-threshold -1
```

Mixed allocations now distinguish a bounded 160 MiB active Vector working set
from empty reserve tensors. The reserve still occupies the full requested HBM
budget, but is not traversed by every Vector queue iteration. On a healthy
910B2 device, ten consecutive `npu-smi info -t usages` samples reported
99–100% AI Core, 96% AI Vector, 100% HBM bandwidth, and 100% total NPU
utilization while HBM usage remained at 92%.

The six-device 910B1 run associated the KeepNPU PID only with devices
0, 1, 4, 5, 6, and 7, at roughly 57.5 GiB per device. Devices 2 and 3 retained
only their pre-existing VLLM processes. Three consecutive rounds reported
94–96% AI Core, 87–90% AI Vector, 99–100% HBM bandwidth, and 100% total NPU
utilization on every selected device. All six 910B1 devices reported
`Health=Warning`; the same workload reached the higher counters above on the
healthy 910B2 device.

Release supervision now gives the mixed feeders a consistent bounded shutdown
window. Multi-device release stops all workers in parallel and performs one
process-wide allocator cache flush instead of six redundant concurrent flushes.
