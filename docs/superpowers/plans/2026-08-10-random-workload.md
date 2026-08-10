# Random Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--workload random` mode that keeps requested HBM resident while varying Cube, Vector, and HBM pressure through natural 10–60 second workload phases.

**Architecture:** Add a pure random-memory planner and a hardware-independent phase scheduler. The Ascend controller will allocate fixed Cube, Vector, HBM, and reserve tensors once, run three bounded feeders on independent streams, and let one coordinator update feeder duty cycles without polling telemetry from feeder threads. The new public value then propagates through CLI, service, MCP, Dashboard, packaged assets, and documentation while `mixed` remains the default.

**Tech Stack:** Python 3.9–3.13, `dataclasses`, `random.Random`, `threading`, PyTorch with `torch_npu`, Typer, pytest, React 18, Vitest, Vite, and `npu-smi` on Ascend 910B1/910B2.

## Global Constraints

- Public workload values are exactly `mixed`, `aicore`, `vector`, and `random`.
- `DEFAULT_WORKLOAD` remains exactly `mixed`.
- Random mode is selected only by `--workload random`; existing workload behavior is unchanged.
- Each phase lasts uniformly between 10 and 60 seconds, uses a 2–5 second linear ramp capped at one quarter of phase duration, and never immediately repeats the prior profile.
- Profiles are equally weighted and use the exact duty-cycle ranges from `docs/superpowers/specs/2026-08-10-random-workload-design.md`.
- Random allocation reserves 64 MiB for Vector work and 160 MiB total for two HBM buffers; all remaining requested HBM stays resident in Cube matrices or empty reserve tensors.
- Phase transitions never allocate tensors or streams.
- Cube, Vector, and HBM feeders use distinct streams, bounded queue chunks, interruptible waits, and one shared shutdown deadline.
- `--busy-threshold -1` skips telemetry; non-negative thresholds pause all feeders and the virtual phase clock together.
- `--interval` remains the telemetry/backoff interval and never controls phase length.
- Production randomness is system-seeded; only internal tests inject a deterministic `random.Random`.
- Random mode writes no phase or telemetry data to disk and logs phase transitions only at DEBUG.
- Remote validation records observed trends rather than requiring exact counter percentages.

---

### Task 1: Add the random public value and fixed HBM planner

**Files:**
- Modify: `src/keep_npu/utilities/session_config.py`
- Modify: `src/keep_npu/single_npu_controller/workload.py`
- Modify: `tests/utilities/test_session_config.py`
- Modify: `tests/test_workload.py`

**Interfaces:**
- Consumes: `parse_vram_to_elements(vram) -> int`, expressed as float32 elements.
- Produces: `validate_workload(value: Any) -> str` accepting exactly `mixed|aicore|vector|random`.
- Produces: `RANDOM_VECTOR_BYTES`, `RANDOM_HBM_BYTES`, and `MIN_RANDOM_BYTES`.
- Produces: `RandomPlan(matrix_dim, vector_elements, hbm_buffer_elements, reserve_elements, allocated_bytes)`.
- Produces: `plan_random_workload(float32_elements: int) -> RandomPlan`.

- [ ] **Step 1: Write failing public-value tests**

Update `tests/utilities/test_session_config.py`:

```python
def test_validate_workload_accepts_public_values():
    assert validate_workload("mixed") == "mixed"
    assert validate_workload("aicore") == "aicore"
    assert validate_workload("vector") == "vector"
    assert validate_workload("random") == "random"


@pytest.mark.parametrize("value", [None, "", "RANDOM", "relu", 1, True])
def test_validate_workload_rejects_noncanonical_values(value):
    with pytest.raises(
        ValueError,
        match="workload must be 'mixed', 'aicore', 'vector', or 'random'",
    ):
        validate_workload(value)
```

- [ ] **Step 2: Run the public-value tests and verify RED**

Run:

```bash
pytest -q \
  tests/utilities/test_session_config.py::test_validate_workload_accepts_public_values \
  tests/utilities/test_session_config.py::test_validate_workload_rejects_noncanonical_values
```

Expected: `random` is rejected and the error message lists only the existing three values.

- [ ] **Step 3: Implement the public value**

Change `src/keep_npu/utilities/session_config.py`:

```python
PUBLIC_WORKLOADS = frozenset({"mixed", "aicore", "vector", "random"})


def validate_workload(value: Any) -> str:
    """Validate and normalize the keepalive workload name."""
    if not isinstance(value, str) or value not in PUBLIC_WORKLOADS:
        raise ValueError(
            "workload must be 'mixed', 'aicore', 'vector', or 'random'"
        )
    return value
```

- [ ] **Step 4: Run the public-value tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing random-plan tests**

Add to `tests/test_workload.py`:

```python
from keep_npu.single_npu_controller.workload import (
    MATRIX_COUNT,
    FP16_BYTES,
    MIN_AICORE_BYTES,
    MIN_RANDOM_BYTES,
    RANDOM_HBM_BYTES,
    RANDOM_VECTOR_BYTES,
    plan_random_workload,
)


def test_minimum_random_plan_reserves_all_three_engines():
    plan = plan_random_workload(MIN_RANDOM_BYTES // 4)

    assert plan.matrix_dim == 16
    assert plan.vector_elements == RANDOM_VECTOR_BYTES // 4
    assert plan.hbm_buffer_elements == (RANDOM_HBM_BYTES // 2) // 4
    assert plan.reserve_elements == 0
    assert plan.allocated_bytes == MIN_RANDOM_BYTES


def test_random_plan_rejects_budget_below_minimum():
    with pytest.raises(
        ValueError,
        match=f"random workload requires --vram of at least {MIN_RANDOM_BYTES} bytes",
    ):
        plan_random_workload((MIN_RANDOM_BYTES // 4) - 1)


def test_one_gib_random_plan_accounts_for_every_byte():
    plan = plan_random_workload(1024**3 // 4)
    matrix_bytes = MATRIX_COUNT * plan.matrix_dim**2 * FP16_BYTES
    active_bytes = plan.vector_elements * 4 + plan.hbm_buffer_elements * 4 * 2

    assert plan.matrix_dim % 16 == 0
    assert plan.matrix_dim <= 12288
    assert active_bytes == RANDOM_VECTOR_BYTES + RANDOM_HBM_BYTES
    assert matrix_bytes + active_bytes + plan.reserve_elements * 4 == 1024**3
    assert plan.allocated_bytes == 1024**3


def test_workload_vram_validation_applies_random_minimum():
    assert validate_workload_vram("random", MIN_RANDOM_BYTES) == MIN_RANDOM_BYTES // 4
    with pytest.raises(ValueError, match="random workload requires --vram"):
        validate_workload_vram("random", MIN_RANDOM_BYTES - 4)
```

- [ ] **Step 6: Run planner tests and verify RED**

Run:

```bash
pytest -q tests/test_workload.py
```

Expected: collection fails because the random constants, dataclass, and planner do not exist.

- [ ] **Step 7: Implement the random planner**

Add to `src/keep_npu/single_npu_controller/workload.py`:

```python
RANDOM_VECTOR_BYTES = 64 * 1024**2
RANDOM_HBM_BYTES = 160 * 1024**2
MIN_RANDOM_BYTES = MIN_AICORE_BYTES + RANDOM_VECTOR_BYTES + RANDOM_HBM_BYTES


@dataclass(frozen=True)
class RandomPlan:
    matrix_dim: int
    vector_elements: int
    hbm_buffer_elements: int
    reserve_elements: int
    allocated_bytes: int


def plan_random_workload(float32_elements: int) -> RandomPlan:
    budget_bytes = float32_elements * 4
    if budget_bytes < MIN_RANDOM_BYTES:
        raise ValueError(
            f"random workload requires --vram of at least {MIN_RANDOM_BYTES} bytes"
        )
    active_bytes = RANDOM_VECTOR_BYTES + RANDOM_HBM_BYTES
    cube_budget = budget_bytes - active_bytes
    raw_dim = isqrt(cube_budget // (MATRIX_COUNT * FP16_BYTES))
    matrix_dim = min(MAX_MIXED_MATRIX_DIM, raw_dim)
    matrix_dim -= matrix_dim % MATRIX_ALIGNMENT
    matrix_bytes = MATRIX_COUNT * matrix_dim**2 * FP16_BYTES
    vector_elements = RANDOM_VECTOR_BYTES // 4
    hbm_buffer_elements = (RANDOM_HBM_BYTES // 2) // 4
    reserve_elements = (budget_bytes - active_bytes - matrix_bytes) // 4
    allocated_bytes = matrix_bytes + active_bytes + reserve_elements * 4
    return RandomPlan(
        matrix_dim=matrix_dim,
        vector_elements=vector_elements,
        hbm_buffer_elements=hbm_buffer_elements,
        reserve_elements=reserve_elements,
        allocated_bytes=allocated_bytes,
    )
```

Extend `validate_workload_vram` with:

```python
if normalized_workload == "mixed":
    plan_mixed_workload(float32_elements)
elif normalized_workload == "random":
    plan_random_workload(float32_elements)
elif normalized_workload == "aicore":
    plan_aicore_workload(float32_elements)
```

- [ ] **Step 8: Run shared planner and validation tests**

Run:

```bash
pytest -q tests/test_workload.py tests/utilities/test_session_config.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit the public contract and planner**

```bash
git add \
  src/keep_npu/utilities/session_config.py \
  src/keep_npu/single_npu_controller/workload.py \
  tests/utilities/test_session_config.py \
  tests/test_workload.py
git commit -m "feat: plan random NPU workloads"
```

---

### Task 2: Build the deterministic phase scheduler

**Files:**
- Create: `src/keep_npu/single_npu_controller/random_workload.py`
- Create: `tests/test_random_workload.py`

**Interfaces:**
- Produces: `ProfileDefinition(name, cube_range, vector_range, hbm_range)`.
- Produces: `PhaseSnapshot(profile, cube_duty, vector_duty, hbm_duty, duration, ramp_duration, elapsed)`.
- Produces: `RandomPhaseScheduler(rng: Optional[random.Random] = None)`.
- Produces: `RandomPhaseScheduler.advance(active_seconds: float) -> PhaseSnapshot`; passing zero freezes the virtual phase clock.

- [ ] **Step 1: Write failing profile and transition tests**

Create `tests/test_random_workload.py`:

```python
import random

import pytest

from keep_npu.single_npu_controller.random_workload import (
    PROFILE_DEFINITIONS,
    RandomPhaseScheduler,
)


def test_profile_definitions_match_the_public_design():
    assert {
        item.name: (item.cube_range, item.vector_range, item.hbm_range)
        for item in PROFILE_DEFINITIONS
    } == {
        "cube": ((0.85, 1.00), (0.20, 0.45), (0.35, 0.65)),
        "vector": ((0.25, 0.55), (0.80, 1.00), (0.40, 0.70)),
        "hbm": ((0.20, 0.50), (0.35, 0.65), (0.85, 1.00)),
        "balanced": ((0.55, 0.85), (0.55, 0.85), (0.55, 0.85)),
    }


def test_seeded_scheduler_stays_in_bounds_and_never_repeats_profile():
    scheduler = RandomPhaseScheduler(random.Random(460))
    snapshots = [scheduler.advance(61.0) for _ in range(20)]

    assert all(10.0 <= item.duration <= 60.0 for item in snapshots)
    assert all(2.0 <= item.ramp_duration <= min(5.0, item.duration / 4) for item in snapshots)
    assert all(left.profile != right.profile for left, right in zip(snapshots, snapshots[1:]))
    for item in snapshots:
        profile = next(p for p in PROFILE_DEFINITIONS if p.name == item.profile)
        assert profile.cube_range[0] <= item.target_cube_duty <= profile.cube_range[1]
        assert profile.vector_range[0] <= item.target_vector_duty <= profile.vector_range[1]
        assert profile.hbm_range[0] <= item.target_hbm_duty <= profile.hbm_range[1]


def test_scheduler_ramps_from_zero_and_freezes_when_advance_is_zero():
    scheduler = RandomPhaseScheduler(random.Random(7))

    initial = scheduler.advance(0.0)
    halfway = scheduler.advance(initial.ramp_duration / 2)
    frozen = scheduler.advance(0.0)

    assert initial.cube_duty == initial.vector_duty == initial.hbm_duty == 0.0
    assert halfway.cube_duty == pytest.approx(halfway.target_cube_duty / 2)
    assert frozen == halfway
```

- [ ] **Step 2: Run scheduler tests and verify RED**

Run:

```bash
pytest -q tests/test_random_workload.py
```

Expected: collection fails because `random_workload.py` does not exist.

- [ ] **Step 3: Implement immutable profiles and scheduler state**

Create `src/keep_npu/single_npu_controller/random_workload.py` with these public structures and constants:

```python
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

DutyRange = Tuple[float, float]


@dataclass(frozen=True)
class ProfileDefinition:
    name: str
    cube_range: DutyRange
    vector_range: DutyRange
    hbm_range: DutyRange


@dataclass(frozen=True)
class PhaseSnapshot:
    profile: str
    cube_duty: float
    vector_duty: float
    hbm_duty: float
    target_cube_duty: float
    target_vector_duty: float
    target_hbm_duty: float
    duration: float
    ramp_duration: float
    elapsed: float


PROFILE_DEFINITIONS = (
    ProfileDefinition("cube", (0.85, 1.00), (0.20, 0.45), (0.35, 0.65)),
    ProfileDefinition("vector", (0.25, 0.55), (0.80, 1.00), (0.40, 0.70)),
    ProfileDefinition("hbm", (0.20, 0.50), (0.35, 0.65), (0.85, 1.00)),
    ProfileDefinition("balanced", (0.55, 0.85), (0.55, 0.85), (0.55, 0.85)),
)
```

Implement `RandomPhaseScheduler` so `_select_phase()` uses `rng.choice` over profiles excluding the previous name, samples `duration = rng.uniform(10.0, 60.0)`, samples `ramp_duration = min(rng.uniform(2.0, 5.0), duration / 4)`, and samples each target with `rng.uniform(*duty_range)`. `advance(active_seconds)` rejects negative/non-finite values, increments only virtual elapsed time, rolls across as many expired phases as necessary, and linearly interpolates from the prior targets to current targets. The first prior targets are `(0.0, 0.0, 0.0)`.

- [ ] **Step 4: Add invalid-elapsed and large-step tests**

Append to `tests/test_random_workload.py`:

```python
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_scheduler_rejects_invalid_active_elapsed(value):
    scheduler = RandomPhaseScheduler(random.Random(1))

    with pytest.raises(ValueError, match="active_seconds must be finite and non-negative"):
        scheduler.advance(value)


def test_large_advance_rolls_over_multiple_phases_without_losing_time():
    scheduler = RandomPhaseScheduler(random.Random(19))

    snapshot = scheduler.advance(180.0)

    assert 0.0 <= snapshot.elapsed < snapshot.duration
```

- [ ] **Step 5: Run scheduler tests and verify GREEN**

Run:

```bash
pytest -q tests/test_random_workload.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the scheduler**

```bash
git add \
  src/keep_npu/single_npu_controller/random_workload.py \
  tests/test_random_workload.py
git commit -m "feat: schedule random NPU phases"
```

---

### Task 3: Allocate and execute three bounded Ascend feeders

**Files:**
- Modify: `src/keep_npu/single_npu_controller/ascend_npu_controller.py`
- Modify: `tests/test_ascend_backend.py`

**Interfaces:**
- Consumes: `RandomPlan`, `plan_random_workload`, `RandomPhaseScheduler`, and `PhaseSnapshot`.
- Produces: `RandomAllocation(left, right, output, vector, hbm_source, hbm_target, reserves, plan, cube_stream, vector_stream, hbm_stream)`.
- Produces: `AscendNPUController._allocate_random(num_elements: int) -> RandomAllocation`.
- Produces: `AscendNPUController._run_random_session(allocation: RandomAllocation) -> None`.
- Adds internal injection point: `controller._random_scheduler_factory`, defaulting to `RandomPhaseScheduler`.

- [ ] **Step 1: Extend the fake backend and write failing allocation tests**

In `tests/test_ascend_backend.py`, replace vector dictionaries with a `FakeTensor(dict)` that implements:

```python
class FakeTensor(dict):
    def __init__(self, fake, **values):
        super().__init__(values)
        self.fake = fake

    def copy_(self, source, non_blocking=False):
        with self.fake.operation_lock:
            self.fake.copy_calls += 1
            self.fake.copy_streams.append(
                getattr(self.fake.stream_state, "active", None)
            )
        if self.fake.on_copy is not None:
            self.fake.on_copy(self.fake.copy_calls)
        return self
```

Make `FakeTorch.rand` and `FakeTorch.empty` return `FakeTensor`, add `sin`, `copy_calls`, `sin_calls`, `copy_streams`, `sin_streams`, `on_copy`, and `on_sin`, then add:

```python
def test_random_allocation_uses_one_budget_and_three_streams(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake = FakeTorch(count=1)
    monkeypatch.setattr(module, "load_torch_npu", lambda: fake)
    monkeypatch.setattr(module, "visible_torch_device_count", lambda: 1)
    controller = module.AscendNPUController(
        rank=0, vram_to_keep="1GiB", workload="random"
    )

    allocation = controller._allocate_random(controller.vram_to_keep)

    assert allocation.vector["elements"] * 4 == 64 * 1024**2
    assert allocation.hbm_source["elements"] * 4 == 80 * 1024**2
    assert allocation.hbm_target["elements"] * 4 == 80 * 1024**2
    assert len({allocation.cube_stream, allocation.vector_stream, allocation.hbm_stream}) == 3
    assert allocation.plan.allocated_bytes == 1024**3
    assert sum(item["elements"] for item in allocation.reserves) == allocation.plan.reserve_elements
```

- [ ] **Step 2: Run the allocation test and verify RED**

Run:

```bash
pytest -q tests/test_ascend_backend.py::test_random_allocation_uses_one_budget_and_three_streams
```

Expected: `random` initialization or `_allocate_random` fails because the controller does not support it.

- [ ] **Step 3: Implement fixed random allocation**

Add `RandomAllocation`, validate `plan_random_workload` in `__init__`, route `random` from `_allocate_workload`, and implement `_allocate_random`. Allocate the two HBM buffers with `torch.rand` and `torch.empty`, allocate the Vector tensor with `torch.rand`, allocate reserve with `_allocate_reserve`, and create exactly three streams. Use keyword arguments when constructing the dataclass so field order cannot silently route tensors to the wrong engine.

- [ ] **Step 4: Run allocation tests and verify GREEN**

Run:

```bash
pytest -q \
  tests/test_ascend_backend.py::test_random_allocation_uses_one_budget_and_three_streams \
  tests/test_ascend_backend.py::test_mixed_allocation_uses_one_budget_and_two_streams
```

Expected: both random and unchanged mixed allocation tests pass.

- [ ] **Step 5: Write failing routing, phase, backoff, and shutdown tests**

Add tests that inject a scheduler returning fixed `PhaseSnapshot` values and a stop event after each engine has executed:

```python
def test_random_session_routes_three_engines_to_distinct_streams(monkeypatch):
    from keep_npu.single_npu_controller import ascend_npu_controller as module

    fake, controller = make_random_controller(monkeypatch)
    allocation = controller._allocate_random(controller.vram_to_keep)
    controller._stop_evt = threading.Event()
    started = {name: threading.Event() for name in ("cube", "vector", "hbm")}
    fake.on_matmul = lambda _calls: started["cube"].set()
    fake.on_sin = lambda _calls: started["vector"].set()

    def on_copy(_calls):
        started["hbm"].set()
        if all(item.is_set() for item in started.values()):
            controller._stop_evt.set()

    fake.on_copy = on_copy

    controller._run_random_session(allocation)

    assert set(fake.matmul_streams) == {allocation.cube_stream}
    assert set(fake.sin_streams) == {allocation.vector_stream}
    assert set(fake.copy_streams) == {allocation.hbm_stream}


def test_random_busy_backoff_freezes_scheduler(monkeypatch):
    fake, controller = make_random_controller(monkeypatch, busy_threshold=25)
    scheduler = RecordingScheduler()
    controller._random_scheduler_factory = lambda: scheduler
    monkeypatch.setattr(controller, "_monitor_utilization", lambda _rank: 100)
    controller._stop_evt = StopAfterWaitEvent()

    controller._run_random_session(controller._allocate_random(controller.vram_to_keep))

    assert scheduler.active_seconds == [0.0]
    assert fake.matmul_calls == fake.sin_calls == fake.copy_calls == 0


@pytest.mark.parametrize(
    ("failed_engine", "message"),
    [
        ("cube", "cube feeder failed"),
        ("vector", "vector feeder failed"),
        ("hbm", "hbm feeder failed"),
    ],
)
def test_random_feeder_failure_names_engine_and_stops_siblings(
    monkeypatch, failed_engine, message
):
    fake, controller = make_random_controller(monkeypatch)
    inject_engine_failure(fake, failed_engine)

    with pytest.raises(RuntimeError, match=message):
        controller._run_random_session(controller._allocate_random(controller.vram_to_keep))

    assert not [thread for thread in threading.enumerate() if thread.name.startswith("npu-random-")]
```

The helper `make_random_controller` must set a 100% fixed snapshot, `RANDOM_QUANTUM_SECONDS = 0.01`, and `busy_threshold=-1` unless overridden. `RecordingScheduler.advance` appends the supplied active seconds. `StopAfterWaitEvent.wait` sets itself and returns `True` on the first interval wait. `inject_engine_failure` assigns the matching fake callback to raise `RuntimeError("stream failed")`.

- [ ] **Step 6: Run random execution tests and verify RED**

Run:

```bash
pytest -q tests/test_ascend_backend.py -k 'random_session or random_feeder'
```

Expected: failures show `_run_random_session` and its feeder supervision do not exist.

- [ ] **Step 7: Implement the coordinator and bounded feeders**

In `ascend_npu_controller.py` define:

```python
RANDOM_QUANTUM_SECONDS = 0.1
RANDOM_CUBE_CHUNK = 1
RANDOM_VECTOR_CHUNK = 8
RANDOM_HBM_CHUNK = 8
RANDOM_FEEDER_JOIN_TIMEOUT = 5.0
```

Implement `_run_random_session` with one cancellation event, one failure list guarded by a lock, and a shared immutable snapshot guarded by a lock. Each feeder repeatedly:

1. reads its duty from the current snapshot;
2. computes `active_deadline = quantum_start + duty * RANDOM_QUANTUM_SECONDS`;
3. submits only its bounded chunk while before that deadline;
4. synchronizes its own stream;
5. waits interruptibly until the quantum boundary.

Use `torch.matmul(..., out=...)` for Cube, `torch.sin(vector, out=vector)` for Vector, and alternating `hbm_target.copy_(hbm_source, non_blocking=True)` / `hbm_source.copy_(hbm_target, non_blocking=True)` for HBM. The coordinator calls `scheduler.advance(real_elapsed)` only while enabled and `scheduler.advance(0.0)` while backoff is active. For non-negative thresholds, call `_monitor_utilization` no more often than every `interval` seconds. DEBUG-log only profile changes. Join all feeders against one `time.monotonic() + RANDOM_FEEDER_JOIN_TIMEOUT` deadline and surface the first named failure.

Route random allocations in `_run_batch` before the single-engine branches:

```python
if self.workload == "random":
    self._run_random_session(allocation)
    return
```

- [ ] **Step 8: Run backend tests and verify GREEN**

Run:

```bash
pytest -q tests/test_ascend_backend.py
```

Expected: all Ascend backend tests pass, including existing mixed/aicore/vector behavior.

- [ ] **Step 9: Commit the controller implementation**

```bash
git add \
  src/keep_npu/single_npu_controller/ascend_npu_controller.py \
  tests/test_ascend_backend.py
git commit -m "feat: run random Cube Vector and HBM phases"
```

---

### Task 4: Propagate random mode through CLI, service, and MCP

**Files:**
- Modify: `src/keep_npu/cli.py`
- Modify: `src/keep_npu/mcp/server.py`
- Modify: `tests/test_cli_thresholds.py`
- Modify: `tests/test_cli_service_commands.py`
- Modify: `tests/mcp/test_server.py`
- Modify: `tests/mcp/test_http_api.py`

**Interfaces:**
- Consumes: `validate_workload`, `validate_workload_vram`, and `MIN_RANDOM_BYTES`.
- Produces: public CLI/help and JSON/MCP schemas accepting `workload="random"`.
- Keeps: omitted workload values normalize to `mixed`.

- [ ] **Step 1: Write failing blocking and service CLI tests**

Add or update tests:

```python
def test_validate_cli_workload_accepts_public_values():
    for workload in ("mixed", "aicore", "vector", "random"):
        assert cli._validate_cli_workload(workload) == workload


def test_blocking_command_accepts_explicit_random_workload(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "_run_blocking", lambda *args: captured.setdefault("workload", args[-1]))

    result = runner.invoke(cli.app, ["--workload", "random", "--vram", "1GiB"])

    assert result.exit_code == 0
    assert captured["workload"] == "random"


def test_service_start_forwards_random_workload(monkeypatch):
    captured = install_start_rpc_capture(monkeypatch)

    result = runner.invoke(cli.app, ["start", "--workload", "random", "--vram", "1GiB"])

    assert result.exit_code == 0
    assert captured["params"]["workload"] == "random"
```

- [ ] **Step 2: Write failing MCP schema and API tests**

Update `tests/mcp/test_server.py` and `tests/mcp/test_http_api.py`:

```python
def test_start_keep_accepts_explicit_random_workload():
    server = make_server()

    job_id = server.start_keep(npu_ids=[0], workload="random", vram="1GiB")["job_id"]

    assert server.status(job_id)["params"]["workload"] == "random"
    assert server._sessions[job_id].controller.workload == "random"


def test_start_keep_schema_exposes_random_minimum():
    schema = get_start_keep_schema()

    assert schema["properties"]["workload"]["enum"] == [
        "mixed", "aicore", "vector", "random"
    ]
    assert {
        "if": {"properties": {"workload": {"const": "random"}}},
        "then": {"properties": {"vram": {"minimum": MIN_RANDOM_BYTES}}},
    } in schema["allOf"]
```

- [ ] **Step 3: Run public-surface tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_cli_thresholds.py \
  tests/test_cli_service_commands.py \
  tests/mcp/test_server.py \
  tests/mcp/test_http_api.py -k 'workload or random'
```

Expected: random validation/schema/help assertions fail.

- [ ] **Step 4: Implement CLI and MCP propagation**

Update both CLI workload help strings to:

```python
help=(
    "Keepalive workload: mixed (default), aicore, vector, or random "
    "phase-based Cube/Vector/HBM pressure."
)
```

In `src/keep_npu/mcp/server.py`, import `MIN_RANDOM_BYTES`, extend the enum to `['mixed', 'aicore', 'vector', 'random']`, add the random conditional minimum to `allOf`, and update descriptions/docstrings without changing request or response field names. Shared validation continues to enforce runtime requests.

- [ ] **Step 5: Run CLI, service, and MCP tests and verify GREEN**

Run the command from Step 3 without the `-k` filter:

```bash
pytest -q \
  tests/test_cli_thresholds.py \
  tests/test_cli_service_commands.py \
  tests/mcp/test_server.py \
  tests/mcp/test_http_api.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit public Python interfaces**

```bash
git add \
  src/keep_npu/cli.py \
  src/keep_npu/mcp/server.py \
  tests/test_cli_thresholds.py \
  tests/test_cli_service_commands.py \
  tests/mcp/test_server.py \
  tests/mcp/test_http_api.py
git commit -m "feat: expose random workload interfaces"
```

---

### Task 5: Add Dashboard selection and rebuild packaged assets

**Files:**
- Modify: `web/dashboard/src/App.jsx`
- Modify: `web/dashboard/src/App.test.jsx`
- Modify: `web/dashboard/src/lib/session.js`
- Modify: `web/dashboard/src/lib/session.test.js`
- Modify: `web/dashboard/src/lib/refresh.js`
- Modify: `web/dashboard/src/lib/refresh.test.js`
- Modify generated: `src/keep_npu/mcp/static/assets/dashboard.js`
- Verify generated: `src/keep_npu/mcp/static/index.html`
- Verify generated: `src/keep_npu/mcp/static/assets/index.css`

**Interfaces:**
- Consumes/produces: session payload `workload` with the fourth value `random`.
- Produces: Dashboard option text `Random workload phases`.
- Keeps: default form workload `mixed`.

- [ ] **Step 1: Write failing Dashboard parser and render tests**

Update `web/dashboard/src/lib/session.test.js`:

```javascript
it("accepts the four public workload modes", () => {
  expect(parseWorkload("mixed")).toBe("mixed")
  expect(parseWorkload("aicore")).toBe("aicore")
  expect(parseWorkload("vector")).toBe("vector")
  expect(parseWorkload("random")).toBe("random")
})
```

Update the invalid error assertion to `Workload must be mixed, aicore, vector, or random`, and add to `App.test.jsx`:

```javascript
expect(markup).toContain('<option value="random">Random workload phases</option>')
```

- [ ] **Step 2: Run Dashboard tests and verify RED**

Run:

```bash
npm --prefix web/dashboard test
```

Expected: parser and option tests fail because `random` is absent.

- [ ] **Step 3: Implement Dashboard support**

Change `parseWorkload` in `web/dashboard/src/lib/session.js` to accept all four exact strings. Add this option after Vector in `App.jsx`:

```jsx
<option value="random">Random workload phases</option>
```

Extend `isKnownWorkload`/refresh validation in `web/dashboard/src/lib/refresh.js` to accept `random`, preserving `mixed` as the omitted/default value.

- [ ] **Step 4: Run Dashboard tests and verify GREEN**

Run:

```bash
npm --prefix web/dashboard test
```

Expected: all Vitest tests pass.

- [ ] **Step 5: Build and verify packaged static assets**

Run:

```bash
npm --prefix web/dashboard run build
npm --prefix web/dashboard test -- staticAssets.test.js
rg -n 'Random workload phases' src/keep_npu/mcp/static/assets/dashboard.js
```

Expected: Vite rebuilds the committed MCP assets, the package parity test passes, and the generated bundle contains the new label.

- [ ] **Step 6: Commit Dashboard and packaged assets**

```bash
git add \
  web/dashboard/src/App.jsx \
  web/dashboard/src/App.test.jsx \
  web/dashboard/src/lib/session.js \
  web/dashboard/src/lib/session.test.js \
  web/dashboard/src/lib/refresh.js \
  web/dashboard/src/lib/refresh.test.js \
  src/keep_npu/mcp/static/index.html \
  src/keep_npu/mcp/static/assets/dashboard.js \
  src/keep_npu/mcp/static/assets/index.css
git commit -m "feat: add random workload to dashboard"
```

---

### Task 6: Document, validate, and calibrate on Ascend hardware

**Files:**
- Modify: `README.md`
- Modify: `docs/compatibility.md`
- Modify: `docs/validation/ascend-remote-results.md`
- Potential calibration target: `src/keep_npu/single_npu_controller/ascend_npu_controller.py`
- Calibration regression coverage: `tests/test_ascend_backend.py`

**Interfaces:**
- Documents: `keep-npu --workload random --npu-ids ... --vram ...`.
- Records: phase-by-phase Cube, Vector, HBM, total utilization, resident HBM, selected-device processes, and shutdown results.

- [ ] **Step 1: Update user-facing documentation**

Add this example and explanation to `README.md`:

```console
keep-npu --workload random --npu-ids 0,1 --vram 1GiB \
  --interval 60 --busy-threshold -1
```

State that random mode keeps requested HBM resident, changes among Cube-, Vector-, HBM-intensive and balanced phases every 10–60 seconds, and shapes trends rather than exact utilization percentages. Update `docs/compatibility.md` to list all four workload values and the roughly 224 MiB random minimum.

- [ ] **Step 2: Run full local verification**

Run:

```bash
pytest -q
ruff check .
ruff format --check \
  src/keep_npu/utilities/session_config.py \
  src/keep_npu/single_npu_controller/workload.py \
  src/keep_npu/single_npu_controller/random_workload.py \
  src/keep_npu/single_npu_controller/ascend_npu_controller.py \
  tests/test_random_workload.py \
  tests/test_workload.py \
  tests/test_ascend_backend.py
npm --prefix web/dashboard test
npm --prefix web/dashboard run build
git diff --check
```

Expected: all commands exit zero. Do not treat the repository's known unrelated strict-Mypy baseline as a feature regression; run Mypy only if its baseline has first been made clean in a separate change.

- [ ] **Step 3: Build a release-candidate wheel**

Run:

```bash
python -m build
python -m twine check dist/*
```

Expected: wheel and sdist build successfully, and Twine reports both as valid.

- [ ] **Step 4: Audit remote devices before mutation**

Use the `pie:remote-access` skill. On each configured NPU host, run `npu-smi info`, record health and existing processes, and select only devices with no process and baseline HBM. Do not terminate or reuse another user's process. Install the candidate wheel into a uniquely named `--system-site-packages` virtual environment inside a container that already has compatible `torch` and `torch_npu`.

- [ ] **Step 5: Validate multiple random phases**

Run on one idle healthy device:

```console
keep-npu --workload random --npu-ids 0 --vram 1GiB \
  --interval 60 --busy-threshold -1
```

Temporarily enable DEBUG logging only for validation so profile names can be correlated with `npu-smi info -t usages -i <id> -c 0`. Capture at least one complete Cube-intensive, Vector-intensive, HBM-intensive, and balanced phase. Verify resident HBM does not drop between phases and the named intensive resource is visibly dominant relative to at least one other captured phase.

- [ ] **Step 6: Calibrate only bounded constants if hardware disproves the initial operator choices**

If Vector or HBM phases do not show the intended trend, vary only `RANDOM_VECTOR_CHUNK`, `RANDOM_HBM_CHUNK`, scheduling quantum, or the Vector/HBM operator while preserving the public profile ranges and memory plan. For every calibration change, first add/update a focused fake-backend regression test, rerun `tests/test_ascend_backend.py`, rebuild the wheel, and repeat the same-device comparison. Do not change `mixed` constants.

- [ ] **Step 7: Validate selection and shutdown on multiple devices**

Run random mode on two or more idle devices. Confirm `npu-smi info` associates KeepNPU only with the selected ordinals. Send SIGTERM, verify no feeder timeout/error, confirm memory returns to each device's pre-test range, and confirm no KeepNPU process remains.

- [ ] **Step 8: Record empirical results and clean remote artifacts**

Append the exact hardware, health, command, profile readings, process mapping, release timing, and limitations to `docs/validation/ascend-remote-results.md`. Remove only the uniquely named candidate wheel, virtual environment, scripts, logs, and PID files created by this validation, then re-run `npu-smi info` to prove cleanup.

- [ ] **Step 9: Re-run final verification after calibration/docs**

Repeat Step 2 and rebuild the final artifacts from the exact final commit candidate.

- [ ] **Step 10: Commit documentation and validated tuning**

```bash
git add \
  README.md \
  docs/compatibility.md \
  docs/validation/ascend-remote-results.md \
  src/keep_npu/single_npu_controller/ascend_npu_controller.py \
  tests/test_ascend_backend.py
git commit -m "docs: validate random workload phases"
```

---

### Task 7: Final review and release preparation

**Files:**
- Review: all files changed since the plan's base commit.
- Modify only if releasing now: `src/keep_npu/__init__.py`, `pyproject.toml`.

**Interfaces:**
- Produces: a clean, tested branch ready for integration or an explicitly requested versioned release.

- [ ] **Step 1: Inspect scope and history**

Run:

```bash
git status -sb
git log --oneline --decorate -12
git diff --stat main...HEAD
git diff --check main...HEAD
```

Expected: only random-workload implementation, tests, docs, and generated assets appear.

- [ ] **Step 2: Use verification-before-completion**

Invoke `superpowers:verification-before-completion`, rerun the full commands from Task 6 Step 2 and artifact checks from Task 6 Step 3, read every exit status, and compare the implementation line-by-line with `docs/superpowers/specs/2026-08-10-random-workload-design.md`.

- [ ] **Step 3: Prepare a release only under existing user authorization**

If direct release authorization still applies, bump the patch version consistently in `src/keep_npu/__init__.py` and `pyproject.toml`, rebuild and verify artifacts, commit the version bump, fast-forward the verified branch to `main`, push `main` and the version tag, publish the GitHub Release, monitor the PyPI workflow to success, and install the published version from both PyPI and the Huawei Cloud mirror. Otherwise stop with the verified feature branch ready for review.

- [ ] **Step 4: Confirm repository and remote cleanliness**

Run:

```bash
git status -sb
git worktree list
```

Recheck all tested NPU hosts for zero KeepNPU test processes and absence of uniquely named validation artifacts. Report the final commit/tag, local test counts, remote metrics, published install command if applicable, and any hardware-specific counter limitations.
