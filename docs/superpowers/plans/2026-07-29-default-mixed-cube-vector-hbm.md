# Default Mixed Cube, Vector, and HBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KeepNPU's default workload concurrently sustain Cube/AI Core, AI Vector, and HBM bandwidth pressure while preserving explicit Cube-only and Vector-only compatibility modes.

**Architecture:** Add a pure mixed-memory planner that divides one per-NPU HBM budget between aligned FP16 matrices and float32 Vector buffers. The existing per-device controller will run two bounded feeder threads on independent `torch.npu.Stream` instances, propagate the first feeder failure through existing runtime supervision, and retain the current single-engine paths for explicit modes. One normalized `mixed` default is then propagated through the CLI, service protocols, dashboard, documentation, and packaged static assets.

**Tech Stack:** Python 3.9–3.13, PyTorch with `torch_npu`, `threading`, Typer, pytest, React 18, Vitest, Vite, `npu-smi` on Ascend 910B2.

## Global Constraints

- Public workload values are exactly `mixed`, `aicore`, and `vector`.
- `DEFAULT_WORKLOAD` is exactly `mixed`; users do not need `--workload` for the mixed mode.
- `--vram` remains one total per-NPU byte budget shared by Cube matrices and Vector buffers.
- The default `1GiB` plan uses 12288-by-12288 FP16 matrices and assigns the remaining roughly 160 MiB to Vector buffers; explicit `aicore` retains its 8192 cap.
- Mixed mode reserves at least 64 MiB for Vector/HBM work and at least the existing 1536-byte aligned Cube allocation.
- `--busy-threshold -1` skips telemetry gating and inter-batch sleep; non-negative thresholds gate the whole mixed batch.
- Feeder queues are bounded and synchronize their own streams; per-operation global synchronization is forbidden.
- The first feeder failure is preserved with its engine name, stops the sibling, and reaches the existing blocking/service runtime supervision.
- Explicit `aicore` and `vector` requests retain their current allocation and execution behavior.
- Hardware acceptance requires ten post-warm-up samples per device with simultaneous AICore/Cube, AI Vector, and HBM bandwidth readings of at least 90%.

---

### Task 1: Define the mixed public value and shared-memory plan

**Files:**
- Modify: `src/keep_npu/utilities/session_config.py:9,121-125`
- Modify: `src/keep_npu/single_npu_controller/workload.py`
- Modify: `tests/utilities/test_session_config.py:178-187`
- Modify: `tests/test_workload.py`

**Interfaces:**
- Consumes: `parse_vram_to_elements(vram) -> int`, where the return value is a float32 element count.
- Produces: `DEFAULT_WORKLOAD = "mixed"`.
- Produces: `validate_workload(value: Any) -> str` accepting exactly `mixed|aicore|vector`.
- Produces: `MixedPlan(matrix_dim: int, vector_elements: int, allocated_bytes: int)`.
- Produces: `plan_mixed_workload(float32_elements: int) -> MixedPlan`.
- Produces: `MIN_MIXED_VECTOR_BYTES = 64 * 1024**2` and `MIN_MIXED_BYTES = MIN_AICORE_BYTES + MIN_MIXED_VECTOR_BYTES`.

- [ ] **Step 1: Write failing public-value tests**

Add to `tests/utilities/test_session_config.py`:

```python
def test_default_workload_is_mixed():
    from keep_npu.utilities.session_config import DEFAULT_WORKLOAD

    assert DEFAULT_WORKLOAD == "mixed"


def test_validate_workload_accepts_public_values():
    assert validate_workload("mixed") == "mixed"
    assert validate_workload("aicore") == "aicore"
    assert validate_workload("vector") == "vector"


@pytest.mark.parametrize("value", [None, "", "MIXED", "relu", 1, True])
def test_validate_workload_rejects_noncanonical_values(value):
    with pytest.raises(
        ValueError, match="workload must be 'mixed', 'aicore', or 'vector'"
    ):
        validate_workload(value)
```

Replace the existing two-value acceptance/error assertions rather than keeping
duplicate tests.

- [ ] **Step 2: Run the public-value tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/utilities/test_session_config.py::test_default_workload_is_mixed \
  tests/utilities/test_session_config.py::test_validate_workload_accepts_public_values \
  tests/utilities/test_session_config.py::test_validate_workload_rejects_noncanonical_values
```

Expected: failures show that the default is `aicore`, `mixed` is rejected, and
the old error message lists only two values.

- [ ] **Step 3: Implement the public value**

Change `src/keep_npu/utilities/session_config.py` to:

```python
DEFAULT_WORKLOAD = "mixed"
PUBLIC_WORKLOADS = frozenset({"mixed", "aicore", "vector"})


def validate_workload(value: Any) -> str:
    """Validate and normalize the keepalive workload name."""
    if not isinstance(value, str) or value not in PUBLIC_WORKLOADS:
        raise ValueError("workload must be 'mixed', 'aicore', or 'vector'")
    return value
```

- [ ] **Step 4: Run the public-value tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing mixed-plan tests**

Add these imports and tests to `tests/test_workload.py`:

```python
from keep_npu.single_npu_controller.workload import (
    MAX_MATRIX_DIM,
    MIN_MIXED_BYTES,
    MIN_MIXED_VECTOR_BYTES,
    plan_mixed_workload,
)


def test_minimum_mixed_plan_reserves_aligned_cube_and_vector_memory():
    plan = plan_mixed_workload(MIN_MIXED_BYTES // 4)

    assert plan.matrix_dim == 16
    assert plan.vector_elements == MIN_MIXED_VECTOR_BYTES // 4
    assert plan.allocated_bytes == MIN_MIXED_BYTES


def test_mixed_plan_rejects_budget_below_minimum():
    with pytest.raises(
        ValueError,
        match=f"mixed workload requires --vram of at least {MIN_MIXED_BYTES} bytes",
    ):
        plan_mixed_workload((MIN_MIXED_BYTES // 4) - 1)


def test_default_mixed_plan_caps_cube_and_assigns_remainder_to_vector():
    budget_elements = 1024**3 // 4
    plan = plan_mixed_workload(budget_elements)
    matrix_bytes = 3 * MAX_MATRIX_DIM**2 * 2

    assert plan.matrix_dim == MAX_MATRIX_DIM
    assert plan.vector_elements == (1024**3 - matrix_bytes) // 4
    assert plan.allocated_bytes == 1024**3


def test_workload_vram_validation_applies_mixed_minimum():
    assert validate_workload_vram("mixed", MIN_MIXED_BYTES) == MIN_MIXED_BYTES // 4
    with pytest.raises(ValueError, match="mixed workload requires --vram"):
        validate_workload_vram("mixed", MIN_MIXED_BYTES - 4)
```

- [ ] **Step 6: Run the mixed-plan tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_workload.py
```

Expected: collection fails because the mixed constants, dataclass, and planner
do not exist.

- [ ] **Step 7: Implement the pure mixed planner**

Add to `src/keep_npu/single_npu_controller/workload.py`:

```python
MIN_MIXED_VECTOR_BYTES = 64 * 1024**2
MIN_MIXED_BYTES = MIN_AICORE_BYTES + MIN_MIXED_VECTOR_BYTES


@dataclass(frozen=True)
class MixedPlan:
    """Disjoint Cube and Vector allocations inside one public HBM budget."""

    matrix_dim: int
    vector_elements: int
    allocated_bytes: int


def plan_mixed_workload(float32_elements: int) -> MixedPlan:
    budget_bytes = float32_elements * 4
    if budget_bytes < MIN_MIXED_BYTES:
        raise ValueError(
            f"mixed workload requires --vram of at least {MIN_MIXED_BYTES} bytes"
        )
    cube_budget = budget_bytes - MIN_MIXED_VECTOR_BYTES
    raw_dim = isqrt(cube_budget // (MATRIX_COUNT * FP16_BYTES))
    matrix_dim = min(MAX_MATRIX_DIM, raw_dim)
    matrix_dim -= matrix_dim % MATRIX_ALIGNMENT
    matrix_bytes = MATRIX_COUNT * matrix_dim**2 * FP16_BYTES
    vector_elements = (budget_bytes - matrix_bytes) // 4
    allocated_bytes = matrix_bytes + vector_elements * 4
    return MixedPlan(matrix_dim, vector_elements, allocated_bytes)
```

Update `validate_workload_vram`:

```python
if normalized_workload == "mixed":
    plan_mixed_workload(float32_elements)
elif normalized_workload == "aicore":
    plan_aicore_workload(float32_elements)
```

- [ ] **Step 8: Run planner and shared-validation tests**

Run:

```bash
.venv/bin/pytest -q tests/test_workload.py tests/utilities/test_session_config.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit the public contract and planner**

```bash
git add \
  src/keep_npu/utilities/session_config.py \
  src/keep_npu/single_npu_controller/workload.py \
  tests/utilities/test_session_config.py \
  tests/test_workload.py
git commit -m "feat: plan mixed Cube and Vector memory"
```

---

### Task 2: Execute Cube and Vector feeders concurrently on independent streams

**Files:**
- Modify: `src/keep_npu/single_npu_controller/ascend_npu_controller.py`
- Modify: `tests/test_ascend_backend.py`

**Interfaces:**
- Consumes: `MixedPlan` and `plan_mixed_workload(float32_elements)`.
- Produces: `MixedAllocation(left, right, output, vectors, plan, cube_stream, vector_stream)`.
- Produces: `AscendNPUController._allocate_mixed(num_elements) -> MixedAllocation`.
- Produces: `AscendNPUController._run_mixed_batch(allocation) -> None`.
- Keeps: `allocation_status() -> Optional[Exception]` as the runtime supervisor boundary.

- [ ] **Step 1: Extend the fake backend with observable stream semantics**

Add a `FakeStream` and stream context to `tests/test_ascend_backend.py`:

```python
class FakeStream:
    def __init__(self, fake, device=None):
        self.fake = fake
        self.device = device
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class FakeStreamContext:
    def __init__(self, fake, stream):
        self.fake = fake
        self.stream = stream
        self.previous = None

    def __enter__(self):
        self.previous = getattr(self.fake.stream_state, "active", None)
        self.fake.stream_state.active = self.stream

    def __exit__(self, exc_type, exc, tb):
        self.fake.stream_state.active = self.previous
```

Extend `FakeNPU` with `Stream(device=None)` and `stream(stream)` methods that
create these objects. Initialize `FakeTorch.stream_state = threading.local()`
and `FakeTorch.operation_lock = threading.Lock()`. Protect
`FakeTorch.matmul_calls`, `FakeTorch.relu_calls`, and the per-call stream
record lists with `operation_lock`; record
`getattr(fake.stream_state, "active", None)` for every operation. Thread-local
stream state is required so one feeder cannot overwrite the other feeder's
test observation.

- [ ] **Step 2: Write failing allocation and stream-routing tests**

Add:

```python
def test_controller_defaults_to_mixed_workload(monkeypatch):
    controller, _fake = make_controller(monkeypatch, vram_to_keep="1GiB")
    assert controller.workload == "mixed"


def test_mixed_allocation_uses_one_budget_and_two_streams(monkeypatch):
    controller, fake = make_controller(monkeypatch, vram_to_keep="1GiB")

    allocation = controller._allocate_mixed(controller.vram_to_keep)

    assert allocation.left["device"] == "npu:0"
    assert allocation.right["device"] == "npu:0"
    assert allocation.output["device"] == "npu:0"
    assert allocation.vectors
    assert allocation.cube_stream is not allocation.vector_stream
    assert allocation.plan.allocated_bytes == 1024**3
    assert sum(item["elements"] for item in allocation.vectors) == (
        allocation.plan.vector_elements
    )


def test_mixed_batch_routes_operations_to_distinct_streams(monkeypatch):
    controller, fake = make_controller(monkeypatch, vram_to_keep="1GiB")
    allocation = controller._allocate_mixed(controller.vram_to_keep)
    controller._stop_evt = threading.Event()
    fake.stop_after_matmul = 2
    fake.stop_event = controller._stop_evt

    controller._run_mixed_batch(allocation)

    assert fake.matmul_streams
    assert fake.relu_streams
    assert set(fake.matmul_streams) == {allocation.cube_stream}
    assert set(fake.relu_streams) == {allocation.vector_stream}
```

The existing test helper may be extracted from repeated monkeypatch setup, but
it must return a real `AscendNPUController` and a complete `FakeTorch`.

- [ ] **Step 3: Run the new controller tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_ascend_backend.py \
  -k "defaults_to_mixed or mixed_allocation or mixed_batch_routes"
```

Expected: failures show missing mixed allocation and execution methods, and
the current default remains `aicore`.

- [ ] **Step 4: Add mixed allocation and dispatch**

Add:

```python
MIXED_BATCH_SECONDS = 1.0
MIXED_UNCONDITIONAL_BATCH_SECONDS = 60.0
MIXED_FEEDER_JOIN_TIMEOUT = 5.0
MIXED_CUBE_CHUNK = 1
MIXED_VECTOR_CHUNK = 64


@dataclass
class MixedAllocation:
    left: Any
    right: Any
    output: Any
    vectors: List[Any]
    plan: MixedPlan
    cube_stream: Any
    vector_stream: Any
```

Implement `_allocate_mixed` with the same FP16 tensor construction as
`_allocate_aicore`, `_allocate_vector(plan.vector_elements)`, and:

```python
cube_stream = self._torch.npu.Stream(device=self.device)
vector_stream = self._torch.npu.Stream(device=self.device)
```

Update `_allocate_workload` and `_run_batch` to use three explicit branches:

```python
if self.workload == "mixed":
    return self._allocate_mixed(num_elements)
if self.workload == "aicore":
    return self._allocate_aicore(num_elements)
return self._allocate_vector(num_elements)
```

- [ ] **Step 5: Implement bounded feeder coordination**

Implement `_run_mixed_batch` around one deadline and one batch-cancel event:

```python
def _run_mixed_batch(self, allocation: MixedAllocation) -> None:
    deadline = time.monotonic() + MIXED_BATCH_SECONDS
    cancel = threading.Event()
    failures: list[tuple[str, Exception]] = []
    failure_lock = threading.Lock()

    def should_stop() -> bool:
        return (
            cancel.is_set()
            or (self._stop_evt is not None and self._stop_evt.is_set())
            or time.monotonic() >= deadline
        )

    def record_failure(engine: str, exc: Exception) -> None:
        with failure_lock:
            if not failures:
                failures.append((engine, exc))
        cancel.set()

    def cube_feeder() -> None:
        try:
            self._torch.npu.set_device(self.rank)
            with self._torch.npu.stream(allocation.cube_stream):
                while not should_stop():
                    for _ in range(MIXED_CUBE_CHUNK):
                        self._torch.matmul(
                            allocation.left,
                            allocation.right,
                            out=allocation.output,
                        )
                        if should_stop():
                            break
                    allocation.cube_stream.synchronize()
        except Exception as exc:
            record_failure("cube", exc)

    def vector_feeder() -> None:
        try:
            self._torch.npu.set_device(self.rank)
            with self._torch.npu.stream(allocation.vector_stream):
                while not should_stop():
                    for _ in range(MIXED_VECTOR_CHUNK):
                        for tensor in allocation.vectors:
                            self._torch.relu_(tensor)
                        if should_stop():
                            break
                    allocation.vector_stream.synchronize()
        except Exception as exc:
            record_failure("vector", exc)

    feeders = [
        threading.Thread(target=cube_feeder, name=f"npu-cube-{self.rank}", daemon=True),
        threading.Thread(
            target=vector_feeder, name=f"npu-vector-{self.rank}", daemon=True
        ),
    ]
    for feeder in feeders:
        feeder.start()
    for feeder in feeders:
        feeder.join(timeout=MIXED_FEEDER_JOIN_TIMEOUT)
    stuck = [feeder.name for feeder in feeders if feeder.is_alive()]
    if stuck:
        cancel.set()
        raise TimeoutError(f"mixed feeder threads did not stop: {', '.join(stuck)}")
    if failures:
        engine, exc = failures[0]
        raise RuntimeError(f"{engine} feeder failed: {exc}") from exc
```

Do not call `torch.npu.synchronize()` inside this method. Each feeder
synchronizes only its own stream after a bounded chunk.

- [ ] **Step 6: Run allocation and routing tests and verify GREEN**

Run the command from Step 3.

Expected: all selected tests pass.

- [ ] **Step 7: Write failing first-error and sibling-stop tests**

Add:

```python
@pytest.mark.parametrize(("failed_engine", "message"), [
    ("cube", "cube feeder failed: cube stream failed"),
    ("vector", "vector feeder failed: vector stream failed"),
])
def test_mixed_feeder_failure_stops_batch_and_preserves_engine(
    monkeypatch, failed_engine, message
):
    controller, fake = make_controller(monkeypatch, vram_to_keep="1GiB")
    allocation = controller._allocate_mixed(controller.vram_to_keep)
    controller._stop_evt = threading.Event()
    if failed_engine == "cube":
        fake.on_matmul = lambda _calls: (_ for _ in ()).throw(
            RuntimeError("cube stream failed")
        )
    else:
        fake.on_relu = lambda _calls: (_ for _ in ()).throw(
            RuntimeError("vector stream failed")
        )

    with pytest.raises(RuntimeError, match=message):
        controller._run_mixed_batch(allocation)

    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name in {"npu-cube-0", "npu-vector-0"}
    ]
```

Also add an integration-level fake-backend test that calls `controller.keep()`,
waits until `allocation_status()` is non-`None`, and asserts the stored message
contains the failed engine and original exception text.

- [ ] **Step 8: Run failure tests and verify RED, then make only necessary fixes**

Run:

```bash
.venv/bin/pytest -q tests/test_ascend_backend.py \
  -k "mixed_feeder_failure or mixed_runtime_failure"
```

Expected before fixes: at least one assertion exposes missing cancellation,
incorrect exception text, or a feeder left alive. Adjust only
`_run_mixed_batch` coordination until both tests pass.

- [ ] **Step 9: Run all controller, planner, and global-controller tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_workload.py \
  tests/test_ascend_backend.py \
  tests/test_global_npu_controller.py
```

Expected: all pass, including explicit `aicore` and `vector` regression tests.

- [ ] **Step 10: Commit mixed execution**

```bash
git add \
  src/keep_npu/single_npu_controller/ascend_npu_controller.py \
  tests/test_ascend_backend.py
git commit -m "feat: run Cube and Vector streams concurrently"
```

---

### Task 3: Propagate the mixed default through CLI and service protocols

**Files:**
- Modify: `src/keep_npu/cli.py`
- Modify: `src/keep_npu/legacy.py`
- Modify: `src/keep_npu/mcp/server.py`
- Modify: `tests/test_cli_thresholds.py`
- Modify: `tests/test_cli_service_commands.py`
- Modify: `tests/test_keep_npu_alive.py`
- Modify: `tests/mcp/test_server.py`
- Modify: `tests/mcp/test_http_api.py`

**Interfaces:**
- Consumes: `DEFAULT_WORKLOAD == "mixed"` and three-value
  `validate_workload`.
- Produces: CLI, JSON-RPC, REST, MCP, status, and legacy translations that
  normalize omitted workload to `mixed`.
- Preserves: blocking worker health polling from commit `edd6dbc`.

- [ ] **Step 1: Change focused expectations to the new default and enum**

Update or add assertions:

```python
assert captured["workload"] == "mixed"
assert called["args"] == (120, "0", "1GiB", None, 25, "mixed")
assert status["params"]["workload"] == "mixed"
assert workload_schema["enum"] == ["mixed", "aicore", "vector"]
assert workload_schema["default"] == "mixed"
```

Add explicit CLI acceptance:

```python
def test_blocking_command_accepts_explicit_mixed_workload(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "_run_blocking",
        lambda *args: captured.setdefault("workload", args[-1]),
    )

    result = runner.invoke(cli.app, ["--workload", "mixed"])

    assert result.exit_code == 0
    assert captured["workload"] == "mixed"
```

Update test fixture/session dictionaries only where they represent an omitted
default. Keep `"aicore"` in tests that explicitly exercise Cube-only behavior.

- [ ] **Step 2: Run focused protocol tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_cli_thresholds.py \
  tests/test_cli_service_commands.py \
  tests/test_keep_npu_alive.py \
  tests/mcp/test_server.py \
  tests/mcp/test_http_api.py
```

Expected: failures identify old two-value schema/help text, legacy `aicore`
translation, and stale default fixture expectations.

- [ ] **Step 3: Update CLI and MCP public descriptions**

Use these exact user-facing descriptions:

```python
help=(
    "Keepalive workload: mixed (default, drives Cube, Vector, and HBM), "
    "aicore, or vector."
)
```

Set MCP schema to:

```python
"workload": {
    "type": "string",
    "enum": ["mixed", "aicore", "vector"],
    "default": DEFAULT_WORKLOAD,
    "description": (
        "Mixed Cube, Vector, and HBM pressure by default; "
        "aicore and vector select single-engine compatibility modes."
    ),
}
```

Extend the schema's workload-specific minimum rule with a `mixed` branch using
`MIN_MIXED_BYTES`, while retaining the existing `aicore` minimum. Import the
constant from `single_npu_controller.workload`.

Change `legacy.py`'s translated status/config workload from hard-coded
`"aicore"` to `DEFAULT_WORKLOAD`; do not change the legacy tensor creation
path unless a failing test proves that executable uses the new controller.

- [ ] **Step 4: Run focused protocol tests and verify GREEN**

Run the command from Step 2.

Expected: all focused tests pass.

- [ ] **Step 5: Verify runtime-failure output still has no Rich traceback**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_cli_thresholds.py::test_blocking_command_reports_startup_failure_without_rich_traceback \
  tests/test_cli_thresholds.py::test_run_blocking_surfaces_worker_failure_and_releases_controller
```

Expected: both pass; the mixed default must not regress worker supervision.

- [ ] **Step 6: Commit protocol propagation**

```bash
git add \
  src/keep_npu/cli.py \
  src/keep_npu/legacy.py \
  src/keep_npu/mcp/server.py \
  tests/test_cli_thresholds.py \
  tests/test_cli_service_commands.py \
  tests/test_keep_npu_alive.py \
  tests/mcp/test_server.py \
  tests/mcp/test_http_api.py
git commit -m "feat: expose mixed workload as the default"
```

---

### Task 4: Update and rebuild the service dashboard

**Files:**
- Modify: `web/dashboard/src/App.jsx`
- Modify: `web/dashboard/src/lib/session.js`
- Modify: `web/dashboard/src/App.test.jsx`
- Modify: `web/dashboard/src/lib/session.test.js`
- Regenerate: `src/keep_npu/mcp/static/index.html`
- Regenerate: `src/keep_npu/mcp/static/assets/index.css`
- Regenerate: `src/keep_npu/mcp/static/assets/dashboard.js`
- Test: `web/dashboard/staticAssets.test.js`

**Interfaces:**
- Consumes: service workload values `mixed|aicore|vector`.
- Produces: default form payload `workload: "mixed"` and a three-option
  selector.
- Produces: packaged static assets built from the tested React source.

- [ ] **Step 1: Write failing dashboard default and validation tests**

Update `web/dashboard/src/lib/session.test.js`:

```javascript
it("builds a mixed workload payload by default", () => {
  expect(buildSessionPayload({
    npuIds: "",
    vram: "1GiB",
    interval: "300",
    busyThreshold: "25",
    workload: "mixed"
  })).toEqual({
    npu_ids: null,
    vram: "1GiB",
    interval: 300,
    busy_threshold: 25,
    workload: "mixed"
  })
})

it.each(["mixed", "aicore", "vector"])(
  "accepts the %s workload",
  (workload) => {
    expect(() => parseWorkload(workload)).not.toThrow()
  }
)
```

Add an `App.test.jsx` assertion that the selected option is
`Mixed Cube + Vector + HBM (default)` before user interaction and that the
other two options remain available.

- [ ] **Step 2: Run dashboard tests and verify RED**

Run:

```bash
npm test
```

from `web/dashboard`.

Expected: failures show the `aicore` default, two-value validation, and missing
mixed selector option.

- [ ] **Step 3: Update the dashboard source**

Change `defaultForm.workload` to `"mixed"`. Change the parser in
`src/lib/session.js` to:

```javascript
export function parseWorkload(value) {
  if (!["mixed", "aicore", "vector"].includes(value)) {
    throw new Error("Workload must be mixed, aicore, or vector")
  }
  return value
}
```

Render these options in `App.jsx`:

```jsx
<option value="mixed">Mixed Cube + Vector + HBM (default)</option>
<option value="aicore">AI Core / Cube only</option>
<option value="vector">Vector + HBM only</option>
```

Update session-record validation to accept exactly the same three values.

- [ ] **Step 4: Run dashboard tests and verify GREEN**

Run `npm test` from `web/dashboard`.

Expected: all Vitest suites pass.

- [ ] **Step 5: Build and verify packaged assets**

Run:

```bash
npm run build
npm test
```

The existing Vite configuration must copy the build into
`src/keep_npu/mcp/static`. Confirm `staticAssets.test.js` passes so source and
packaged assets do not drift.

- [ ] **Step 6: Commit dashboard source and generated assets**

```bash
git add \
  web/dashboard/src/App.jsx \
  web/dashboard/src/lib/session.js \
  web/dashboard/src/App.test.jsx \
  web/dashboard/src/lib/session.test.js \
  src/keep_npu/mcp/static/index.html \
  src/keep_npu/mcp/static/assets/index.css \
  src/keep_npu/mcp/static/assets/dashboard.js
git commit -m "feat: select mixed workload by default in dashboard"
```

---

### Task 5: Document the default and verify the complete local package

**Files:**
- Modify: `README.md`
- Modify: `docs/validation/ascend-remote-results.md`
- Modify if a release is requested separately: `src/keep_npu/__init__.py`
- Modify if a release is requested separately: `pyproject.toml`

**Interfaces:**
- Consumes: the finalized CLI and service behavior from Tasks 1–4.
- Produces: documentation that distinguishes engine utilization, HBM capacity,
  and HBM bandwidth.
- Does not bump or publish a version without separate release authorization.

- [ ] **Step 1: Update README commands and semantics**

Document the normal maximum-pressure command without a workload flag:

```console
keep-npu \
  --npu-ids 4,5,6,7 \
  --vram 1GiB \
  --busy-threshold -1
```

State that `--vram` is HBM capacity per device, while the mixed Vector feeder
drives HBM bandwidth. Add explicit compatibility examples:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --busy-threshold -1 --workload aicore
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --busy-threshold -1 --workload vector
```

Remove claims that `aicore` is the default, but keep historical validation
sections clearly dated.

- [ ] **Step 2: Run documentation/source consistency searches**

Run:

```bash
rg -n \
  "aicore \\(default|AI Core matmul by default|enum.*aicore.*vector|workload.*==.*aicore" \
  README.md docs src tests web/dashboard/src \
  --glob '!src/keep_npu/mcp/static/assets/*'
```

Classify every hit: update stale current behavior, retain explicit
compatibility tests, and retain dated historical records.

- [ ] **Step 3: Run the complete local verification**

Run:

```bash
git diff --check
.venv/bin/ruff check src tests
.venv/bin/black --check \
  src/keep_npu/utilities/session_config.py \
  src/keep_npu/single_npu_controller/workload.py \
  src/keep_npu/single_npu_controller/ascend_npu_controller.py \
  src/keep_npu/cli.py \
  src/keep_npu/legacy.py \
  src/keep_npu/mcp/server.py \
  tests/utilities/test_session_config.py \
  tests/test_workload.py \
  tests/test_ascend_backend.py \
  tests/test_cli_thresholds.py \
  tests/test_cli_service_commands.py \
  tests/test_keep_npu_alive.py \
  tests/mcp/test_server.py \
  tests/mcp/test_http_api.py
.venv/bin/pytest -q
(cd web/dashboard && npm test && npm run build)
```

Expected: Ruff and Black exit zero; all Python, subtests, and Vitest tests pass;
the dashboard production build exits zero.

- [ ] **Step 4: Build and inspect an installable wheel**

Run:

```bash
.venv/bin/python -m pip wheel . --no-deps --wheel-dir dist
.venv/bin/python -c \
  "import zipfile, glob; p=sorted(glob.glob('dist/keep_npu-*.whl'))[-1]; z=zipfile.ZipFile(p); assert any(n.endswith('mcp/static/assets/dashboard.js') for n in z.namelist()); print(p)"
```

Expected: one wheel is built and contains the regenerated dashboard asset.

- [ ] **Step 5: Commit local documentation**

```bash
git add README.md
git commit -m "docs: describe default mixed NPU pressure"
```

Do not commit `dist/`.

---

### Task 6: Tune and validate simultaneous utilization on Ascend 910B2

**Files:**
- Modify only if measurements require tuning:
  `src/keep_npu/single_npu_controller/ascend_npu_controller.py`
- Modify only with matching tests:
  `tests/test_ascend_backend.py`
- Modify: `docs/validation/ascend-remote-results.md`

**Interfaces:**
- Consumes: the exact wheel and commit produced by Task 5.
- Produces: raw ten-sample utilization evidence for every selected device and
  bounded-cleanup evidence.

- [ ] **Step 1: Audit target devices before starting**

On `npu0`, run:

```bash
npu-smi info
pgrep -af 'keep-npu|keep_npu' || true
df -h /
```

Use only devices with no running process. Record selected physical IDs and
pre-test HBM usage.

- [ ] **Step 2: Install the exact candidate wheel in an isolated environment**

Copy the wheel to a unique `/tmp/keep-npu-mixed-*` directory. Create a venv
with `--system-site-packages` using the validated VALOR Python so its matching
`torch_npu` and CANN runtime are reused:

```bash
/root/miniconda3/envs/VALOR/bin/python -m venv --system-site-packages "$TEST_DIR/venv"
"$TEST_DIR/venv/bin/python" -m pip install --no-deps "$TEST_DIR"/keep_npu-*.whl
"$TEST_DIR/venv/bin/keep-npu" --version
```

Do not install or upgrade `torch` or `torch_npu`.

- [ ] **Step 3: Start the normal default command**

Run without `--workload`:

```bash
"$TEST_DIR/venv/bin/keep-npu" \
  --npu-ids 4,5,6,7 \
  --vram 1GiB \
  --busy-threshold -1
```

Capture its PID and stderr/stdout. If any chosen device became busy after the
audit, stop cleanly and repeat on idle devices.

- [ ] **Step 4: Collect ten simultaneous post-warm-up samples**

After a 10-second warm-up, collect:

```bash
for sample in $(seq 1 10); do
  date '+sample=%s timestamp=%FT%T%z'
  for id in 4 5 6 7; do
    npu-smi info -t usages -i "$id"
  done
  sleep 1
done
```

Preserve the raw output. For every device and every accepted sample, verify:

- `Aicore Usage Rate(%) >= 90`;
- `Aivector Usage Rate(%) >= 90`;
- `HBM Bandwidth Usage Rate(%) >= 90`; and
- the KeepNPU worker has no runtime error.

- [ ] **Step 5: Tune one bounded variable at a time if acceptance fails**

Allowed tuning variables are:

- `MIXED_CUBE_CHUNK`;
- `MIXED_VECTOR_CHUNK`; and
- `MIXED_BATCH_SECONDS`; and
- the mixed-only matrix cap.

For each attempted change:

1. write or adjust a unit test that constrains queue bounds or shutdown;
2. run it RED;
3. change one constant;
4. run controller tests GREEN;
5. rebuild the wheel; and
6. repeat the ten-sample hardware measurement.

Do not change the Vector operator, stream count, public CLI, or HBM budget in
the same tuning attempt. Matrix shape may be tuned independently while
preserving the explicit `aicore` cap and the mixed Vector minimum.

- [ ] **Step 6: Verify bounded cleanup**

Send SIGTERM to the captured KeepNPU PID, wait up to 30 seconds, and verify:

```bash
pgrep -af 'keep-npu|keep_npu' || true
npu-smi info
```

The process must be gone, selected-device HBM must return to the pre-test
range, and no test listener or child process may remain.

- [ ] **Step 7: Record evidence and commit final validation**

Append to `docs/validation/ascend-remote-results.md`:

- candidate commit and wheel filename;
- server hostname, NPU model, CANN, torch, and torch_npu versions;
- exact command;
- raw or tabulated ten-sample values per device;
- any tuning sequence and rejected measurements;
- final cleanup audit.

Then run the complete Task 5 verification again and commit:

```bash
git add \
  src/keep_npu/single_npu_controller/ascend_npu_controller.py \
  tests/test_ascend_backend.py \
  docs/validation/ascend-remote-results.md
git commit -m "perf: validate mixed Ascend engine saturation"
```

If no source tuning was needed, add and commit only the validation document.
