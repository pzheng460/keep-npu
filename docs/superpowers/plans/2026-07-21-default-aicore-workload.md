# Default AI Core Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FP16 AI Core/Cube matrix multiplication the default KeepNPU workload so `nputop` UTL reaches near 100%, while retaining ReLU only as the explicit `vector` option.

**Architecture:** A focused workload module validates the `aicore|vector` choice and plans matrix/filler memory within the existing per-NPU byte budget. The single-device controller allocates and executes the selected plan; the global controller, CLI, service protocols, MCP schema, and dashboard propagate one normalized optional `workload` value whose default is `aicore`.

**Tech Stack:** Python 3.9–3.13, PyTorch + torch_npu, Typer, stdlib HTTP/JSON-RPC, pytest, React/Vite/Vitest, setuptools, GitHub Actions/PyPI.

## Global Constraints

- A normal command does not require a workload option; omitted workload always normalizes to `aicore`.
- `--workload vector` is the only way to select ReLU, and vector mode is never an automatic fallback.
- AI Core matrices are FP16, square, aligned to 16, initially capped at dimension 4096, and executed in batches of 32.
- Three AI Core matrices count inside `--vram`; filler may undershoot the rounded byte target by at most three bytes.
- AI Core mode requires at least 1536 bytes; vector mode retains the existing four-byte minimum.
- `--busy-threshold -1` runs unconditionally; thresholds `0..100` retain conservative total-utilization backoff.
- The ordinary maximum-UTL command remains `keep-npu --npu-ids 4,5,6,7 --vram 1GiB --interval 0.001 --busy-threshold -1`.
- Only selected visible ordinals may receive device contexts, processes, or allocations.
- Real-hardware success is repeated `nputop`/AI Core samples predominantly at least 95%, clean release, and no new process on unselected cards.

---

### Task 1: Workload validation and memory planning

**Files:**
- Create: `src/keep_npu/single_npu_controller/workload.py`
- Create: `tests/test_workload.py`
- Modify: `src/keep_npu/utilities/session_config.py`
- Modify: `tests/utilities/test_session_config.py`

**Interfaces:**
- Produces: `DEFAULT_WORKLOAD = "aicore"` and `validate_workload(value: object) -> str`.
- Produces: immutable `AICorePlan(matrix_dim: int, filler_elements: int, allocated_bytes: int)`.
- Produces: `plan_aicore_workload(float32_elements: int) -> AICorePlan`.
- Consumes: the existing internal float32-element count returned by `parse_vram_to_elements`.

- [ ] **Step 1: Write workload validation tests**

```python
@pytest.mark.parametrize("value", [None, "", "AICORE", "relu", 1, True])
def test_validate_workload_rejects_noncanonical_values(value):
    with pytest.raises(ValueError, match="workload must be 'aicore' or 'vector'"):
        validate_workload(value)


def test_validate_workload_accepts_public_values():
    assert validate_workload("aicore") == "aicore"
    assert validate_workload("vector") == "vector"
```

- [ ] **Step 2: Write AI Core planning tests**

```python
def test_minimum_aicore_plan_is_three_aligned_fp16_matrices():
    plan = plan_aicore_workload(1536 // 4)
    assert plan == AICorePlan(matrix_dim=16, filler_elements=0, allocated_bytes=1536)


def test_aicore_plan_rejects_budget_below_minimum():
    with pytest.raises(ValueError, match="aicore workload requires --vram of at least 1536 bytes"):
        plan_aicore_workload((1536 // 4) - 1)


def test_aicore_plan_is_aligned_capped_and_inside_budget():
    budget_elements = 1024**3 // 4
    plan = plan_aicore_workload(budget_elements)
    assert plan.matrix_dim == 4096
    assert plan.matrix_dim % 16 == 0
    assert budget_elements * 4 - 3 <= plan.allocated_bytes <= budget_elements * 4
    assert plan.allocated_bytes == 3 * 4096 * 4096 * 2 + plan.filler_elements * 4
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run: `PYTHONPATH=src python -m pytest tests/test_workload.py tests/utilities/test_session_config.py -q`

Expected: collection fails because `workload.py`, `validate_workload`, and `DEFAULT_WORKLOAD` do not exist.

- [ ] **Step 4: Implement the validator and pure planner**

```python
# session_config.py
DEFAULT_WORKLOAD = "aicore"


def validate_workload(value: object) -> str:
    if value not in {"aicore", "vector"}:
        raise ValueError("workload must be 'aicore' or 'vector'")
    return str(value)
```

```python
# workload.py
from dataclasses import dataclass
from math import isqrt

FP16_BYTES = 2
MATRIX_COUNT = 3
MATRIX_ALIGNMENT = 16
MAX_MATRIX_DIM = 4096
AICORE_BATCH_ITERATIONS = 32
MIN_AICORE_BYTES = MATRIX_COUNT * MATRIX_ALIGNMENT**2 * FP16_BYTES


@dataclass(frozen=True)
class AICorePlan:
    matrix_dim: int
    filler_elements: int
    allocated_bytes: int


def plan_aicore_workload(float32_elements: int) -> AICorePlan:
    budget_bytes = float32_elements * 4
    if budget_bytes < MIN_AICORE_BYTES:
        raise ValueError(
            f"aicore workload requires --vram of at least {MIN_AICORE_BYTES} bytes"
        )
    raw_dim = isqrt(budget_bytes // (MATRIX_COUNT * FP16_BYTES))
    matrix_dim = min(MAX_MATRIX_DIM, raw_dim)
    matrix_dim -= matrix_dim % MATRIX_ALIGNMENT
    matrix_bytes = MATRIX_COUNT * matrix_dim**2 * FP16_BYTES
    filler_elements = (budget_bytes - matrix_bytes) // 4
    allocated_bytes = matrix_bytes + filler_elements * 4
    return AICorePlan(matrix_dim, filler_elements, allocated_bytes)
```

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_workload.py tests/utilities/test_session_config.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the pure planning unit**

```bash
git add src/keep_npu/single_npu_controller/workload.py src/keep_npu/utilities/session_config.py tests/test_workload.py tests/utilities/test_session_config.py
git commit -m "feat: plan AI Core keepalive workloads"
```

---

### Task 2: Execute AI Core by default and Vector only on request

**Files:**
- Modify: `src/keep_npu/single_npu_controller/ascend_npu_controller.py`
- Modify: `src/keep_npu/global_npu_controller/global_npu_controller.py`
- Modify: `src/keep_npu/legacy.py`
- Modify: `tests/test_ascend_backend.py`
- Modify: `tests/test_global_npu_controller.py`
- Modify: `tests/test_keep_npu_alive.py`

**Interfaces:**
- Consumes: `validate_workload`, `DEFAULT_WORKLOAD`, `plan_aicore_workload`, and `AICORE_BATCH_ITERATIONS` from Task 1.
- Produces: `AscendNPUController(..., workload: str = DEFAULT_WORKLOAD)` and `GlobalNPUController(..., workload: str = DEFAULT_WORKLOAD)`.
- Produces: internal allocation shape `{"left", "right", "output", "fillers"}` for AI Core and the existing tensor list for Vector.

- [ ] **Step 1: Upgrade the fake backend and write default/opt-in execution tests**

```python
class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, count=2):
        self.npu = FakeNPU(count)
        self.allocations = []
        self.matmul_calls = 0
        self.relu_calls = 0

    def rand(self, *shape, **kwargs):
        tensor = {"shape": shape, **kwargs}
        self.allocations.append(tensor)
        return tensor

    def empty(self, *shape, **kwargs):
        tensor = {"shape": shape, **kwargs}
        self.allocations.append(tensor)
        return tensor

    def matmul(self, left, right, *, out):
        self.matmul_calls += 1
        return out
```

```python
def test_controller_defaults_to_aicore_matmul(monkeypatch):
    controller, fake = build_controller(monkeypatch, vram_to_keep="1MiB")
    controller.keep()
    controller.release()
    assert controller.workload == "aicore"
    assert fake.matmul_calls >= 1
    assert fake.relu_calls == 0


def test_controller_runs_relu_only_for_explicit_vector(monkeypatch):
    controller, fake = build_controller(
        monkeypatch, vram_to_keep=8, workload="vector", iterations=2
    )
    controller.keep()
    controller.release()
    assert fake.relu_calls >= 1
    assert fake.matmul_calls == 0
```

- [ ] **Step 2: Write memory, stop-bound, failure, and global propagation tests**

```python
def test_aicore_allocation_uses_selected_device_and_budget(monkeypatch):
    controller, fake = build_controller(monkeypatch, rank=1, vram_to_keep="1MiB")
    allocation = controller._allocate_aicore(controller._num_elements or 0)
    assert {tensor["device"] for tensor in fake.allocations} == {"npu:1"}
    assert allocation.plan.allocated_bytes >= 1024**2 - 3
    assert allocation.plan.allocated_bytes <= 1024**2


def test_aicore_batch_observes_stop_event(monkeypatch):
    controller, fake = build_controller(monkeypatch, vram_to_keep="1MiB")
    controller._stop_evt = threading.Event()
    fake.on_matmul = lambda calls: controller._stop_evt.set() if calls == 2 else None
    controller._run_aicore_batch(controller._allocate_aicore(controller.vram_to_keep))
    assert fake.matmul_calls == 2


def test_global_controller_passes_workload_to_each_device(monkeypatch):
    controller = GlobalNPUController(npu_ids=[0, 1], workload="vector")
    assert [item.workload for item in controller.controllers] == ["vector", "vector"]
```

- [ ] **Step 3: Run controller tests and confirm RED**

Run: `PYTHONPATH=src python -m pytest tests/test_ascend_backend.py tests/test_global_npu_controller.py tests/test_keep_npu_alive.py -q`

Expected: failures show missing workload parameters, AI Core allocators, and matmul dispatch.

- [ ] **Step 4: Implement workload-specific allocation and dispatch**

```python
@dataclass
class AICoreAllocation:
    left: object
    right: object
    output: object
    fillers: list[object]
    plan: AICorePlan


def _allocate_aicore(self, num_elements: int) -> AICoreAllocation:
    plan = plan_aicore_workload(num_elements)
    shape = (plan.matrix_dim, plan.matrix_dim)
    common = {"device": self.device, "dtype": self._torch.float16, "requires_grad": False}
    left = self._torch.rand(shape, **common)
    right = self._torch.rand(shape, **common)
    output = self._torch.empty(shape, **common)
    fillers = self._allocate_vector(plan.filler_elements)
    return AICoreAllocation(left, right, output, fillers, plan)


def _run_aicore_batch(self, allocation: AICoreAllocation) -> None:
    for _ in range(AICORE_BATCH_ITERATIONS):
        self._torch.matmul(allocation.left, allocation.right, out=allocation.output)
        if self._stop_evt is not None and self._stop_evt.is_set():
            break
    self._torch.npu.synchronize()


def _run_batch(self, allocation) -> None:
    if self.workload == "aicore":
        self._run_aicore_batch(allocation)
    else:
        self._run_vector_batch(allocation)
```

Rename the existing `_allocate` and ReLU `_run_batch` bodies to
`_allocate_vector` and `_run_vector_batch`. Validate `workload` in each
controller constructor, select the allocator once after utilization gating,
and pass the normalized value through the global controller. The legacy
wrapper must request `workload="aicore"` explicitly inside its translated
controller kwargs so its matrix-oriented historical behavior remains clear.

- [ ] **Step 5: Run controller tests and confirm GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_ascend_backend.py tests/test_global_npu_controller.py tests/test_keep_npu_alive.py -q`

Expected: all selected tests pass, including startup rollback and release cases.

- [ ] **Step 6: Commit controller execution**

```bash
git add src/keep_npu/single_npu_controller src/keep_npu/global_npu_controller src/keep_npu/legacy.py tests/test_ascend_backend.py tests/test_global_npu_controller.py tests/test_keep_npu_alive.py
git commit -m "feat: run AI Core keepalive by default"
```

---

### Task 3: Propagate workload through CLI and service protocols

**Files:**
- Modify: `src/keep_npu/cli.py`
- Modify: `src/keep_npu/mcp/server.py`
- Modify: `tests/test_cli_thresholds.py`
- Modify: `tests/test_cli_service_commands.py`
- Modify: `tests/mcp/test_server.py`

**Interfaces:**
- Consumes: `DEFAULT_WORKLOAD` and `validate_workload` from Task 1, plus controller constructor parameters from Task 2.
- Produces: root and `start` CLI option `--workload [aicore|vector]`, defaulting to `aicore`.
- Produces: optional `workload` on `start_keep`, REST session creation, JSON-RPC, and MCP; session `params.workload` is always normalized.

- [ ] **Step 1: Write blocking and service CLI contract tests**

```python
def test_blocking_mode_defaults_to_aicore(monkeypatch):
    captured = install_fake_blocking_runner(monkeypatch)
    result = runner.invoke(cli.app, ["--npu-ids", "0", "--vram", "1MiB"])
    assert result.exit_code == 0
    assert captured["workload"] == "aicore"


def test_start_accepts_explicit_vector(monkeypatch):
    captured = install_fake_rpc(monkeypatch)
    result = runner.invoke(cli.app, ["start", "--workload", "vector"])
    assert result.exit_code == 0
    assert captured["params"]["workload"] == "vector"


def test_cli_rejects_unknown_workload_without_traceback(monkeypatch):
    result = runner.invoke(cli.app, ["--workload", "relu"])
    assert result.exit_code != 0
    assert "workload must be 'aicore' or 'vector'" in result.output
    assert "Traceback" not in result.output
```

- [ ] **Step 2: Write direct, REST, JSON-RPC, and MCP schema tests**

```python
def test_start_keep_defaults_and_reports_aicore():
    job_id = server.start_keep(npu_ids=[0])["job_id"]
    assert server.status(job_id)["params"]["workload"] == "aicore"
    assert controller.workload == "aicore"


def test_jsonrpc_start_keep_accepts_vector():
    response = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "start_keep",
        "params": {"npu_ids": [0], "workload": "vector"},
    })
    assert response["result"]["job_id"]
    assert server.status(response["result"]["job_id"])["params"]["workload"] == "vector"


def test_mcp_schema_declares_workload_enum_and_default():
    schema = start_keep_tool()["inputSchema"]["properties"]["workload"]
    assert schema["enum"] == ["aicore", "vector"]
    assert schema["default"] == "aicore"
```

- [ ] **Step 3: Run interface tests and confirm RED**

Run: `PYTHONPATH=src python -m pytest tests/test_cli_thresholds.py tests/test_cli_service_commands.py tests/mcp/test_server.py -q`

Expected: tests fail because the CLI signatures, allowed parameter sets, schemas, and session records lack `workload`.

- [ ] **Step 4: Implement CLI and protocol propagation**

Add a shared CLI validator that wraps `validate_workload` in `typer.BadParameter`.
Extend `_run_blocking`, the root callback, and `start` with:

```python
workload: str = typer.Option(
    DEFAULT_WORKLOAD,
    "--workload",
    help="Keepalive workload: aicore (default, drives nputop UTL) or vector.",
)
```

Validate before hardware/service startup and pass `workload` into
`GlobalNPUController` or RPC params. Extend the server direct-method allowed
field sets, REST field set, MCP schema, `start_keep` signature, controller
factory call, and session snapshot:

```python
workload = validate_workload(workload)
params = {
    "npu_ids": npu_ids,
    "vram": vram,
    "interval": interval,
    "busy_threshold": busy_threshold,
    "workload": workload,
}
```

- [ ] **Step 5: Run interface tests and confirm GREEN**

Run: `PYTHONPATH=src python -m pytest tests/test_cli_thresholds.py tests/test_cli_service_commands.py tests/mcp/test_server.py -q`

Expected: all selected tests pass and machine JSON remains exactly one object.

- [ ] **Step 6: Commit public interface propagation**

```bash
git add src/keep_npu/cli.py src/keep_npu/mcp/server.py tests/test_cli_thresholds.py tests/test_cli_service_commands.py tests/mcp/test_server.py
git commit -m "feat: expose optional vector workload"
```

---

### Task 4: Add dashboard selection and user documentation

**Files:**
- Modify: `web/dashboard/src/App.jsx`
- Modify: `web/dashboard/src/lib/session.js`
- Modify: `web/dashboard/src/lib/session.test.js`
- Modify: `web/dashboard/src/lib/refresh.js`
- Modify: `web/dashboard/src/lib/refresh.test.js`
- Regenerate: `src/keep_npu/mcp/static/index.html`
- Regenerate: `src/keep_npu/mcp/static/assets/*`
- Modify: `README.md`
- Modify: `docs/compatibility.md`

**Interfaces:**
- Consumes: service field `workload: "aicore" | "vector"` from Task 3.
- Produces: dashboard form field `form.workload`, REST payload field `workload`, and session validation/display.

- [ ] **Step 1: Write dashboard payload and response-validation tests**

```javascript
it("defaults new sessions to aicore", () => {
  expect(buildSessionPayload({
    npuIds: "4,5", vram: "1GiB", interval: "0.001",
    busyThreshold: "-1", workload: "aicore"
  })).toEqual({
    npu_ids: [4, 5], vram: "1GiB", interval: 0.001,
    busy_threshold: -1, workload: "aicore"
  })
})

it("rejects malformed session workload values", async () => {
  const bad = {...validSessionRecord, params: {...validSessionRecord.params, workload: "relu"}}
  expect(isValidSessionRecord(bad)).toBe(false)
})
```

- [ ] **Step 2: Run dashboard tests and confirm RED**

Run: `cd web/dashboard && npm test -- --run`

Expected: workload payload/record assertions fail.

- [ ] **Step 3: Implement selector, payload, validation, and display**

Add `workload: "aicore"` to `INITIAL_FORM`, include it in
`buildSessionPayload`, accept only `aicore|vector` in session records, render a
two-choice `<select>` labelled `Workload`, and display the normalized workload
beside each session. The choices are:

```jsx
<select value={form.workload} onChange={(event) => updateForm("workload", event.target.value)}>
  <option value="aicore">AI Core (default)</option>
  <option value="vector">Vector (lightweight)</option>
</select>
```

- [ ] **Step 4: Document behavior and maximum-UTL command**

Update README and compatibility documentation to state:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --interval 0.001 --busy-threshold -1
```

uses default FP16 AI Core/Cube matmul and should raise `nputop` UTL, while
`--workload vector` opts into ReLU and may show zero `nputop` UTL. Retain the
existing explanation that `torch` and `torch_npu` must match CANN and are not
installed automatically.

- [ ] **Step 5: Test and rebuild production dashboard assets**

Run: `cd web/dashboard && npm test -- --run && npm run build`

Expected: all Vitest tests pass and Vite refreshes `src/keep_npu/mcp/static`.

Run: `PYTHONPATH=src python -m pytest tests/mcp/test_server.py -q`

Expected: packaged static-asset tests pass.

- [ ] **Step 6: Commit dashboard and documentation**

```bash
git add web/dashboard/src src/keep_npu/mcp/static README.md docs/compatibility.md
git commit -m "docs: expose AI Core and vector workload behavior"
```

---

### Task 5: Local regression, installed-wheel verification, and version 1.0.3

**Files:**
- Modify: `src/keep_npu/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all implementation and documentation from Tasks 1–4.
- Produces: release artifacts for `keep-npu==1.0.3`.

- [ ] **Step 1: Run the complete local quality gate**

Run:

```bash
PYTHONPATH=src python -m pytest -q
ruff check .
python -m compileall -q src tests keep_npu_alive.py
cd web/dashboard && npm test -- --run && npm run build
```

Expected: every command exits 0 and the asset rebuild leaves no unexpected diff.

- [ ] **Step 2: Bump the patch version without creating an automatic tag**

Change `src/keep_npu/__init__.py` to `__version__ = "1.0.3"` and
`pyproject.toml` to `current_version = "1.0.3"` using `apply_patch` so release
tagging remains a separate verified action.

- [ ] **Step 3: Build and inspect artifacts**

Run:

```bash
python -m build
python -m zipfile -l dist/keep_npu-1.0.3-py3-none-any.whl
```

Expected: both `dist/keep_npu-1.0.3.tar.gz` and the wheel exist; the wheel
contains the package, CLI entry points, and dashboard assets.

- [ ] **Step 4: Test a clean installed wheel**

Run:

```bash
release_venv=$(mktemp -d)/venv
python -m venv "$release_venv"
"$release_venv/bin/python" -m pip install --no-cache-dir dist/keep_npu-1.0.3-py3-none-any.whl
"$release_venv/bin/keep-npu" --help
"$release_venv/bin/keep-npu" start --help
"$release_venv/bin/python" -c 'import keep_npu; assert keep_npu.__version__ == "1.0.3"'
```

Expected: installation succeeds, both help pages show workload behavior, and
the imported version is 1.0.3. Hardware startup is not attempted locally.

- [ ] **Step 5: Commit the release version**

```bash
git add src/keep_npu/__init__.py pyproject.toml
git commit -m "chore: prepare 1.0.3"
```

---

### Task 6: Ascend 910B2 tuning and device-isolation acceptance

**Files:**
- Modify if tuning is required: `src/keep_npu/single_npu_controller/workload.py`
- Modify if tuning is required: `tests/test_workload.py`
- Modify: `docs/validation/ascend-remote-results.md`

**Interfaces:**
- Consumes: built 1.0.3 candidate and the existing SSH host alias `npu0`.
- Produces: recorded AI Core/UTL, memory, device-process, backoff, and cleanup evidence.

- [ ] **Step 1: Preflight the remote environment read-only**

Run remote checks for hostname, `npu-smi info`, the previously identified
`/data2/RL-Framework/conda_envs/embodied-ai/bin/python`, `torch.__version__`,
`torch_npu.__version__`, visible device count, and active device processes.

Expected: eight visible Ascend 910B2 devices and no KeepNPU process from this test.

- [ ] **Step 2: Deploy to a unique recoverable temporary directory**

Create the directory with remote `mktemp -d`, copy the source tree or wheel,
and install only into the known compatible environment or a temporary venv
that can import its system `torch_npu`. Record the exact directory and PID for
cleanup; do not modify the shared environment's installed KeepNPU package.

- [ ] **Step 3: Test one idle device and sample UTL**

Run the unchanged default command on one currently idle visible ordinal with
1 GiB, 0.001-second interval, and threshold -1. Take at least five post-warmup
samples from the AICore metric used by `nputop`, total utilization, HBM use, and
the per-device process list.

Expected: at least four of five UTL samples are at least 95%, only the selected
device contains the test PID, and allocated memory is approximately 1 GiB plus
documented runtime overhead.

- [ ] **Step 4: Tune internal constants only if acceptance fails**

If fewer than four samples reach 95%, change only `MAX_MATRIX_DIM` and/or
`AICORE_BATCH_ITERATIONS`, add/update exact constant tests, run their RED/GREEN
cycle locally, redeploy, and repeat Step 3. Do not add another public option.

- [ ] **Step 5: Test devices 4,5,6,7 together**

Run:

```console
keep-npu --npu-ids 4,5,6,7 --vram 1GiB --interval 0.001 --busy-threshold -1
```

Expected: each selected device meets the repeated 95% UTL criterion; devices
0–3 gain no process belonging to the test command.

- [ ] **Step 6: Test vector opt-in, safe backoff, and cleanup**

Run a short `--workload vector` smoke test, then a default-threshold AI Core
test against a busy device to confirm backoff. Terminate with SIGTERM and
Ctrl+C paths, verify PIDs disappear, verify memory returns to its pre-test
range, and remove only the recorded temporary directory.

- [ ] **Step 7: Record and commit reproducible evidence**

Append remote versions, exact commands, selected devices, five-sample tables,
process isolation, before/after memory, signal results, and cleanup audit to
`docs/validation/ascend-remote-results.md`. If constants changed, rebuild and
repeat Task 5's full gate before committing.

```bash
git add src/keep_npu/single_npu_controller/workload.py tests/test_workload.py docs/validation/ascend-remote-results.md
git commit -m "test: validate AI Core utilization on Ascend"
```

---

### Task 7: Final review, publish, and public pip verification

**Files:**
- No planned source changes; review fixes must receive their own tests and commit.

**Interfaces:**
- Consumes: verified 1.0.3 branch from Tasks 1–6.
- Produces: merged GitHub release `v1.0.3` and publicly installable `keep-npu==1.0.3`.

- [ ] **Step 1: Run verification-before-completion checks**

Repeat the complete Python/dashboard/build/installed-wheel gate from Task 5
after confirming `git status --short` contains only intentional files.

- [ ] **Step 2: Request code review and address only evidence-backed findings**

Use `superpowers:requesting-code-review`; for each actionable issue, reproduce
it, add a failing test, implement the smallest correction, rerun the focused
and full suites, and commit separately.

- [ ] **Step 3: Publish through the repository's normal release workflow**

Push the reviewed branch, open and merge the pull request, create tag/release
`v1.0.3`, and monitor `.github/workflows/publish-pypi.yml` until the trusted
publishing job succeeds. Never upload with an ad-hoc credential.

- [ ] **Step 4: Verify the public index without cache**

In a fresh temporary venv run:

```bash
python -m pip install -U --no-cache-dir keep-npu==1.0.3
keep-npu --version
keep-npu --help
```

Expected: the public index resolves 1.0.3, the reported version is 1.0.3, and
help shows AI Core as the default with Vector opt-in.

- [ ] **Step 5: Report the exact user command and evidence**

Hand off the unchanged devices 4–7 command, the observed UTL range, the remote
environment used, release URL/version, and confirmation that no test process or
temporary deployment remains.
