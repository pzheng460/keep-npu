# KeepGPU-Compatible KeepNPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Ascend-native `keep-npu` package whose renamed public interfaces and lifecycle behavior match KeepGPU 1.0.0/current compatibility fixes, then validate it on configured SSH NPU hosts.

**Architecture:** Port the tested KeepGPU package, service, dashboard, and test boundaries with a deterministic GPU-to-NPU naming map. Replace the platform-specific controllers and telemetry with one dependency-injected Ascend backend using `torch_npu` and `npu-smi`, while retaining the upstream validation, protocol, concurrency, and daemon-safety contracts.

**Tech Stack:** Python 3.9–3.13, setuptools, Typer, Rich, PyTorch + torch_npu, stdlib HTTP/JSON-RPC, React/Vite/Vitest/Tailwind, pytest.

## Global Constraints

- Reference behavior is KeepGPU upstream commit `691720383f5325cffb9fc5960304541caec31444` and release 1.0.0.
- Public names use `keep-npu`, `keep-npu-mcp-server`, `--npu-ids`, `list-npus`, `npu_ids`, `npus`, and `~/.keepnpu`.
- Default service endpoint is `127.0.0.1:8765`.
- Unknown utilization is `null` and causes backoff unless `busy_threshold=-1`.
- Remote work uses unique temporary directories and leaves no daemon or workload behind.
- Production behavior is introduced only after a corresponding test has been observed failing.

---

### Task 1: Port the public contract and package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/keep_npu/__init__.py`
- Create: `src/keep_npu/utilities/*.py`
- Create: `tests/utilities/*.py`
- Create: `tests/test_package_metadata.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: validators `validate_interval`, `validate_busy_threshold`, `validate_npu_ids`, `validate_job_id`; `parse_size`; strict JSON helpers; endpoint validators.
- Consumes: no project interfaces.

- [ ] Copy the upstream metadata and utility tests, mechanically map the approved public names, and run `pytest tests/utilities tests/test_package_metadata.py -q`; expect collection/import failure because `keep_npu` does not exist.
- [ ] Add the mapped package metadata and utility modules, keeping numeric, size, endpoint, session-id, and strict-JSON limits unchanged.
- [ ] Run `pytest tests/utilities tests/test_package_metadata.py -q`; expect all selected tests to pass.
- [ ] Run `python -m build`; expect wheel and source distribution creation with both console entry points declared.
- [ ] Commit with `feat: add KeepNPU package contracts`.

### Task 2: Implement Ascend discovery, telemetry, and controllers

**Files:**
- Create: `src/keep_npu/utilities/platform_manager.py`
- Create: `src/keep_npu/utilities/npu_info.py`
- Create: `src/keep_npu/utilities/npu_monitor.py`
- Create: `src/keep_npu/single_npu_controller/base_npu_controller.py`
- Create: `src/keep_npu/single_npu_controller/ascend_npu_controller.py`
- Create: `src/keep_npu/global_npu_controller/global_npu_controller.py`
- Create: `tests/npu_controller/*.py`
- Create: `tests/global_controller/*.py`
- Create: `tests/fixtures/npu_smi/*.txt`

**Interfaces:**
- Consumes: Task 1 validators and size parsing.
- Produces: `visible_torch_device_count()`, `list_npus()`, `NPUInfo`, `AscendNPUController`, and `GlobalNPUController`.

- [ ] Port controller contract, timing, rollback, release, and health tests with fake Torch/NPU adapters; run them and expect import failure for controller modules.
- [ ] Add `npu-smi` fixture tests for table parsing, missing fields, invalid utilization, inconsistent memory, and command failure; run and expect missing parser failures.
- [ ] Implement lazy `torch`/`torch_npu` discovery and visible ordinal selection without importing hardware libraries during local validation.
- [ ] Implement telemetry parsing that returns nullable fields and never converts unavailable utilization to zero.
- [ ] Implement chunked float32 allocation, low-cost keepalive work, monotonic interval timing, utilization backoff, runtime health, idempotent keep/release, multi-device rollback, and parallel release.
- [ ] Run `pytest tests/npu_controller tests/global_controller tests/utilities/test_npu_info.py tests/utilities/test_npu_monitor.py -q`; expect all selected tests to pass.
- [ ] Commit with `feat: add Ascend NPU controllers and telemetry`.

### Task 3: Port blocking CLI and legacy wrapper

**Files:**
- Create: `src/keep_npu/cli.py`
- Create: `tests/test_cli_thresholds.py`
- Create: `tests/test_cli_blocking.py`
- Modify: `keep_npu_alive.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `GlobalNPUController`, Task 1 validators.
- Produces: `keep-npu` blocking mode and compatibility wrapper behavior.

- [ ] Port the upstream blocking CLI tests with `--npu-ids`; add regression cases for `nan`, `inf`, duplicate IDs, signed zero, full-width digits, and root options before subcommands; run and expect CLI import/behavior failures.
- [ ] Implement the Typer root command with defaults `interval=300`, `vram=1GiB`, `busy_threshold=25`, all-visible NPUs, legacy `--threshold`, and plain validation errors.
- [ ] Convert `keep_npu_alive.py` into a wrapper that translates `--device`, `--interval`, and the legacy matrix-size flags into the new controller path while retaining `--once` as a bounded compatibility smoke operation.
- [ ] Run `pytest tests/test_cli_thresholds.py tests/test_cli_blocking.py tests/test_keep_npu_alive.py -q`; expect all selected tests to pass.
- [ ] Run `PYTHONPATH=src python -m keep_npu.cli --help`; expect mapped NPU options and all service subcommands.
- [ ] Commit with `feat: add KeepNPU blocking CLI`.

### Task 4: Port service lifecycle and service CLI

**Files:**
- Create: `src/keep_npu/mcp/server.py`
- Create: `tests/mcp/test_server.py`
- Create: `tests/test_cli_service_commands.py`

**Interfaces:**
- Consumes: `GlobalNPUController`, `list_npus()`, all shared validators.
- Produces: `KeepNPUServer.start_keep`, `stop_keep`, `status`, `list_npus`; CLI `serve/start/status/stop/list-npus/service-stop`.

- [ ] Port the upstream session concurrency, starting/stopping, runtime failure, release timeout, and status contract tests with the naming map; run and expect missing server failures.
- [ ] Port service CLI tests for local validation, auto-start rollback, machine JSON purity, malformed RPC envelopes, ownership records, and safe daemon stop; run and expect missing commands.
- [ ] Implement the shared session registry and lifecycle states without holding the registry lock during slow controller startup or health probes.
- [ ] Implement CLI RPC clients, result validators, daemon auto-start, endpoint-specific logs/PID records, ownership verification, stop-all fallback, and command hints.
- [ ] Run `pytest tests/mcp/test_server.py tests/test_cli_service_commands.py -q`; expect all selected tests to pass.
- [ ] Commit with `feat: add KeepNPU service workflows`.

### Task 5: Port REST, JSON-RPC, MCP, and dashboard

**Files:**
- Create: `tests/mcp/test_http_api.py`
- Create: `web/dashboard/**`
- Create: `src/keep_npu/mcp/static/index.html`
- Create: `src/keep_npu/mcp/static/assets/*`

**Interfaces:**
- Consumes: `KeepNPUServer` direct methods.
- Produces: REST `/api/npus` and session endpoints, JSON-RPC `/` and `/rpc`, stdio MCP lifecycle/tools, and the KeepNPU dashboard.

- [ ] Port HTTP routing, strict request-body, JSON-RPC envelope, MCP lifecycle, method/error classification, HEAD/static asset, and dashboard API tests; run and expect route/asset failures.
- [ ] Implement HTTP and stdio transports with exact NPU-native result keys and upstream route-hardening behavior.
- [ ] Port dashboard source and tests with NPU labels and `/api/npus`; run `npm test -- --run` in `web/dashboard` and expect failures before mapped components are complete.
- [ ] Complete the mapped React dashboard, run `npm run build`, and synchronize built assets into `src/keep_npu/mcp/static`.
- [ ] Run `pytest tests/mcp -q` and `npm test -- --run`; expect all selected tests to pass.
- [ ] Commit with `feat: add KeepNPU APIs and dashboard`.

### Task 6: Full compatibility and packaging verification

**Files:**
- Create/modify: mapped remaining tests under `tests/`
- Create: `docs/compatibility.md`
- Modify: `README.md`
- Modify: `MANIFEST.in`

**Interfaces:**
- Consumes: all prior public interfaces.
- Produces: reproducible compatibility evidence and installable artifacts.

- [ ] Port every remaining hardware-independent upstream test and run the full suite; record and fix each mapped contract failure using a failing regression test first.
- [ ] Run `python -m pytest -q`, `ruff check .`, and `python -m compileall -q src tests keep_npu_alive.py`; expect clean output.
- [ ] Build wheel/sdist, install the wheel into a temporary virtual environment, and smoke-test `keep-npu --help` plus `keep-npu-mcp-server --help`.
- [ ] Build dashboard production assets and verify packaged assets match the React build.
- [ ] Document the explicit name mapping, supported Python/CANN expectations, CLI/API examples, and known platform-only differences.
- [ ] Commit with `test: complete KeepGPU compatibility coverage`.

### Task 7: SSH Ascend validation and hardware corrections

**Files:**
- Create: `docs/validation/ascend-remote-results.md`
- Create/modify: telemetry fixtures and regression tests discovered from hardware.
- Modify: hardware modules only when a failing remote case is reproduced locally.

**Interfaces:**
- Consumes: built KeepNPU artifact and configured SSH hosts.
- Produces: remote evidence, platform corrections, and cleanup verification.

- [ ] Discover SSH host aliases read-only and preflight each reachable candidate with bounded commands for `npu-smi`, Python, CANN variables, `torch`, `torch_npu`, and device count.
- [ ] Select at least two usable environments, or multiple devices on one host, based on availability and current memory pressure.
- [ ] Rsync only source, tests, metadata, docs, and built dashboard assets to a unique `/tmp/keep-npu-<timestamp>` directory on each target.
- [ ] Run remote unit smoke tests, `list-npus`, a low-memory short blocking session, and before/during/after telemetry; use explicit timeouts and capture logs.
- [ ] Run service lifecycle, REST, JSON-RPC, MCP, dashboard endpoint, multi-NPU, invalid-selection, signal, repeated-stop, and rollback checks.
- [ ] For each hardware discrepancy, add a local failing fixture/regression test, implement the smallest correction, rerun locally, redeploy, and rerun the affected remote check.
- [ ] Stop all test sessions and services, verify no KeepNPU process remains, verify memory recovery, and remove each remote temporary directory.
- [ ] Write exact host/runtime/device/version/result evidence and any irreducible Ascend differences to `docs/validation/ascend-remote-results.md`.
- [ ] Run the complete local verification suite once more and commit with `test: validate KeepNPU on Ascend hardware`.
