import logging
import os

import pytest
import typer
from typer.testing import CliRunner

from keep_npu import cli

runner = CliRunner()


def test_parse_npu_ids_treats_omitted_as_all_visible():
    assert cli._parse_npu_ids(None) is None


@pytest.mark.parametrize("npu_ids", ["", "   "])
def test_parse_npu_ids_rejects_empty_values(npu_ids):
    with pytest.raises(typer.BadParameter, match="npu_ids must not be empty"):
        cli._parse_npu_ids(npu_ids)


def test_parse_npu_ids_rejects_negative_values():
    with pytest.raises(typer.BadParameter, match="non-negative integers"):
        cli._parse_npu_ids("0,-1")


def test_parse_npu_ids_rejects_duplicate_values():
    with pytest.raises(typer.BadParameter, match="duplicate values"):
        cli._parse_npu_ids("0,1,0")


@pytest.mark.parametrize("npu_ids", ["1_000", "１２３", "+0", "-0", "-00"])
def test_parse_npu_ids_rejects_non_canonical_numeric_tokens(npu_ids):
    with pytest.raises(typer.BadParameter, match="Invalid characters in --npu-ids"):
        cli._parse_npu_ids(npu_ids)


@pytest.mark.parametrize("interval", ["1_000", "１２３", "+1"])
def test_validate_cli_interval_rejects_non_canonical_numeric_tokens(interval):
    with pytest.raises(
        typer.BadParameter, match="interval must be finite and positive"
    ):
        cli._validate_cli_interval(interval)


def test_validate_cli_interval_preserves_exponent_plus_sign():
    assert cli._validate_cli_interval("1e+3") == 1000


@pytest.mark.parametrize("busy_threshold", ["1_0", "１２", "+25", "-0"])
def test_validate_cli_busy_threshold_rejects_non_canonical_numeric_tokens(
    busy_threshold,
):
    with pytest.raises(
        typer.BadParameter,
        match="busy_threshold must be -1 or an integer between 0 and 100",
    ):
        cli._validate_cli_busy_threshold(busy_threshold)


def test_validate_cli_workload_accepts_public_values():
    assert cli._validate_cli_workload("aicore") == "aicore"
    assert cli._validate_cli_workload("vector") == "vector"


@pytest.mark.parametrize("workload", ["", "AICORE", "relu"])
def test_validate_cli_workload_rejects_unknown_values(workload):
    with pytest.raises(typer.BadParameter, match="workload must be"):
        cli._validate_cli_workload(workload)


def test_apply_legacy_threshold_none():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", None, -1)
    assert vram == "1GiB"
    assert threshold == -1
    assert mode is None


def test_apply_legacy_threshold_numeric():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", "25", -1)
    assert vram == "1GiB"
    assert threshold == 25
    assert mode == "busy"


@pytest.mark.parametrize("legacy_threshold", ["1_0", "１２", "+25", "-0"])
def test_apply_legacy_threshold_rejects_non_canonical_numeric_tokens(
    legacy_threshold,
):
    with pytest.raises(
        typer.BadParameter,
        match="threshold must be an integer utilization value or a VRAM size",
    ):
        cli._apply_legacy_threshold("1GiB", legacy_threshold, -1)


def test_apply_legacy_threshold_memory_string():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", "2GiB", -1)
    assert vram == "2GiB"
    assert threshold == -1
    assert mode == "vram"


def test_validate_cli_busy_threshold_rejects_legacy_value_above_percent_range():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", "101", -1)

    assert vram == "1GiB"
    assert threshold == 101
    assert mode == "busy"
    with pytest.raises(
        typer.BadParameter,
        match="busy_threshold must be -1 or an integer between 0 and 100",
    ):
        cli._validate_cli_busy_threshold(threshold)


def test_blocking_command_rejects_empty_npu_ids_before_run_blocking(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_blocking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocking mode should not start")
        ),
    )

    result = runner.invoke(cli.app, ["--npu-ids", ""])

    assert result.exit_code == 1
    assert "npu_ids must not be empty" in result.output


def test_blocking_command_accepts_fractional_interval(monkeypatch):
    captured = {}

    def fake_run_blocking(
        interval, npu_ids, vram, legacy_threshold, busy_threshold, workload
    ):
        captured["interval"] = interval
        captured["npu_ids"] = npu_ids
        captured["vram"] = vram
        captured["legacy_threshold"] = legacy_threshold
        captured["busy_threshold"] = busy_threshold
        captured["workload"] = workload

    monkeypatch.setattr(cli, "_run_blocking", fake_run_blocking)

    result = runner.invoke(cli.app, ["--interval", "0.5"])

    assert result.exit_code == 0
    assert captured["interval"] == 0.5
    assert captured["workload"] == "aicore"


def test_blocking_command_accepts_explicit_vector_workload(monkeypatch):
    captured = {}

    def fake_run_blocking(*args):
        captured["workload"] = args[-1]

    monkeypatch.setattr(cli, "_run_blocking", fake_run_blocking)

    result = runner.invoke(cli.app, ["--workload", "vector"])

    assert result.exit_code == 0
    assert captured["workload"] == "vector"


def test_blocking_command_reports_startup_failure_without_rich_traceback(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_blocking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Failed to import torch/torch_npu. Install Ascend PyTorch first."
            )
        ),
    )

    result = runner.invoke(cli.app, ["--npu-ids", "4,5,6,7"])

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Error: Failed to import torch/torch_npu. Install Ascend PyTorch first."
    )
    assert "Traceback" not in result.output
    assert "╭" not in result.output


def test_run_blocking_preserves_ascend_visible_devices_for_npu_ids(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7")
    captured = {}

    class DummyGlobalController:
        def __init__(
            self, *, npu_ids, interval, vram_to_keep, busy_threshold, workload
        ):
            captured["npu_ids"] = npu_ids
            captured["interval"] = interval
            captured["vram_to_keep"] = vram_to_keep
            captured["busy_threshold"] = busy_threshold
            captured["workload"] = workload

        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True

    import keep_npu.global_npu_controller.global_npu_controller as global_module

    monkeypatch.setattr(global_module, "GlobalNPUController", DummyGlobalController)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)

    cli._run_blocking(
        interval=1,
        npu_ids="3",
        vram="1MiB",
        legacy_threshold=None,
        busy_threshold=-1,
    )

    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "7"
    assert captured == {
        "npu_ids": [3],
        "interval": 1,
        "vram_to_keep": "1MiB",
        "busy_threshold": -1,
        "workload": "aicore",
        "entered": True,
        "exited": True,
    }


def test_run_blocking_installs_and_restores_sigterm_handler(monkeypatch):
    captured = {"signal_calls": []}

    class DummyGlobalController:
        def __init__(self, **_kwargs):
            assert captured["signal_calls"], "SIGTERM must be handled during startup"

        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True

    import keep_npu.global_npu_controller.global_npu_controller as global_module

    previous_handler = object()

    def fake_getsignal(signum):
        assert signum == cli.signal.SIGTERM
        return previous_handler

    def fake_signal(signum, handler):
        captured["signal_calls"].append((signum, handler))

    def terminate_sleep(_seconds):
        installed = captured["signal_calls"][0][1]
        installed(cli.signal.SIGTERM, None)

    monkeypatch.setattr(global_module, "GlobalNPUController", DummyGlobalController)
    monkeypatch.setattr(cli.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(cli.signal, "signal", fake_signal)
    monkeypatch.setattr(cli.time, "sleep", terminate_sleep)

    cli._run_blocking(1, "0", "1MiB", None, -1)

    assert captured["entered"] is True
    assert captured["exited"] is True
    assert captured["signal_calls"][0][0] == cli.signal.SIGTERM
    assert captured["signal_calls"][-1] == (cli.signal.SIGTERM, previous_handler)


def test_run_blocking_defers_omitted_npu_enumeration_to_global_controller(
    monkeypatch,
):
    captured = {}

    class DummyGlobalController:
        def __init__(
            self, *, npu_ids, interval, vram_to_keep, busy_threshold, workload
        ):
            captured["npu_ids"] = npu_ids
            captured["interval"] = interval
            captured["vram_to_keep"] = vram_to_keep
            captured["busy_threshold"] = busy_threshold
            captured["workload"] = workload

        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True

    import keep_npu.global_npu_controller.global_npu_controller as global_module

    monkeypatch.setattr(global_module, "GlobalNPUController", DummyGlobalController)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)

    cli._run_blocking(
        interval=1,
        npu_ids=None,
        vram="1MiB",
        legacy_threshold=None,
        busy_threshold=-1,
    )

    assert captured == {
        "npu_ids": None,
        "interval": 1,
        "vram_to_keep": "1MiB",
        "busy_threshold": -1,
        "workload": "aicore",
        "entered": True,
        "exited": True,
    }


@pytest.mark.parametrize(
    ("busy_threshold", "expected_message", "forbidden_message"),
    [
        (
            -1,
            "Busy threshold: unconditional (utilization backoff disabled)",
            "Busy threshold: -1%",
        ),
        (25, "Busy threshold: 25%", "unconditional"),
    ],
)
def test_run_blocking_logs_busy_threshold_semantically(
    monkeypatch, caplog, busy_threshold, expected_message, forbidden_message
):
    class DummyGlobalController:
        def __init__(
            self, *, npu_ids, interval, vram_to_keep, busy_threshold, workload
        ):
            self.npu_ids = npu_ids
            self.interval = interval
            self.vram_to_keep = vram_to_keep
            self.busy_threshold = busy_threshold
            self.workload = workload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    import keep_npu.global_npu_controller.global_npu_controller as global_module

    monkeypatch.setattr(global_module, "GlobalNPUController", DummyGlobalController)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)
    monkeypatch.setattr(cli.logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=cli.logger.name)

    cli._run_blocking(
        interval=1,
        npu_ids="0",
        vram="1MiB",
        legacy_threshold=None,
        busy_threshold=busy_threshold,
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == cli.logger.name
    ]
    assert any(expected_message in message for message in messages)
    assert not any(forbidden_message in message for message in messages)
