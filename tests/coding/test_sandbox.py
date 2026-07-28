"""Tests for the Sandbox class and its supporting primitives.

test_sandbox_run_executes_real_subprocess and test_sandbox_enforces_cpu_time_limit
are genuine integration tests, no mocking, they actually spawn a subprocess
under the real resource limiter and check its real behavior.
"""
from __future__ import annotations

import sys

from scripts.coding.sandbox import (
    Sandbox,
    _execution_contract_audit,
    _has_internal_validation_metric,
    _has_stability_metric,
)


def test_sandbox_wrap_command_disabled_network_passthrough() -> None:
    sandbox = Sandbox(deny_network=False)
    command, mode = sandbox.wrap_command(["python3", "-c", "pass"])
    assert command == ["python3", "-c", "pass"]
    assert mode["enforced"] is False
    assert mode["requested"] is False


def test_sandbox_metadata_reflects_configured_limits() -> None:
    sandbox = Sandbox(timeout_seconds=45, memory_limit_mb=512, cpu_time_seconds=20)
    meta = sandbox.metadata(run_dir=__import__("pathlib").Path("/tmp/x"), network_mode={"enforced": True})
    assert meta["timeout_seconds"] == 45
    assert meta["memory_limit_mb"] == 512
    assert meta["cpu_time_seconds"] == 20
    assert meta["container_isolation"] is False


def test_sandbox_run_executes_real_subprocess(tmp_path) -> None:
    sandbox = Sandbox(timeout_seconds=10, memory_limit_mb=256, cpu_time_seconds=5, deny_network=False)
    command = [sys.executable, "-c", "print('hello from sandbox')"]
    result = sandbox.run(command, cwd=tmp_path)
    assert result.returncode == 0
    assert "hello from sandbox" in result.stdout


def test_sandbox_enforces_cpu_time_limit(tmp_path) -> None:
    """A real busy-loop should be killed once it exceeds cpu_time_seconds."""
    sandbox = Sandbox(timeout_seconds=10, memory_limit_mb=256, cpu_time_seconds=1, deny_network=False)
    busy_loop = "i = 0\nwhile True:\n    i += 1\n"
    command = [sys.executable, "-c", busy_loop]
    result = sandbox.run(command, cwd=tmp_path)
    # Killed by RLIMIT_CPU (SIGXCPU), so it must not exit as if it succeeded.
    assert result.returncode != 0


def test_execution_contract_audit_flags_missing_step_results() -> None:
    method_spec = {"algorithm_steps": [{"id": "s01"}, {"id": "s02"}]}
    evidence = {"method_spec_step_results": {"s01": {}}, "robustness": {"cv_r2_mean": 0.8, "bootstrap_sign_stability": 1.0}}
    audit = _execution_contract_audit(evidence, method_spec=method_spec)
    assert any("s02" in issue for issue in audit["issues"])


def test_execution_contract_audit_passes_when_complete() -> None:
    method_spec = {
        "algorithm_steps": [{"id": "s01"}],
        "assumptions": [{"id": "a01"}],
        "output_contract": ["result"],
    }
    evidence = {
        "method_spec_step_results": {"s01": {"implemented": True}},
        "assumptions_checked": {"a01": {"passed": True}},
        "output_contract_satisfied": {"result": True},
        "robustness": {"cv_r2_mean": 0.9, "bootstrap_sign_stability": 1.0},
    }
    audit = _execution_contract_audit(evidence, method_spec=method_spec)
    assert audit["status"] == "ok"
    assert audit["issues"] == []


def test_has_internal_validation_metric_detects_nested_forms() -> None:
    assert _has_internal_validation_metric({"cv_r2_mean": 0.5}) is True
    assert _has_internal_validation_metric({"custom": {"r2_mean": 0.5}}) is True
    assert _has_internal_validation_metric({"unrelated": "value"}) is False


def test_has_stability_metric_detects_nested_forms() -> None:
    assert _has_stability_metric({"bootstrap_sign_stability": 1.0}) is True
    assert _has_stability_metric({"custom": {"variance": 0.1}}) is True
    assert _has_stability_metric({"unrelated": "value"}) is False
