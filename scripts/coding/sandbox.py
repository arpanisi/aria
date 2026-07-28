#!/usr/bin/env python3
"""Sandboxed subprocess execution of generated analysis code.

Sandbox is the one genuine class in this codebase's split: it holds real
configuration (memory limit, CPU time limit, wall-clock timeout, network
policy) used across two call sites (the initial run and the post-repair
rerun in code_agent.py), with .run() as its one real behavior. Everything
else in this module is the free-function mechanics Sandbox composes.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class Sandbox:
    """Resource-limited, network-restricted subprocess execution.

    This is stronger than raw subprocess execution but not a full container
    sandbox: it relies on OS resource limits (RLIMIT_AS/RLIMIT_CPU/RLIMIT_NOFILE)
    plus a best-effort network-isolation wrapper (sandbox-exec on macOS,
    unshare --net on Linux), not a container or VM boundary.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = 60,
        memory_limit_mb: int = 1024,
        cpu_time_seconds: int = 30,
        deny_network: bool = True,
        require_network_isolation: bool = False,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.cpu_time_seconds = cpu_time_seconds
        self.deny_network = deny_network
        self.require_network_isolation = require_network_isolation

    def wrap_command(self, base_command: list[str]) -> tuple[list[str], dict[str, Any]]:
        return _network_sandbox_command(
            base_command,
            deny_network=self.deny_network,
            require_network_isolation=self.require_network_isolation,
        )

    def metadata(self, *, run_dir: Path, network_mode: dict[str, Any]) -> dict[str, Any]:
        return sandbox_metadata(
            run_dir=run_dir,
            timeout_seconds=self.timeout_seconds,
            memory_limit_mb=self.memory_limit_mb,
            cpu_time_seconds=self.cpu_time_seconds,
            network_mode=network_mode,
        )

    def run(self, command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        """Execute an already network-wrapped command under the resource limits."""
        return subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            env=_sandbox_env(),
            preexec_fn=_resource_limiter(
                memory_limit_mb=self.memory_limit_mb,
                cpu_time_seconds=self.cpu_time_seconds,
            ),
        )


def parse_json_stdout(stdout: str) -> dict[str, Any]:
    """Parse either full pretty-printed JSON or a final-line JSON object."""
    text = stdout.strip()
    if not text:
        raise ValueError("empty stdout")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(text.splitlines()[-1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON output was not an object")
    return parsed


def sandbox_metadata(
    *,
    run_dir: Path,
    timeout_seconds: int,
    memory_limit_mb: int,
    cpu_time_seconds: int,
    network_mode: dict[str, Any],
) -> dict[str, Any]:
    return {
        "isolation_level": "subprocess_resource_limited",
        "run_dir": str(run_dir),
        "network": network_mode,
        "read_only_inputs": True,
        "timeout_seconds": timeout_seconds,
        "cpu_time_seconds": cpu_time_seconds,
        "memory_limit_mb": memory_limit_mb,
        "stdout_stderr_captured": True,
        "artifact_allowlist": ["analysis_script.py", "input.csv", "candidate.json", "analysis_method.json"],
        "container_isolation": False,
        "warning": "This is stronger than raw subprocess execution but not a full container sandbox.",
    }


def _network_sandbox_command(
    command: list[str],
    *,
    deny_network: bool,
    require_network_isolation: bool,
) -> tuple[list[str], dict[str, Any]]:
    if not deny_network:
        return command, {
            "requested": False,
            "enforced": False,
            "mechanism": None,
            "warning": "Network isolation disabled by configuration.",
        }
    if sys.platform == "darwin":
        wrapped, mode = _macos_sandbox_exec_command(command)
    elif sys.platform.startswith("linux"):
        wrapped, mode = _linux_netns_command(command)
    else:
        wrapped, mode = command, {
            "requested": True,
            "enforced": False,
            "mechanism": None,
            "warning": f"no network isolation mechanism for platform {sys.platform!r}; relies only on code validation.",
        }
    if not mode["enforced"] and require_network_isolation:
        mode["blocked_execution"] = True
    return wrapped, mode


def _macos_sandbox_exec_command(command: list[str]) -> tuple[list[str], dict[str, Any]]:
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return command, {
            "requested": True,
            "enforced": False,
            "mechanism": None,
            "warning": "sandbox-exec not found; network isolation relies only on code validation.",
        }
    profile = "(version 1)(allow default)(deny network*)"
    preflight = _subprocess_preflight([sandbox_exec, "-p", profile, sys.executable, "-c", "print('sandbox-ok')"])
    if not preflight["ok"]:
        return command, {
            "requested": True,
            "enforced": False,
            "mechanism": "macos_sandbox_exec_deny_network",
            "profile": profile,
            "preflight": preflight,
            "warning": "sandbox-exec exists but failed preflight; network isolation relies only on code validation.",
        }
    return [sandbox_exec, "-p", profile, *command], {
        "requested": True,
        "enforced": True,
        "mechanism": "macos_sandbox_exec_deny_network",
        "profile": profile,
        "preflight": preflight,
    }


def _linux_netns_command(command: list[str]) -> tuple[list[str], dict[str, Any]]:
    unshare = shutil.which("unshare")
    if not unshare:
        return command, {
            "requested": True,
            "enforced": False,
            "mechanism": None,
            "warning": "unshare not found; network isolation relies only on code validation.",
        }
    netns_prefix = [unshare, "--user", "--net", "--map-root-user", "--"]
    preflight = _subprocess_preflight([*netns_prefix, sys.executable, "-c", "print('sandbox-ok')"])
    if not preflight["ok"]:
        return command, {
            "requested": True,
            "enforced": False,
            "mechanism": "linux_netns_unshare",
            "preflight": preflight,
            "warning": (
                "unshare --net exists but failed preflight, likely because unprivileged user "
                "namespaces are disabled on this host/container; network isolation relies only "
                "on code validation."
            ),
        }
    return [*netns_prefix, *command], {
        "requested": True,
        "enforced": True,
        "mechanism": "linux_netns_unshare",
        "preflight": preflight,
    }


def _subprocess_preflight(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=_sandbox_env(),
        )
        return {
            "ok": completed.returncode == 0 and "sandbox-ok" in completed.stdout,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-500:],
            "stderr": completed.stderr[-500:],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def _sandbox_env() -> dict[str, str]:
    keep = {}
    for key in ("PATH", "PYTHONPATH", "SYSTEMROOT"):
        value = os.environ.get(key)
        if value:
            keep[key] = value
    keep["PYTHONNOUSERSITE"] = "1"
    # Unset, BLAS libraries default to one thread per visible CPU core and
    # allocate per-thread scratch buffers; on many-core hosts this can exceed
    # the sandbox's memory_limit_mb cap on its own, before the generated
    # script does any real work. Small tabular analyses have no need for
    # multi-threaded BLAS.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        keep[key] = "1"
    return keep


def _resource_limiter(*, memory_limit_mb: int, cpu_time_seconds: int):
    def limit() -> None:
        memory_bytes = int(memory_limit_mb) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_time_seconds), int(cpu_time_seconds) + 1))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except Exception:
            pass

    return limit


def _execution_contract_audit(evidence: dict[str, Any], *, method_spec: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    step_ids = [
        str(step.get("id") or f"s{i:02d}")
        for i, step in enumerate(method_spec.get("algorithm_steps") or [], start=1)
        if isinstance(step, dict)
    ]
    step_results = evidence.get("method_spec_step_results")
    if step_ids:
        if not isinstance(step_results, dict):
            issues.append("method_spec_step_results_missing_or_not_object")
        else:
            missing = [step_id for step_id in step_ids if step_id not in step_results]
            if missing:
                issues.append(f"method_spec_step_results_missing_ids:{','.join(missing)}")
    assumption_keys = []
    for i, item in enumerate(method_spec.get("assumptions") or [], start=1):
        if isinstance(item, dict):
            assumption_keys.append(str(item.get("id") or item.get("name") or f"a{i:02d}"))
        else:
            assumption_keys.append(str(item))
    assumptions_checked = evidence.get("assumptions_checked")
    if assumption_keys:
        if not isinstance(assumptions_checked, dict):
            issues.append("assumptions_checked_missing_or_not_object")
        else:
            missing = [key for key in assumption_keys if key not in assumptions_checked]
            if missing:
                issues.append(f"assumptions_checked_missing_keys:{','.join(missing)}")
    output_contract = [str(item) for item in method_spec.get("output_contract") or []]
    output_satisfied = evidence.get("output_contract_satisfied")
    if output_contract:
        if not isinstance(output_satisfied, dict):
            issues.append("output_contract_satisfied_missing_or_not_object")
        else:
            missing = [key for key in output_contract if key not in output_satisfied]
            if missing:
                issues.append(f"output_contract_satisfied_missing_keys:{','.join(missing)}")
    robustness = evidence.get("robustness")
    if not isinstance(robustness, dict):
        issues.append("robustness_missing_or_not_object")
        robustness = {}
    if not _has_internal_validation_metric(robustness):
        issues.append("robustness_missing_internal_validation_metric")
    if not _has_stability_metric(robustness):
        issues.append("robustness_missing_stability_metric")
    return {
        "status": "ok" if not issues else "issues",
        "issues": issues,
        "required_algorithm_step_ids": step_ids,
        "required_assumption_keys": assumption_keys,
        "required_output_contract": output_contract,
    }


def _has_internal_validation_metric(robustness: dict[str, Any]) -> bool:
    keys = {
        "cv_r2_mean",
        "cv_score_mean",
        "cross_validation",
        "bootstrap_cv_mean",
        "internal_validation",
        "holdout_consistency",
        "reconstruction_error",
        "output_invariance",
    }
    if any(key in robustness for key in keys):
        return True
    for value in robustness.values():
        if isinstance(value, dict) and any(key in value for key in {"r2_mean", "score_mean", "mse_mean", "error", "agreement"}):
            return True
    return False


def _has_stability_metric(robustness: dict[str, Any]) -> bool:
    keys = {
        "bootstrap_sign_stability",
        "stability",
        "weight_matrix_std",
        "sensitivity",
        "perturbation_sensitivity",
        "bootstrap_agreement",
        "output_rank_correlation",
    }
    if any(key in robustness for key in keys):
        return True
    for value in robustness.values():
        if isinstance(value, dict) and any(key in value for key in {"std", "coefficient_stds", "variance", "agreement", "rank_correlation"}):
            return True
    return False
