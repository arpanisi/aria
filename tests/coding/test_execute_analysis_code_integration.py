"""End-to-end integration test for execute_analysis_code.

No mocking, no OpenRouter API needed: execute_analysis_code only needs code
text already present in state, it doesn't call the generation path itself.
This exercises the real Sandbox class, real subprocess execution, real JSON
parsing, and the real execution-contract audit all together, exactly the
pieces that moved during the code_agent.py split.
"""
from __future__ import annotations

import pandas as pd

from scripts.coding.code_agent import execute_analysis_code

VALID_SCRIPT = '''
import json
import sys

input_csv, candidate_json, method_json = sys.argv[1], sys.argv[2], sys.argv[3]

with open(candidate_json) as f:
    candidate = json.load(f)
with open(method_json) as f:
    method = json.load(f)

result = {
    "status": "ok",
    "action": "operate_on_data",
    "task_type": "regression",
    "method": "toy_method",
    "candidate_id": candidate.get("candidate_id"),
    "outcome": candidate.get("outcome"),
    "predictors": candidate.get("predictors", []),
    "n_observations": 10,
    "fit_metrics": {"coefficient": 1.23},
    "diagnostics": {},
    "robustness": {"cv_r2_mean": 0.75, "bootstrap_sign_stability": 0.9},
    "warnings": [],
    "method_spec_id": "toy_spec",
    "method_spec_step_results": {"s01": {"implemented": True, "status": "ok", "output": "done"}},
    "assumptions_checked": {"a01": {"passed": True, "diagnostic": "checked", "value": 1.0}},
    "output_contract_satisfied": {"toy_result": True},
}
print(json.dumps(result))
'''

FAILING_SCRIPT = "import json\nraise RuntimeError('boom')\n"


def _base_state(script: str) -> dict:
    return {
        "trajectory": {"trajectory_id": "test-traj-001"},
        "candidate_relationship": {"candidate_id": "c001", "outcome": "y", "predictors": ["x1"]},
        "analysis_method": {"selected_method": "toy_method"},
        "method_spec": {
            "method_spec_id": "toy_spec",
            "algorithm_steps": [{"id": "s01", "description": "do the one step"}],
            "assumptions": [{"id": "a01", "name": "toy_assumption"}],
            "output_contract": ["toy_result"],
        },
        "analysis_code": {"code": script, "policy": "openrouter"},
    }


def test_execute_analysis_code_real_subprocess_success(tmp_path) -> None:
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [4, 5, 6]})
    state = _base_state(VALID_SCRIPT)

    evidence = execute_analysis_code(df=df, state=state, work_dir=tmp_path, deny_network=False)

    assert evidence["status"] == "ok"
    assert evidence["method"] == "toy_method"
    assert evidence["execution_contract_audit"]["status"] == "ok"
    assert evidence["execution_contract_audit"]["issues"] == []
    assert evidence["generated_code"]["sandbox"]["isolation_level"] == "subprocess_resource_limited"


def test_execute_analysis_code_writes_readonly_inputs(tmp_path) -> None:
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [4, 5, 6]})
    state = _base_state(VALID_SCRIPT)

    execute_analysis_code(df=df, state=state, work_dir=tmp_path, deny_network=False)

    run_dir = tmp_path / "test-traj-001"
    input_csv = run_dir / "input.csv"
    assert input_csv.exists()
    # 0o444 read-only permission bits were applied.
    assert not (input_csv.stat().st_mode & 0o222)


def test_execute_analysis_code_failure_without_api_key_stays_invalid(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [4, 5, 6]})
    state = _base_state(FAILING_SCRIPT)

    evidence = execute_analysis_code(df=df, state=state, work_dir=tmp_path, deny_network=False)

    # Repair is attempted (policy == "openrouter") but has no API key, so it
    # cannot recover; the real failure must still surface as invalid, not
    # silently swallowed.
    assert evidence["status"] == "invalid"
