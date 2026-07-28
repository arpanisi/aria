"""Integration tests for state_io's real file I/O (no mocking)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.orchestration.state_io import (
    _append_run_log,
    _collect_model_calls,
    _json_default,
    _load_env_file,
    _load_table,
    _write_json,
)


def test_load_table_reads_real_csv(tmp_path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    df = _load_table(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_table_reads_real_tsv(tmp_path) -> None:
    path = tmp_path / "d.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    df = _load_table(path)
    assert list(df.columns) == ["a", "b"]


def test_load_table_rejects_unsupported_suffix(tmp_path) -> None:
    path = tmp_path / "d.parquet"
    path.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load_table(path)


def test_write_json_round_trips_and_handles_numpy_scalars(tmp_path) -> None:
    path = tmp_path / "nested" / "state.json"
    payload = {"reward": np.float64(0.75), "count": np.int64(3)}
    _write_json(path, payload)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["reward"] == 0.75
    assert reloaded["count"] == 3


def test_json_default_falls_back_to_str_for_non_item_objects() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    assert _json_default(Opaque()) == "opaque-value"
    assert _json_default(np.float32(1.5)) == pytest.approx(1.5)


def test_append_run_log_writes_one_jsonl_record_per_call(tmp_path) -> None:
    path = tmp_path / "run_log.jsonl"
    state = {
        "trajectory": {"trajectory_id": "t1"},
        "final": {"status": "abstained"},
        "trajectory_reward": {"reward": 0.1, "metrics": {}},
    }
    _append_run_log(path, state)
    _append_run_log(path, {**state, "trajectory": {"trajectory_id": "t2"}})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trajectory_id"] == "t1"
    assert json.loads(lines[1])["trajectory_id"] == "t2"


def test_collect_model_calls_deduplicates_by_model_call_id() -> None:
    state = {
        "a": {"telemetry": {"model_call_id": "c1", "tool_name": "x"}},
        "nested": {"b": {"telemetry": {"model_call_id": "c1", "tool_name": "x"}}},
        "list_field": [{"telemetry": {"model_call_id": "c2", "tool_name": "y"}}],
    }
    calls = _collect_model_calls(state)
    assert len(calls) == 2
    assert {c["model_call_id"] for c in calls} == {"c1", "c2"}


def test_load_env_file_sets_only_missing_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ARIA_TEST_NEW_VAR", raising=False)
    monkeypatch.setenv("ARIA_TEST_EXISTING_VAR", "already_set")
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nARIA_TEST_NEW_VAR=new_value\nARIA_TEST_EXISTING_VAR=should_not_override\n",
        encoding="utf-8",
    )
    _load_env_file(path)
    import os

    assert os.environ["ARIA_TEST_NEW_VAR"] == "new_value"
    assert os.environ["ARIA_TEST_EXISTING_VAR"] == "already_set"


def test_load_env_file_missing_path_is_a_noop(tmp_path) -> None:
    _load_env_file(tmp_path / "does-not-exist.env")
