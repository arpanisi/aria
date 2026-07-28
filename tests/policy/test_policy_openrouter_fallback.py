"""Integration test for choose_action_openrouter's no-API-key fallback path.

No mocking: this exercises the real early-return branch that calls the real
deterministic policy when OPENROUTER_API_KEY is absent.
"""
from __future__ import annotations

from scripts.policy.policy_openrouter import choose_action_openrouter
from scripts.policy.policy_stub import choose_action

STATE = {"cleaning_report": None}


def test_choose_action_openrouter_falls_back_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    action = choose_action_openrouter(STATE, api_key=None)

    assert action["policy"] == "deterministic_fallback"
    assert action["policy_warning"] == "OPENROUTER_API_KEY missing"
    assert action["tool"] == choose_action(STATE)["tool"]
