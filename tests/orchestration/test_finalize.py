"""Tests for trajectory finalization: reward computation and critique veto.

test_finalize_state_real_reward_pipeline_abstains_with_no_candidate is a real
integration test driving the actual compute_trajectory_reward /
final_decision_from_reward / finalize_trajectory pipeline end to end, no
mocking. The critique-veto tests monkeypatch those two collaborators to force
the "emitted" branch deterministically, isolating the one piece of decision
logic that lives in _finalize_state itself (not in the reward module) without
needing to construct a full passing rollout.
"""
from __future__ import annotations

import scripts.orchestration.finalize as finalize_module
from scripts.orchestration.finalize import _finalize_state


def test_finalize_state_real_reward_pipeline_abstains_with_no_candidate() -> None:
    state = {"action_history": [], "final": {"status": "running"}, "critique": None}
    reward = _finalize_state(state)

    assert reward["reward"] == 0.0
    assert state["final"]["status"] == "abstained"
    assert state["final"]["termination_reason"] == "abstained_no_candidate"
    assert state["trajectory_reward"] is reward
    assert state["trajectory"]["final_reward"] is reward


def _patch_reward_pipeline(monkeypatch) -> None:
    canned_reward = {"reward": 0.9, "components": {}, "metrics": {}}
    canned_decision = {
        "status": "emitted",
        "termination_reason": "emitted",
        "finding": {"candidate_id": "c1"},
    }
    monkeypatch.setattr(finalize_module, "compute_trajectory_reward", lambda state: dict(canned_reward))
    monkeypatch.setattr(
        finalize_module,
        "final_decision_from_reward",
        lambda state, reward: dict(canned_decision),
    )


def test_finalize_state_downgrades_to_abstained_when_critique_rejects(monkeypatch) -> None:
    _patch_reward_pipeline(monkeypatch)
    state = {"final": {"status": "running"}, "critique": {"approved_for_emit": False}}

    _finalize_state(state)

    assert state["final"]["status"] == "abstained"
    assert state["final"]["termination_reason"] == "critique_rejected"
    assert state["final"]["finding"] is None


def test_finalize_state_downgrades_to_abstained_when_critique_missing(monkeypatch) -> None:
    _patch_reward_pipeline(monkeypatch)
    state = {"final": {"status": "running"}, "critique": None}

    _finalize_state(state)

    assert state["final"]["status"] == "abstained"
    assert state["final"]["termination_reason"] == "critique_rejected"


def test_finalize_state_keeps_emitted_when_critique_approves(monkeypatch) -> None:
    _patch_reward_pipeline(monkeypatch)
    state = {"final": {"status": "running"}, "critique": {"approved_for_emit": True}}

    reward = _finalize_state(state)

    assert state["final"]["status"] == "emitted"
    assert state["final"]["finding"]["candidate_id"] == "c1"
    assert reward["reward"] == 0.9


def test_finalize_state_stamps_reward_onto_already_emitted_finding() -> None:
    state = {
        "final": {"status": "emitted", "finding": {"candidate_id": "c1"}},
        "action_history": [],
        "critique": {"approved_for_emit": True},
    }
    reward = _finalize_state(state)

    assert state["final"]["finding"]["reward"] == reward["reward"]
