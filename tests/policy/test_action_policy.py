"""Protocol-conformance tests for ActionPolicy.

These are the type-checker's guarantee made runtime-checkable: both concrete
action policies must satisfy the same callable shape.
"""
from __future__ import annotations

from scripts.policy.action_policy import ActionPolicy
from scripts.policy.policy_openrouter import choose_action_openrouter
from scripts.policy.policy_stub import choose_action


def test_choose_action_satisfies_action_policy() -> None:
    assert isinstance(choose_action, ActionPolicy)


def test_choose_action_openrouter_satisfies_action_policy() -> None:
    assert isinstance(choose_action_openrouter, ActionPolicy)


def test_non_callable_does_not_satisfy_action_policy() -> None:
    # runtime_checkable Protocol isinstance checks are structural on member
    # names only (not signatures), so the meaningful negative case is
    # "has no __call__ at all", not "wrong argument shape".
    assert isinstance({"branch": "x", "tool": "y", "reason": "z"}, ActionPolicy) is False
