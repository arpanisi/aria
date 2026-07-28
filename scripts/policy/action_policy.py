#!/usr/bin/env python3
"""Structural interface shared by the deterministic and OpenRouter action policies.

Both choose_action (policy_stub.py) and choose_action_openrouter
(policy_openrouter.py) decide the next tool/branch for a trajectory given its
state. This Protocol documents that shared contract and lets a type checker
verify both functions satisfy it, without requiring either to change.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ActionPolicy(Protocol):
    """Callable taking trajectory state and returning the next action.

    The returned dict must contain at least ``branch``, ``tool``, and
    ``reason``; implementations may add extra fields (``policy``,
    ``policy_model``, ``token_usage``, ``telemetry``, and so on).
    """

    def __call__(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        ...
