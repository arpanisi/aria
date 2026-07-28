#!/usr/bin/env python3
"""Run the first deterministic closed-loop discovery prototype."""

from __future__ import annotations

import argparse
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data.data_profile import profile_dataset  # noqa: E402
from scripts.coding.code_agent import (  # noqa: E402
    DEFAULT_CODE_MODEL,
    DEFAULT_REPAIR_MODEL,
    execute_analysis_code,
    generate_analysis_code,
)
from scripts.data.data_tools import (  # noqa: E402
    clean_data,
    discover_candidate_relationships,
    select_candidate,
    select_analysis_method,
)
from scripts.validation.critique_tools import critique_finding  # noqa: E402
from scripts.core.discovery_state import (  # noqa: E402
    append_action,
    append_transition,
    decrement_budget,
    make_initial_state,
    state_snapshot,
)
from scripts.retrieval.literature_tools import retrieve_local_literature  # noqa: E402
from scripts.policy.action_policy import ActionPolicy  # noqa: E402
from scripts.policy.policy_openrouter import choose_action_openrouter  # noqa: E402
from scripts.policy.policy_stub import choose_action  # noqa: E402
from scripts.policy.query_policy import DEFAULT_QUERY_POLICY_MODEL, generate_query_action  # noqa: E402
from scripts.extraction.method_guidance_tools import assess_method_guidance  # noqa: E402
from scripts.extraction.method_spec_tools import DEFAULT_SUMMARIZER_MODEL  # noqa: E402
from scripts.validation.paper_program_eval import evaluate_paper_program  # noqa: E402
from scripts.extraction.hypothesis_tools import build_structured_hypothesis  # noqa: E402
from scripts.validation.statistical_validation import evaluate_statistical_validation  # noqa: E402
from scripts.orchestration.finalize import _finalize_state, _print_summary  # noqa: E402
from scripts.orchestration.literature_summarization import (  # noqa: E402
    _seen_literature_paper_ids,
    _summarize_retrieved_method_specs,
)
from scripts.orchestration.method_spec_selection import _analysis_method_from_spec  # noqa: E402
from scripts.orchestration.state_io import (  # noqa: E402
    _append_run_log,
    _load_env_file,
    _load_table,
    _write_json,
)


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    df = _load_table(args.data)
    profile = profile_dataset(df)
    state = make_initial_state(
        dataset_path=str(args.data),
        dataset_profile=profile,
        budgets={
            "data_actions": args.data_budget,
            "literature_actions": args.literature_budget,
            "method_guidance_checks": args.method_guidance_budget,
            "paper_summarizer_calls": args.paper_summarizer_budget,
        },
    )
    working_df = df

    analysis_attempts_completed = 0
    tool_actions_completed = 0
    max_tool_actions = max(args.steps * args.max_tool_actions_per_step, args.steps)
    state["analysis_attempt_budget"] = args.steps
    state["analysis_attempts_completed"] = analysis_attempts_completed
    state["max_tool_actions"] = max_tool_actions

    while analysis_attempts_completed < args.steps and tool_actions_completed < max_tool_actions:
        if state["final"]["status"] != "running":
            break
        policy_started_at = perf_counter()
        policy_action_marker = {
            "tool": "choose_action",
            "branch": "policy",
            "policy": args.policy,
            "policy_model": args.openrouter_model if args.policy == "openrouter" else None,
        }
        _record_action_progress(
            state,
            status="started",
            action=policy_action_marker,
            tool_actions_completed=tool_actions_completed,
            analysis_attempts_completed=analysis_attempts_completed,
        )
        _write_json(args.out, state)
        try:
            action = _choose_policy_action(
                state,
                policy=args.policy,
                openrouter_model=args.openrouter_model,
                openrouter_reasoning=args.openrouter_reasoning,
                policy_hint_mode=args.policy_hint_mode,
            )
        except Exception as exc:  # noqa: BLE001
            _record_runtime_error(
                state,
                phase="choose_action",
                tool="choose_action",
                error=exc,
                elapsed_seconds=perf_counter() - policy_started_at,
            )
            _write_json(args.out, state)
            break
        _record_action_progress(
            state,
            status="completed",
            action=policy_action_marker,
            tool_actions_completed=tool_actions_completed,
            analysis_attempts_completed=analysis_attempts_completed,
            elapsed_seconds=perf_counter() - policy_started_at,
        )
        state_before = state_snapshot(state)
        action_started_at = perf_counter()
        _record_action_progress(
            state,
            status="started",
            action=action,
            tool_actions_completed=tool_actions_completed,
            analysis_attempts_completed=analysis_attempts_completed,
        )
        _write_json(args.out, state)
        try:
            working_df, observation = _run_action(
                working_df,
                state,
                action,
                max_candidates=args.max_candidates,
                arxiv_snapshot=args.arxiv_snapshot,
                arxiv_index=args.arxiv_index,
                arxiv_category_prefixes=args.arxiv_category_prefix,
                q_value_threshold=args.q_value_threshold,
                literature_limit=args.literature_limit,
                literature_scan_limit=args.literature_scan_limit,
                literature_index_strategy=args.literature_index_strategy,
                literature_top_k=args.literature_top_k,
                fetch_pdfs=args.fetch_pdfs,
                pdf_cache_dir=args.pdf_cache_dir,
                method_guidance_classifier=args.method_guidance_classifier,
                paper_summarizer=args.paper_summarizer,
                paper_summarizer_model=args.paper_summarizer_model,
                paper_summarizer_limit=args.paper_summarizer_limit,
                method_spec_cache_dir=args.method_spec_cache_dir,
                query_policy=args.query_policy,
                query_policy_model=args.query_policy_model,
                query_policy_temperature=args.query_policy_temperature,
                query_policy_max_tokens=args.query_max_tokens,
                query_policy_base_url=args.query_policy_base_url,
                query_policy_api_key_env=args.query_policy_api_key_env,
                query_rollout_index=args.query_rollout_index,
                openrouter_model=args.openrouter_model,
                openrouter_reasoning=args.openrouter_reasoning,
                critic=args.critic,
                code_policy=args.code_policy,
                code_writer_model=args.code_writer_model,
                code_repair_model=args.code_repair_model,
                code_repair_wall_timeout=args.code_repair_wall_timeout,
                generated_code_dir=args.generated_code_dir,
                generated_code_timeout=args.generated_code_timeout,
                generated_code_memory_mb=args.generated_code_memory_mb,
                generated_code_cpu_seconds=args.generated_code_cpu_seconds,
                deny_generated_code_network=not args.allow_generated_code_network,
                require_generated_code_network_isolation=args.require_generated_code_network_isolation,
            )
        except Exception as exc:  # noqa: BLE001
            _record_runtime_error(
                state,
                phase="run_action",
                tool=str(action.get("tool") or ""),
                error=exc,
                elapsed_seconds=perf_counter() - action_started_at,
            )
            _write_json(args.out, state)
            break
        _record_action_progress(
            state,
            status="completed",
            action=action,
            tool_actions_completed=tool_actions_completed + 1,
            analysis_attempts_completed=analysis_attempts_completed,
            elapsed_seconds=perf_counter() - action_started_at,
        )
        append_action(state, action=action, observation=observation)
        if action["tool"] in {"discover_candidates", "execute_analysis_code"}:
            decrement_budget(state, "data_actions")
        elif action["branch"] == "search_literature":
            if action["tool"] == "assess_method_guidance":
                decrement_budget(state, "method_guidance_checks")
            elif action["tool"] == "summarize_method_specs":
                decrement_budget(state, "paper_summarizer_calls")
            else:
                decrement_budget(state, "literature_actions")
        append_transition(
            state,
            state_before=state_before,
            action=action,
            observation=observation,
            state_after=state_snapshot(state),
        )
        tool_actions_completed += 1
        if _completed_analysis_attempt(action, observation, remaining_budget=state.get("remaining_budget")):
            analysis_attempts_completed += 1
            state["analysis_attempts_completed"] = analysis_attempts_completed
        _write_json(args.out, state)

    if state["final"]["status"] == "running":
        state["stop_reason"] = (
            "analysis_attempt_budget_exhausted"
            if analysis_attempts_completed >= args.steps
            else "tool_action_limit_exhausted"
        )
        state["analysis_attempt_budget"] = args.steps
        state["analysis_attempts_completed"] = analysis_attempts_completed
        state["tool_actions_completed"] = tool_actions_completed
        state["max_tool_actions"] = max_tool_actions
    _finalize_state(state)
    _write_json(args.out, state)
    _append_run_log(args.run_log, state)
    _print_summary(state, args.out)


def _record_action_progress(
    state: dict[str, Any],
    *,
    status: str,
    action: dict[str, Any],
    tool_actions_completed: int,
    analysis_attempts_completed: int,
    elapsed_seconds: float | None = None,
) -> None:
    record = {
        "status": status,
        "tool": action.get("tool"),
        "branch": action.get("branch"),
        "policy": action.get("policy"),
        "policy_model": action.get("policy_model"),
        "tool_actions_completed": tool_actions_completed,
        "analysis_attempts_completed": analysis_attempts_completed,
    }
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(float(elapsed_seconds), 6)
    state["current_action"] = record if status == "started" else None
    state.setdefault("runtime_events", []).append(record)


def _record_runtime_error(
    state: dict[str, Any],
    *,
    phase: str,
    tool: str,
    error: Exception,
    elapsed_seconds: float,
) -> None:
    event = {
        "status": "failed",
        "phase": phase,
        "tool": tool,
        "error_type": type(error).__name__,
        "error": str(error),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
    }
    state["current_action"] = None
    state["stop_reason"] = "runtime_error"
    state["runtime_error"] = event
    state.setdefault("runtime_events", []).append(event)


def _run_action(
    df: pd.DataFrame,
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    max_candidates: int,
    arxiv_snapshot: Path,
    arxiv_index: Path,
    arxiv_category_prefixes: list[str],
    q_value_threshold: float,
    literature_limit: int,
    literature_scan_limit: int | None,
    literature_index_strategy: str,
    literature_top_k: int,
    fetch_pdfs: bool,
    pdf_cache_dir: Path,
    method_guidance_classifier: str,
    paper_summarizer: str,
    paper_summarizer_model: str,
    paper_summarizer_limit: int,
    method_spec_cache_dir: Path,
    query_policy: str,
    query_policy_model: str,
    query_policy_temperature: float,
    query_policy_max_tokens: int,
    query_policy_base_url: str | None,
    query_policy_api_key_env: str,
    query_rollout_index: int,
    openrouter_model: str,
    openrouter_reasoning: str,
    critic: str,
    code_policy: str,
    code_writer_model: str,
    code_repair_model: str,
    code_repair_wall_timeout: int,
    generated_code_dir: Path,
    generated_code_timeout: int,
    generated_code_memory_mb: int,
    generated_code_cpu_seconds: int,
    deny_generated_code_network: bool,
    require_generated_code_network_isolation: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tool = action["tool"]
    if tool == "clean_data":
        cleaned_df, report = clean_data(df, state["dataset_profile"])
        state["cleaning_report"] = report
        return cleaned_df, {"status": "ok", **report}

    if tool == "discover_candidates":
        candidates, report = discover_candidate_relationships(
            df,
            state["dataset_profile"],
            analysis_method=state.get("analysis_method"),
            max_candidates=max_candidates,
            q_value_threshold=q_value_threshold,
        )
        state["candidate_pool"] = candidates
        state["candidate_screening"] = report
        return df, {"status": "ok", "n_candidates": len(candidates), **report}

    if tool == "select_candidate":
        candidate = select_candidate(state["candidate_pool"])
        state["candidate_relationship"] = candidate
        state["hypothesis"] = build_structured_hypothesis(
            candidate=candidate,
            method_spec=state.get("method_spec"),
            dataset_profile=state.get("dataset_profile") or {},
        )
        return df, {
            "status": "ok" if candidate else "empty",
            "selected_candidate_id": candidate.get("candidate_id") if candidate else None,
            "hypothesis_id": (state.get("hypothesis") or {}).get("hypothesis_id"),
        }

    if tool in {"retrieve_local", "retrieve_more"}:
        candidate = state.get("candidate_relationship") or {}
        seen_paper_ids = _seen_literature_paper_ids(state) if tool == "retrieve_more" else set()
        query_action = generate_query_action(
            state,
            policy=query_policy,
            model=query_policy_model,
            temperature=query_policy_temperature,
            rollout_index=query_rollout_index,
            reasoning_mode=openrouter_reasoning,
            base_url=query_policy_base_url,
            api_key_env=query_policy_api_key_env,
            max_tokens=query_policy_max_tokens,
        )
        state.setdefault("query_actions", []).append(query_action)
        observation = retrieve_local_literature(
            snapshot_path=arxiv_snapshot,
            index_path=arxiv_index,
            candidate=candidate,
            dataset_profile=state.get("dataset_profile") or {},
            data_evidence=state.get("data_evidence", []),
            category_prefixes=arxiv_category_prefixes,
            max_records=literature_limit,
            scan_limit=literature_scan_limit,
            index_strategy=literature_index_strategy,
            top_k=literature_top_k * 2 if tool == "retrieve_more" else literature_top_k,
            raw_search_multiplier=50 if tool == "retrieve_more" else 4,
            exclude_paper_ids=seen_paper_ids,
            retrieval_round="followup" if tool == "retrieve_more" else "initial",
            fetch_pdfs=fetch_pdfs,
            pdf_cache_dir=pdf_cache_dir,
            query_override=query_action.get("query"),
            query_policy_action=query_action,
        )
        state["literature_evidence"].append(observation)
        return df, observation

    if tool == "summarize_method_specs":
        observation = _summarize_retrieved_method_specs(
            state,
            summarizer=paper_summarizer,
            model=paper_summarizer_model,
            reasoning_mode=openrouter_reasoning,
            limit=paper_summarizer_limit,
            method_spec_cache_dir=method_spec_cache_dir,
        )
        state.setdefault("method_spec_evidence", []).append(observation)
        selected_spec = observation.get("selected_method_spec")
        if selected_spec:
            state["method_spec"] = selected_spec
            state["analysis_method"] = _analysis_method_from_spec(selected_spec)
        return df, observation

    if tool == "assess_method_guidance":
        candidate = state.get("candidate_relationship") or {}
        unverified_literature = state.get("literature_evidence", [])[
            len(state.get("method_guidance_evidence", [])) :
        ]
        observation = assess_method_guidance(
            candidate=candidate,
            dataset_profile=state.get("dataset_profile") or {},
            literature_evidence=unverified_literature,
            classifier=method_guidance_classifier,
            openrouter_model=openrouter_model,
            openrouter_reasoning=openrouter_reasoning,
        )
        state["method_guidance_evidence"].append(observation)
        return df, observation

    if tool == "select_analysis_method":
        if not state.get("method_spec"):
            return df, {
                "status": "blocked",
                "reason": "select_analysis_method requires an active paper-derived method_spec",
                "selected_method": None,
            }
        observation = select_analysis_method(
            state.get("dataset_profile") or {},
            state.get("method_guidance_evidence", []),
        )
        state["analysis_method"] = _analysis_method_from_spec(state["method_spec"])
        return df, observation

    if tool == "generate_analysis_code":
        if not state.get("method_spec"):
            return df, {
                "status": "blocked",
                "reason": "generate_analysis_code requires an active paper-derived method_spec",
            }
        observation = generate_analysis_code(
            state=state,
            policy=code_policy,
            model=code_writer_model,
            repair_model=code_repair_model,
            reasoning_mode=openrouter_reasoning,
            repair_wall_timeout_seconds=code_repair_wall_timeout,
            generation_wall_timeout_seconds=generated_code_timeout,
        )
        if observation.get("status") == "ok":
            state["analysis_code"] = observation
            state["code_generation_failed"] = False
        else:
            state["analysis_code"] = None
            state["code_generation_failed"] = True
            state["code_generation_error"] = {
                "status": observation.get("status"),
                "policy": observation.get("policy"),
                "model": observation.get("model"),
                "code": observation.get("code"),
                "validation": observation.get("validation"),
                "repair_attempts": observation.get("repair_attempts"),
                "warnings": observation.get("warnings", []),
                "telemetry": observation.get("telemetry"),
            }
        return df, observation

    if tool == "execute_analysis_code":
        evidence = execute_analysis_code(
            df=df,
            state=state,
            work_dir=generated_code_dir,
            timeout_seconds=generated_code_timeout,
            memory_limit_mb=generated_code_memory_mb,
            cpu_time_seconds=generated_code_cpu_seconds,
            deny_network=deny_generated_code_network,
            require_network_isolation=require_generated_code_network_isolation,
        )
        method_spec = state.get("method_spec")
        if method_spec:
            paper_program_evaluation = evaluate_paper_program(
                method_spec=method_spec,
                code_record=state.get("analysis_code") or {},
                execution=evidence,
            )
            evidence["paper_program_evaluation"] = paper_program_evaluation
            evidence["rubric_score"] = paper_program_evaluation.get("rubric_score")
            evidence["paper_program_fidelity"] = paper_program_evaluation.get("paper_program_fidelity")
            evidence["hard_gate_verdict"] = paper_program_evaluation.get("hard_gate_verdict")
            state.setdefault("paper_program_evaluations", []).append(paper_program_evaluation)
            statistical_validation = evaluate_statistical_validation(
                hypothesis=state.get("hypothesis"),
                method_spec=method_spec,
                execution=evidence,
                paper_program_evaluation=paper_program_evaluation,
                dataset_profile=state.get("dataset_profile") or {},
                candidate_screening=state.get("candidate_screening") or {},
            )
            evidence["statistical_validation"] = statistical_validation
            evidence["statistical_validation_gate"] = statistical_validation.get("terminal_gate")
            state.setdefault("statistical_validations", []).append(statistical_validation)
        state["data_evidence"].append(evidence)
        return df, evidence

    if tool == "critique_finding":
        observation = critique_finding(
            state=state,
            critic=critic,
            openrouter_model=openrouter_model,
            openrouter_reasoning=openrouter_reasoning,
        )
        state["critique"] = observation
        return df, observation

    if tool == "abstain_or_emit":
        reward = _finalize_state(state)
        return df, {"status": state["final"]["status"], "trajectory_reward": reward}

    return df, {"status": "unknown_action", "warnings": [f"unknown tool: {tool}"]}


def _completed_analysis_attempt(
    action: dict[str, Any],
    observation: dict[str, Any],
    *,
    remaining_budget: dict[str, Any] | None = None,
) -> bool:
    """Count one scientific attempt, not one internal tool call."""
    tool = action.get("tool")
    if tool == "execute_analysis_code":
        return True
    if tool == "generate_analysis_code" and observation.get("status") != "ok":
        return True
    if tool == "summarize_method_specs" and observation.get("status") == "empty":
        return True
    if tool == "summarize_method_specs" and observation.get("selected_method_spec") is None and observation.get("n_papers_summarized"):
        # A batch of retrieved papers yielded no feasible method_spec -- that
        # is not the same as the analysis attempt being over. If there is
        # still literature/paper-summarizer budget left, the policy should
        # get a chance to retrieve a different batch (a new query, or more
        # unsummarized results) before the whole attempt is abandoned.
        budget = remaining_budget or {}
        can_retry = (
            int(budget.get("literature_actions") or 0) > 0
            or int(budget.get("paper_summarizer_calls") or 0) > 0
        )
        return not can_retry
    if tool == "abstain_or_emit":
        return True
    return False


def _choose_policy_action(
    state: dict[str, Any],
    *,
    policy: str,
    openrouter_model: str,
    openrouter_reasoning: str,
    policy_hint_mode: str,
) -> dict[str, Any]:
    if policy == "openrouter":
        openrouter_policy: ActionPolicy = choose_action_openrouter
        return openrouter_policy(
            state,
            model=openrouter_model,
            reasoning_mode=openrouter_reasoning,
            hint_mode=policy_hint_mode,
        )
    deterministic_policy: ActionPolicy = choose_action
    action = deterministic_policy(state)
    action["policy"] = "deterministic"
    return action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("tmp/agentic-run.json"))
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("data/outputs/logs/agentic_trajectory_log.jsonl"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of complete analysis attempts, not low-level tool calls.",
    )
    parser.add_argument(
        "--max-tool-actions-per-step",
        type=int,
        default=12,
        help="Internal guardrail: maximum low-level tool calls allowed per analysis attempt.",
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--q-value-threshold", type=float, default=0.10)
    parser.add_argument("--data-budget", type=int, default=3)
    parser.add_argument("--literature-budget", type=int, default=3)
    parser.add_argument(
        "--method-guidance-budget",
        "--support-budget",
        dest="method_guidance_budget",
        type=int,
        default=3,
        help="Budget for method-guidance assessment calls. --support-budget is a deprecated alias.",
    )
    parser.add_argument("--paper-summarizer-budget", type=int, default=3)
    parser.add_argument(
        "--policy",
        choices=["deterministic", "openrouter"],
        default="deterministic",
    )
    parser.add_argument(
        "--policy-hint-mode",
        choices=["deterministic", "none"],
        default="deterministic",
        help="Whether OpenRouter policy receives the deterministic next-action hint.",
    )
    parser.add_argument(
        "--method-guidance-classifier",
        "--support-classifier",
        dest="method_guidance_classifier",
        choices=["deterministic", "openrouter"],
        default="deterministic",
        help="Classifier for method-guidance assessment. --support-classifier is a deprecated alias.",
    )
    parser.add_argument(
        "--paper-summarizer",
        choices=["deterministic", "openrouter"],
        default="deterministic",
        help="Paper reader that converts retrieved literature into structured method specs.",
    )
    parser.add_argument("--paper-summarizer-model", default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument("--paper-summarizer-limit", type=int, default=2)
    parser.add_argument(
        "--method-spec-cache-dir",
        type=Path,
        default=Path("tmp/arxiv/method-spec-cache"),
        help="Extraction depends only on the paper's own text, never on dataset or query, "
        "so specs are cached by (paper_id, evidence_depth) and reused across rollouts.",
    )
    parser.add_argument(
        "--query-policy",
        choices=["deterministic", "openrouter", "openai_compatible"],
        default="deterministic",
        help="Policy that emits the arXiv methodology-search query optimized by future GRPO.",
    )
    parser.add_argument("--query-policy-model", default=DEFAULT_QUERY_POLICY_MODEL)
    parser.add_argument("--query-policy-temperature", type=float, default=0.7)
    parser.add_argument("--query-max-tokens", type=int, default=700)
    parser.add_argument("--query-policy-base-url", default=None)
    parser.add_argument("--query-policy-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--query-rollout-index", type=int, default=0)
    parser.add_argument("--openrouter-model", default="qwen/qwen3.5-plus-20260420")
    parser.add_argument(
        "--critic",
        choices=["deterministic", "openrouter"],
        default="deterministic",
    )
    parser.add_argument(
        "--openrouter-reasoning",
        choices=["none", "minimal", "hidden", "capture"],
        default="none",
        help=(
            "Reasoning control for OpenRouter calls. 'none' is cheapest; "
            "'capture' stores returned reasoning traces for audit/training data."
        ),
    )
    parser.add_argument(
        "--code-policy",
        choices=["deterministic", "openrouter"],
        default="deterministic",
        help="Whether analysis code is generated from deterministic templates or an OpenRouter coding model.",
    )
    parser.add_argument("--code-writer-model", default=DEFAULT_CODE_MODEL)
    parser.add_argument("--code-repair-model", default=DEFAULT_REPAIR_MODEL)
    parser.add_argument("--code-repair-wall-timeout", type=int, default=150)
    parser.add_argument(
        "--generated-code-dir",
        type=Path,
        default=Path("tmp/agentic-generated-code"),
    )
    parser.add_argument("--generated-code-timeout", type=int, default=150)
    parser.add_argument("--generated-code-memory-mb", type=int, default=1024)
    parser.add_argument("--generated-code-cpu-seconds", type=int, default=30)
    parser.add_argument(
        "--allow-generated-code-network",
        action="store_true",
        help="Disable OS-level network denial for generated analysis code. Off by default.",
    )
    parser.add_argument(
        "--require-generated-code-network-isolation",
        action="store_true",
        help="Fail generated-code execution if OS-level network isolation cannot be enforced.",
    )
    parser.add_argument(
        "--arxiv-snapshot",
        type=Path,
        default=Path("data/arxiv-metadata-oai-snapshot.json"),
    )
    parser.add_argument(
        "--arxiv-index",
        type=Path,
        default=Path("tmp/arxiv/arxiv_fts.sqlite"),
    )
    parser.add_argument(
        "--arxiv-category-prefix",
        action="append",
        default=[
            "cs.LG",
            "stat.ML",
            "stat.ME",
            "stat.AP",
        ],
    )
    parser.add_argument("--literature-limit", type=int, default=100000)
    parser.add_argument("--literature-scan-limit", type=_optional_int, default=None)
    parser.add_argument(
        "--literature-index-strategy",
        choices=["recent", "first_match"],
        default="recent",
    )
    parser.add_argument("--literature-top-k", type=int, default=5)
    parser.add_argument("--fetch-pdfs", action="store_true")
    parser.add_argument(
        "--pdf-cache-dir",
        type=Path,
        default=Path("tmp/arxiv/pdf-cache"),
    )
    return parser.parse_args()


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "all", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()
