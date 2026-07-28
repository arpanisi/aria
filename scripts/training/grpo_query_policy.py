#!/usr/bin/env python3
"""Online GRPO training for the ARIA retrieval-query policy.

There is no offline export step and no static reward cache. Each round: the
in-training model is served in-process (serve_policy.py); run_one_loop.py
rolls out full trajectories against that endpoint for one dataset, with
variable length driven entirely by its own existing retrieve_more branch;
every query emitted inside a trajectory shares that trajectory's terminal
reward; advantage is leave-one-out within each dataset's group of rollouts,
a trajectory's reward minus the mean reward of the other rollouts in the
same group. All rollouts in a group share the same frozen policy and the
same starting context, so their rewards are directly comparable to each
other regardless of which terminal gate each one reached, and a rollout is
never centered against a baseline that includes its own reward. One policy
gradient step is taken per round. The policy does not choose a statistical
method; it emits the arXiv methodology query whose downstream trajectory
receives a delayed terminal reward.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    args = parse_args()
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"GRPO training requires torch. Import failed: {exc}") from exc

    from scripts.policy.serve_policy import start_server

    model, tokenizer = load_policy_model(args)
    print(f"{_ts()} attn_implementation={getattr(model.config, '_attn_implementation', 'unknown')}", flush=True)
    ref_model = load_reference_model(args) if args.kl_coef > 0 else None
    start_server(model, tokenizer, port=args.serve_port, enable_reasoning=args.enable_reasoning)
    base_url = f"http://127.0.0.1:{args.serve_port}/v1/chat/completions"

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    datasets = args.datasets

    for step in range(args.steps):
        step_datasets = _select_datasets_for_step(datasets, step, args.datasets_per_step)
        out_dir = args.output_dir / "rollouts" / f"step_{step:05d}"
        states_by_dataset = run_multi_group(
            step_datasets,
            base_url=base_url,
            group_size=args.group_size,
            out_dir=out_dir,
            query_policy_model=args.model,
            query_policy_temperature=args.query_policy_temperature,
            query_max_tokens=args.query_max_tokens,
            rollout_timeout=args.rollout_timeout,
            arxiv_snapshot=args.arxiv_snapshot,
            arxiv_index=args.arxiv_index,
            env_file=args.env_file,
            generated_code_memory_mb=args.generated_code_memory_mb,
            fetch_pdfs=args.fetch_pdfs,
            pdf_cache_dir=args.pdf_cache_dir,
            method_spec_cache_dir=args.method_spec_cache_dir,
            generated_code_dir=args.generated_code_dir,
            run_log=args.run_log,
        )
        n_format_invalid = sum(
            1
            for states in states_by_dataset.values()
            for state in states
            for query_action in state.get("query_actions", [])
            if query_action.get("status") == "format_invalid"
        )

        # Advantage is leave-one-out within each dataset's own group of
        # rollouts: reward minus the mean reward of the other rollouts in the
        # same group. All rollouts in a group share the same frozen policy
        # and the same starting context, so their rewards are directly
        # comparable regardless of which gate each one reached; the baseline
        # for a given rollout never uses that rollout's own reward, so it
        # stays unbiased without needing any cross-round state.
        samples: list[dict[str, Any]] = []
        for states in states_by_dataset.values():
            rewards = [float((state.get("trajectory_reward") or {}).get("reward") or 0.0) for state in states]
            n = len(states)
            total_reward = sum(rewards)
            for state, reward in zip(states, rewards):
                advantage = (reward - (total_reward - reward) / (n - 1)) if n > 1 else 0.0
                for sample in extract_samples(state):
                    sample["advantage"] = advantage
                    samples.append(sample)

        if not samples:
            print(f"step {step}: no usable samples, skipped (n_format_invalid={n_format_invalid})")
            continue
        loss = grpo_step(model, tokenizer, samples, ref_model, args.kl_coef, optimizer, args.enable_reasoning)
        mean_reward = sum(s["reward"] for s in samples) / len(samples)
        names = ", ".join(d.name for d in step_datasets)
        print(f"step {step} [{names}]: n={len(samples)} mean_reward={mean_reward:.4f} loss={loss:.4f} n_format_invalid={n_format_invalid}")
        if (step + 1) % args.save_steps == 0:
            model.save_pretrained(str(args.output_dir / f"checkpoint-{step + 1}"))

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


def _select_datasets_for_step(datasets: list[Path], step: int, datasets_per_step: int) -> list[Path]:
    n = len(datasets)
    start = (step * datasets_per_step) % n
    return [Path(datasets[(start + i) % n]) for i in range(min(datasets_per_step, n))]


def run_multi_group(
    dataset_paths: list[Path],
    *,
    base_url: str,
    group_size: int,
    out_dir: Path,
    query_policy_model: str,
    query_policy_temperature: float,
    query_max_tokens: int,
    rollout_timeout: int,
    arxiv_snapshot: Path,
    arxiv_index: Path,
    env_file: Path,
    generated_code_memory_mb: int,
    fetch_pdfs: bool,
    pdf_cache_dir: Path,
    method_spec_cache_dir: Path,
    generated_code_dir: Path,
    run_log: Path,
) -> dict[Path, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset_path, i) for dataset_path in dataset_paths for i in range(group_size)]
    results: dict[Path, list[dict[str, Any]]] = {d: [] for d in dataset_paths}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        future_to_dataset = {
            pool.submit(
                run_rollout,
                dataset_path,
                base_url=base_url,
                out_path=out_dir / dataset_path.stem / f"rollout_{i:03d}.json",
                rollout_index=i,
                query_policy_model=query_policy_model,
                query_policy_temperature=query_policy_temperature,
                query_max_tokens=query_max_tokens,
                timeout=rollout_timeout,
                arxiv_snapshot=arxiv_snapshot,
                arxiv_index=arxiv_index,
                env_file=env_file,
                generated_code_memory_mb=generated_code_memory_mb,
                fetch_pdfs=fetch_pdfs,
                pdf_cache_dir=pdf_cache_dir,
                method_spec_cache_dir=method_spec_cache_dir,
                generated_code_dir=generated_code_dir,
                run_log=run_log,
            ): dataset_path
            for dataset_path, i in jobs
        }
        for future, dataset_path in future_to_dataset.items():
            results[dataset_path].append(future.result())
    return results


def run_rollout(
    dataset_path: Path,
    *,
    base_url: str,
    out_path: Path,
    rollout_index: int,
    query_policy_model: str,
    query_policy_temperature: float,
    query_max_tokens: int,
    timeout: int,
    arxiv_snapshot: Path,
    arxiv_index: Path,
    env_file: Path,
    generated_code_memory_mb: int,
    fetch_pdfs: bool,
    pdf_cache_dir: Path,
    method_spec_cache_dir: Path,
    generated_code_dir: Path,
    run_log: Path,
    poll_seconds: float = 15.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_DIR.parent / "run_one_loop.py"),
        "--data", str(dataset_path),
        "--out", str(out_path),
        "--query-policy", "openai_compatible",
        "--query-policy-base-url", base_url,
        "--query-policy-model", query_policy_model,
        "--query-policy-temperature", str(query_policy_temperature),
        "--query-max-tokens", str(query_max_tokens),
        "--query-rollout-index", str(rollout_index),
        "--paper-summarizer", "openrouter",
        "--code-policy", "openrouter",
        "--arxiv-snapshot", str(arxiv_snapshot),
        "--arxiv-index", str(arxiv_index),
        "--env-file", str(env_file),
        "--generated-code-memory-mb", str(generated_code_memory_mb),
        "--pdf-cache-dir", str(pdf_cache_dir),
        "--method-spec-cache-dir", str(method_spec_cache_dir),
        "--generated-code-dir", str(generated_code_dir),
        "--run-log", str(run_log),
    ]
    if fetch_pdfs:
        command.append("--fetch-pdfs")
    tag = f"[{dataset_path.name} #{rollout_index}]"
    print(f"{_ts()} {tag} starting", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last_progress = None
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            process.kill()
            process.communicate()
            print(f"{_ts()} {tag} KILLED after {elapsed:.0f}s, exceeded {timeout}s timeout", flush=True)
            return {}
        try:
            process.wait(timeout=min(poll_seconds, max(timeout - elapsed, 1)))
            break
        except subprocess.TimeoutExpired:
            progress = _peek_progress(out_path)
            if progress != last_progress:
                print(f"{_ts()} {tag} {elapsed:.0f}s elapsed, {progress}", flush=True)
                last_progress = progress
    _, stderr = process.communicate()
    elapsed = time.monotonic() - started
    if not out_path.exists():
        print(f"{_ts()} {tag} FAILED after {elapsed:.0f}s, no output file. stderr tail: {stderr[-500:]!r}", flush=True)
        return {}
    state = json.loads(out_path.read_text(encoding="utf-8"))
    reward = (state.get("trajectory_reward") or {}).get("reward")
    outcome = (state.get("final") or {}).get("termination_reason")
    print(f"{_ts()} {tag} done in {elapsed:.0f}s, reward={reward}, outcome={outcome}", flush=True)
    return state


def _peek_progress(out_path: Path) -> str | None:
    if not out_path.exists():
        return None
    try:
        state = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "writing..."
    current_action = state.get("current_action") or {}
    tool = current_action.get("tool")
    completed = current_action.get("tool_actions_completed")
    return f"current_action={tool} tool_actions_completed={completed}" if tool else "starting..."


def extract_samples(state: dict[str, Any]) -> list[dict[str, Any]]:
    reward = float((state.get("trajectory_reward") or {}).get("reward") or 0.0)
    samples = []
    for query_action in state.get("query_actions", []):
        messages = query_action.get("messages")
        completion = query_action.get("raw_completion")
        if not messages or not completion:
            continue
        samples.append({"messages": messages, "completion": completion, "reward": reward})
    return samples


def grpo_step(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    ref_model: Any,
    kl_coef: float,
    optimizer: Any,
    enable_reasoning: bool = False,
) -> float:
    import torch

    from scripts.policy.serve_policy import render_prompt

    scored = []
    for sample in samples:
        # Must match the prompt actually shown at generation time
        # (serve_policy.py's start_server is given the same enable_reasoning
        # value) -- otherwise the log-probability computed here is under a
        # different context than the one the completion was sampled from.
        prompt_text = render_prompt(tokenizer, sample["messages"], enable_reasoning=enable_reasoning)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        full_ids = tokenizer(prompt_text + sample["completion"], return_tensors="pt").input_ids.to(model.device)
        completion_len = full_ids.shape[1] - prompt_ids.shape[1]
        if completion_len <= 0:
            continue
        scored.append((sample, full_ids, completion_len))
    if not scored:
        return 0.0

    # Micro-batched (per-sample) backward: PyTorch frees a sample's forward
    # graph as soon as its backward() completes, so peak activation memory is
    # O(one sample) instead of O(group_size) -- gradients still accumulate
    # additively into .grad across the calls, giving the same update a single
    # batched backward over the whole group would have produced, as long as
    # each sample's loss is scaled by 1/n_scored before its own backward().
    optimizer.zero_grad()
    n_scored = len(scored)
    total_loss_value = 0.0
    for sample, full_ids, completion_len in scored:
        token_logprobs = _completion_token_logprobs(model, full_ids, completion_len)
        loss = -sample["advantage"] * token_logprobs.sum()
        if ref_model is not None:
            with torch.no_grad():
                ref_token_logprobs = _completion_token_logprobs(ref_model, full_ids, completion_len)
            loss = loss + kl_coef * (token_logprobs - ref_token_logprobs).sum()
        total_loss_value += float(loss.detach())
        (loss / n_scored).backward()
    optimizer.step()
    return total_loss_value / n_scored


def _completion_token_logprobs(model: Any, full_ids: Any, completion_len: int) -> Any:
    import torch

    logits = model(full_ids).logits[:, :-1, :]
    targets = full_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs[:, -completion_len:]


def load_policy_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from scripts.policy.serve_policy import load_tokenizer

    tokenizer = load_tokenizer(args.model)
    if not args.load_in_4bit:
        return AutoModelForCausalLM.from_pretrained(args.model), tokenizer

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "--load-in-4bit (the default) requires peft and bitsandbytes. "
            f"Import failed: {exc}. Install with: pip install peft bitsandbytes, "
            "or pass --no-load-in-4bit to fall back to full-precision full fine-tuning "
            "(not recommended for 7B/8B models without a large multi-GPU node)."
        ) from exc

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_reference_model(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not args.load_in_4bit:
        return AutoModelForCausalLM.from_pretrained(args.model).eval()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization_config, device_map="auto"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--arxiv-snapshot", type=Path, required=True)
    parser.add_argument("--arxiv-index", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="Absolute path to a KEY=VALUE .env file providing OPENROUTER_API_KEY to every rollout subprocess.",
    )
    parser.add_argument(
        "--fetch-pdfs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch and extract full-text PDFs for retrieved papers rather than extracting from "
        "abstracts alone. Cached by paper_id, so the network/parse cost is paid at most once per "
        "unique paper across the whole run, not once per retrieval.",
    )
    parser.add_argument("--pdf-cache-dir", type=Path, default=Path("tmp/arxiv/pdf-cache"))
    parser.add_argument(
        "--method-spec-cache-dir",
        type=Path,
        default=Path("tmp/arxiv/method-spec-cache"),
        help="Method-spec extraction depends only on the paper's own text, never on dataset or "
        "query, so it is cached by (paper_id, evidence_depth) and reused across rollouts.",
    )
    parser.add_argument(
        "--generated-code-dir",
        type=Path,
        default=Path("tmp/agentic-generated-code"),
        help="Passed through to each run_one_loop.py rollout subprocess. Relative defaults resolve "
        "against that subprocess's cwd, not this script's -- on a runtime root with no bare tmp/ "
        "directory, pass an absolute path here explicitly.",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("data/outputs/logs/agentic_trajectory_log.jsonl"),
        help="Passed through to each run_one_loop.py rollout subprocess as --run-log. Same "
        "relative-path caveat as --generated-code-dir.",
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per dataset per step.")
    parser.add_argument(
        "--datasets-per-step",
        type=int,
        default=1,
        help="How many datasets to roll out concurrently per step, each with its own "
        "--group-size rollouts.",
    )
    parser.add_argument("--query-policy-temperature", type=float, default=0.7)
    parser.add_argument("--query-max-tokens", type=int, default=700)
    parser.add_argument("--rollout-timeout", type=int, default=480)
    parser.add_argument("--generated-code-memory-mb", type=int, default=4096)
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.0, help="0 disables the reference model entirely.")
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="QLoRA: load the base model in 4-bit (NF4) and train LoRA adapters only. "
        "Use --no-load-in-4bit for full-precision full fine-tuning.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names LoRA adapters attach to.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, the served policy's prompt ends with an open <think> block and the model "
            "reasons freely before answering. Default (unset) pre-closes an empty <think></think> "
            "block so generation starts past it. Must stay consistent between rollout generation "
            "(serve_policy.py) and training-time logprob reconstruction (grpo_step) -- this single "
            "flag drives both, since it is threaded through from here to each."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
