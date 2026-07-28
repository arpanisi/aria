#!/usr/bin/env python3
"""Minimal OpenAI-chat-compatible, request-batching server over an in-memory
policy model.

Runs in the same process as the GRPO training loop, wrapping the exact model
object being trained. run_one_loop.py (via --query-policy openai_compatible)
calls this endpoint to generate each query in a rollout, including every
query inside a variable-length, multi-retrieval trajectory. Because the
server and the trainer share one model object, there is no weight-sync step:
an update is visible to the next generation immediately.

A single background worker thread is the only caller of model.generate().
Concurrent HTTP requests are queued and generated together as one batch
rather than serialized one at a time: decoding throughput is
memory-bandwidth-bound per token, so batching amortizes that cost across
however many requests arrive within a short collection window, instead of
paying it once per request in sequence.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]], *, enable_reasoning: bool = False) -> str:
    """Render messages through the model's own chat template, not a hand-rolled joiner.

    The prior implementation joined "role: content" pairs with no special
    tokens at all, so the model never saw the turn-boundary markers
    (<｜User｜>, <｜Assistant｜>, BOS) its chat template defines, and had no
    signal that it was its own turn to respond. This model's template always
    ends a generation prompt with an *open* "<think>\n" block: DeepSeek-R1
    distills are trained to begin every response inside reasoning, not to
    reason only when they choose to. enable_reasoning=False immediately
    closes that block (the standard suppression construction for this model
    family) so generation starts past it; enable_reasoning=True leaves it
    open and lets the model reason freely before answering.
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not enable_reasoning:
        rendered += "\n</think>\n\n"
    return rendered


def load_tokenizer(model_path: str) -> Any:
    """Load a tokenizer without AutoTokenizer's per-architecture subclassing.

    AutoTokenizer resolves a model-family-specific wrapper class from
    tokenizer_config.json's tokenizer_class, and some of those wrappers apply
    decode-time transformations that don't match every architecture's own
    tokenizer.json (e.g. LlamaTokenizerFast's legacy=True path assumes
    SentencePiece spacing even for models whose tokenizer.json is a plain
    byte-level BPE spec), corrupting whitespace on decode. tokenizer.json is
    self-contained for virtually every current model family, so loading it
    through the generic fast-tokenizer wrapper sidesteps any family-specific
    override, whichever model is passed in.
    """
    from pathlib import Path

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # chat_template.jinja ships as a standalone file for this checkpoint
    # rather than embedded in tokenizer_config.json's "chat_template" key.
    # Auto-discovery of that convention is transformers-version-dependent;
    # set it explicitly so rendering doesn't silently fall back to no
    # template at all on an older install.
    if not getattr(tokenizer, "chat_template", None):
        template_path = Path(model_path) / "chat_template.jinja"
        if template_path.exists():
            tokenizer.chat_template = template_path.read_text()

    return tokenizer


@dataclass
class _PendingRequest:
    prompt_text: str
    max_new_tokens: int
    temperature: float
    ready: threading.Event = field(default_factory=threading.Event)
    completion: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def start_server(
    model: Any,
    tokenizer: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_batch_size: int = 32,
    max_wait_seconds: float = 0.4,
    enable_reasoning: bool = False,
) -> ThreadingHTTPServer:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    request_queue: queue.Queue[_PendingRequest] = queue.Queue()

    def batch_worker() -> None:
        while True:
            batch = [request_queue.get()]
            deadline = time.monotonic() + max_wait_seconds
            while len(batch) < max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(request_queue.get(timeout=remaining))
                except queue.Empty:
                    break
            _run_batch(model, tokenizer, batch)

    threading.Thread(target=batch_worker, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            return  # suppress the default raw HTTP access log line; we print our own below

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            messages = body.get("messages", [])
            prompt_text = render_prompt(tokenizer, messages, enable_reasoning=enable_reasoning)
            max_new_tokens = int(body.get("max_tokens") or 512)
            temperature = float(body.get("temperature") or 0.7)
            pending = _PendingRequest(prompt_text=prompt_text, max_new_tokens=max_new_tokens, temperature=temperature)
            print(f"{_ts()} [serve] request queued, {len(prompt_text)} prompt chars", flush=True)
            request_queue.put(pending)
            pending.ready.wait()
            self._respond({
                "choices": [{"message": {"role": "assistant", "content": pending.completion}}],
                "usage": pending.usage,
            })

        def _respond(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    class _Server(ThreadingHTTPServer):
        # Default TCP listen backlog is 5; a whole group's worth of rollouts
        # connects at once, which is routinely more than that.
        request_queue_size = 128

    server = _Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run_batch(model: Any, tokenizer: Any, batch: list[_PendingRequest]) -> None:
    import torch

    started = time.monotonic()
    print(f"{_ts()} [serve] batch of {len(batch)} starting generation", flush=True)
    prompts = [request.prompt_text for request in batch]
    max_new_tokens = max(request.max_new_tokens for request in batch)
    temperature = batch[0].temperature
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-4),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    elapsed = time.monotonic() - started
    token_counts = []
    for row, request in zip(output, batch):
        completion_ids = row[prompt_len:]
        request.completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        request.usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(completion_ids.shape[0]),
            "total_tokens": int(row.shape[0]),
        }
        token_counts.append(int(completion_ids.shape[0]))
        request.ready.set()

    # Batched generation allocates a KV-cache shaped for this batch's prompt
    # and sequence length; the training step that runs next in the same
    # process allocates very differently-shaped tensors (one sample at a
    # time). Freeing the generation tensors explicitly and releasing the
    # cached blocks back to the driver here, once per batch, avoids the two
    # workloads fragmenting against each other.
    del output, inputs
    torch.cuda.empty_cache()
    print(
        f"{_ts()} [serve] batch of {len(batch)} done in {elapsed:.1f}s, "
        f"completion tokens per request: {token_counts}",
        flush=True,
    )
