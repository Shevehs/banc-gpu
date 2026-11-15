"""Shared utilities for the test agents: measured model call, per-call log, synthetic content.

No tool acts on a real system: all the "tool outputs" are fabricated here,
deterministically from a seed. Nothing leaves the machine except the calls to the
model server (URL given to the launcher).
"""
import json
import pathlib
import random
import ssl
import threading
import time

import httpx
from openai import OpenAI

from tools.settings import read_setting

# ---------------------------------------------------------------- log
_lock = threading.Lock()


class CallLog:
    """Write one JSON line per model call (calls.jsonl) and per task (tasks.jsonl)."""

    def __init__(self, out_dir):
        self.f_calls = open(f"{out_dir}/calls.jsonl", "a", encoding="utf-8")
        self.f_tasks = open(f"{out_dir}/tasks.jsonl", "a", encoding="utf-8")

    def log_call(self, **row):
        """Append one call row to calls.jsonl (thread-safe)."""
        with _lock:
            self.f_calls.write(json.dumps(row) + "\n"); self.f_calls.flush()

    def log_task(self, **row):
        """Append one task row to tasks.jsonl (thread-safe)."""
        with _lock:
            self.f_tasks.write(json.dumps(row) + "\n"); self.f_tasks.flush()

    def close(self):
        """Close both log files."""
        self.f_calls.close(); self.f_tasks.close()


# ---------------------------------------------------------------- synthetic content
# Target size in tokens. If a tokenizer is available (a tokenizer.json file in BENCH_TOKENIZER,
# or tools/tokenizer/ in the repository), the size is adjusted to the real token count. Otherwise,
# it is estimated with a characters-per-token ratio specific to each kind of text, calibrated on the
# Qwen BPE vocabulary (INFERRED: same vocabulary as Qwen3.8): each digit counts as one
# token, hence the low ratios for logs and scans.
CHARS_PER_TOKEN = {"code": 3.0, "scan": 2.0, "log": 1.9, "notes": 3.2}

_WORDS = ("buffer index length offset parse header token session config request response "
         "handler payload decode encode user input output value check bound size alloc free "
         "path file socket port host service version module function return error status").split()
_TOKENIZER = {"loaded": False, "tok": None}


def tokenizer():
    """Local tokenizer (tokenizers library), or None."""
    if not _TOKENIZER["loaded"]:
        _TOKENIZER["loaded"] = True
        root = read_setting("TOKENIZER") or str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "tokenizer")
        f = pathlib.Path(root) / "tokenizer.json" if pathlib.Path(root).is_dir() else pathlib.Path(root)
        if f.is_file():
            try:
                from tokenizers import Tokenizer
                _TOKENIZER["tok"] = Tokenizer.from_file(str(f))
            except Exception:  # noqa: BLE001 — without a tokenizer, keep the estimate
                _TOKENIZER["tok"] = None
    return _TOKENIZER["tok"]


def _make_text(kind, n_car, seed):
    r = random.Random(seed)
    rows, n, i = [], 0, 0
    while n < n_car:
        i += 1
        m = [r.choice(_WORDS) for _ in range(6)]
        if kind == "code":
            l = f"int {m[0]}_{i}(char *{m[1]}, size_t {m[2]}) {{ if ({m[2]} > {r.randint(8, 4096)}) return -1; memcpy({m[3]}, {m[1]}, {m[2]}); }} // {m[4]} {m[5]}"
        elif kind == "scan":
            l = f"10.0.{r.randint(0, 255)}.{r.randint(1, 254)}  {r.choice([22, 80, 443, 445, 3389, 8080])}/tcp open  {m[0]}-{m[1]} v{r.randint(1, 9)}.{r.randint(0, 20)}  ({m[2]} {m[3]})"
        elif kind == "log":
            l = f"2026-11-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:00Z host-{r.randint(1, 500)} {m[0]}.{m[1]} status={r.choice([200, 401, 403, 500])} {m[2]}={r.randint(0, 99999)} {m[3]} {m[4]}"
        else:
            l = f"- Finding {i}: {m[0]} {m[1]} on {m[2]}, impact {m[3]}, evidence {m[4]}-{r.randint(100, 999)}, status {m[5]}."
        rows.append(l); n += len(l) + 1
    return "\n".join(rows)


def make_text(kind, n_tokens, seed):
    """Fabricated text of about n_tokens tokens: 'code', 'scan', 'log' or 'notes'."""
    ratio = CHARS_PER_TOKEN.get(kind, 3.0)
    t = _make_text(kind, n_tokens * ratio, seed)
    tok = tokenizer()
    if tok is not None and n_tokens >= 50:
        actual = len(tok.encode(t).ids)
        if actual and abs(actual - n_tokens) / n_tokens > 0.02:
            t = _make_text(kind, n_tokens * ratio * n_tokens / actual, seed)
    return t


def tool_definitions(names, n_tokens_per_tool, seed):
    """Tool definitions in OpenAI format, with a description that gives the target size."""
    tools = []
    for k, name in enumerate(names):
        tools.append({"type": "function", "function": {
            "name": name,
            "description": make_text("notes", n_tokens_per_tool, seed + k),
            "parameters": {"type": "object",
                           "properties": {"target": {"type": "string"}, "detail": {"type": "string"}},
                           "required": ["target"]}}})
    return tools


# ---------------------------------------------------------------- measured call
class Model:
    """OpenAI client for the server under test; each call is timed and logged."""

    def __init__(self, url, model, log, mode, max_delay_s=1800, api_key="bench", label=False):
        # max_retries=0: an error must be logged, not hidden by a client retry.
        # trust_env=False: ignore HTTP(S)_PROXY; the target is on the private network and the key must not go
        # through a proxy (AIPerf does the same).
        self.client = OpenAI(base_url=url.rstrip("/") + "/v1", api_key=api_key, timeout=max_delay_s, max_retries=0,
                             http_client=httpx.Client(trust_env=False, verify=ssl.create_default_context()))
        self.model, self.log, self.mode = model, log, mode
        # label: adds "metadata": {"agent": <profile>} to each request, so that LiteLLM breaks down its
        # metrics by agent (custom_prometheus_metadata_labels). vLLM called directly ignores this field.
        self.label = label

    def call(self, messages, n_output, ctx, tools=None):
        """One streaming call. ctx = dict (agent, task, thread, step) copied into the log.
        'scripted' mode: output forced to n_output tokens (min_tokens + ignore_eos, vLLM parameters),
        no tool call. 'autonomous' mode: tools passed to the model, free length (capped at n_output)."""
        args = dict(model=self.model, messages=messages, max_tokens=n_output, stream=True,
                    stream_options={"include_usage": True}, temperature=0.7)
        if self.mode == "scripted":
            args["extra_body"] = {"min_tokens": n_output, "ignore_eos": True}
        elif tools:
            args["tools"] = tools; args["tool_choice"] = "auto"
        if self.label and ctx.get("agent"):
            args.setdefault("extra_body", {})["metadata"] = {"agent": ctx["agent"]}
        t0 = time.time(); t1 = None; output_text = []; tool_call_count = {}; usage = None; chunks = 0
        error = None
        try:
            for ev in self.client.chat.completions.create(**args):
                if ev.usage is not None:
                    usage = ev.usage
                if not ev.choices:
                    continue
                d = ev.choices[0].delta
                # Reasoning (vLLM reasoning parser): field "reasoning" or "reasoning_content".
                extra = getattr(d, "model_extra", None) or {}
                reasoning_text = extra.get("reasoning") or extra.get("reasoning_content") \
                    or getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
                if (d.content or d.tool_calls or reasoning_text) and t1 is None:
                    t1 = time.time()
                if reasoning_text:
                    chunks += 1
                if d.content:
                    output_text.append(d.content); chunks += 1
                for tc in d.tool_calls or []:
                    a = tool_call_count.setdefault(tc.index, {"name": "", "args": ""})
                    if tc.function and tc.function.name: a["name"] += tc.function.name
                    if tc.function and tc.function.arguments: a["args"] += tc.function.arguments
                    chunks += 1
        except Exception as e:  # noqa: BLE001 — the error is logged; the task continues or stops
            error = f"{type(e).__name__}: {e}"[:300]
        t2 = time.time()
        n_out = getattr(usage, "completion_tokens", None)
        n_in = getattr(usage, "prompt_tokens", None)
        det = getattr(usage, "prompt_tokens_details", None)
        n_cache = getattr(det, "cached_tokens", None) if det is not None else None
        speed = None
        if t1 is not None and n_out and n_out > 1 and t2 > t1:
            speed = (n_out - 1) / (t2 - t1)
        self.log.log_call(**ctx, t_start=t0, ttft_s=(t1 - t0) if t1 else None, duration_s=t2 - t0,
                           input_tokens=n_in, output_tokens=n_out, cached_tokens=n_cache,
                           chunks=chunks, tool_call_count=len(tool_call_count), speed=speed, error=error)
        if error:
            raise RuntimeError(error)
        return "".join(output_text), list(tool_call_count.values())
