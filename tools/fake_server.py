"""Fake OpenAI-compatible server, to check the bench scripts without a GPU or a model.

Mimics: /v1/models, streaming /v1/chat/completions (text or tool call), the usage field with the cached
tokens (prefix shared with previous requests, 4 characters per token), /metrics in Prometheus format
(a few vLLM metrics), /litellm/metrics (LiteLLM-style metrics per api_key_alias, always served), /version
and, with --dev-mode, /server_info. The numbers measure nothing.
Usage: python tools/fake_server.py --port 8001"""
import argparse
import asyncio
import json
import random

from aiohttp import web

SEEN_PROMPTS = []  # requests already seen (text), to simulate the prefix cache
OUTPUT_TOKENS = 0
INPUT_TOKENS = 0
REQUESTS = 0
HTTP = {"2xx": 0}  # like prometheus_fastapi_instrumentator: a series only appears at its first request

# Histograms mimicking vLLM (vLLM's default bounds; --itl-bounds replaces them, like --custom-histogram-buckets)
B_TTFT = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0,
          80.0, 160.0, 640.0, 2560.0]
B_ITL = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]
H = {"ttft": [], "tpot": []}  # observed values

# Imitation of LiteLLM's Prometheus metrics (served at /litellm/metrics), broken down by
# api_key_alias = Bearer key received. Format based on https://docs.litellm.ai/docs/proxy/prometheus,
# compared with LiteLLM 1.109.1 during validation.
LL = {}  # (alias, metadata) -> {"input", "output", "ttft": [], "latency": []}
B_LL = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]


def histogram(name, bounds, vals):
    """Prometheus text of a vLLM-style histogram (one series, engine 0, model bench) over `vals` (seconds)."""
    t = f"# TYPE {name} histogram\n"
    for b in bounds:
        t += f'{name}_bucket{{engine="0",le="{b}",model_name="bench"}} {sum(1 for v in vals if v <= b)}.0\n'
    t += f'{name}_bucket{{engine="0",le="+Inf",model_name="bench"}} {len(vals)}.0\n'
    t += (f'{name}_count{{engine="0",model_name="bench"}} {len(vals)}.0\n'
          f'{name}_sum{{engine="0",model_name="bench"}} {sum(vals)}\n')
    return t


def prompt_text(b):
    """Text of the request used to measure the shared prefix: messages followed by tool definitions."""
    return json.dumps(b.get("messages", []), ensure_ascii=False) + json.dumps(b.get("tools") or [])


def cache(t):
    """Simulated cached tokens of prompt `t`: longest common prefix with the last 200 prompts (4 characters
    per token, rounded down to blocks of 16); records `t` as seen."""
    best = 0
    for v in SEEN_PROMPTS[-200:]:
        n = 0
        limit = min(len(v), len(t))
        while n < limit and v[n] == t[n]:
            n += 1
        best = max(best, n)
    SEEN_PROMPTS.append(t)
    return (best // 4) // 16 * 16


async def models(r):
    """GET /v1/models: a single model, "bench"."""
    return web.json_response({"object": "list", "data": [{"id": "bench", "object": "model", "max_model_len": 32768}]})


async def chat(r):
    """POST /v1/chat/completions: simulated completion (SSE stream or one JSON body), or a 500 at --error-rate."""
    global OUTPUT_TOKENS, INPUT_TOKENS, REQUESTS
    if random.random() < ERROR_RATE:
        HTTP["5xx"] = HTTP.get("5xx", 0) + 1
        return web.json_response({"error": "simulated error"}, status=500)
    b = await r.json()
    t = prompt_text(b)
    n_in = len(t) // 4
    c = cache(t)
    n = int(b.get("max_tokens") or b.get("max_completion_tokens") or 16)
    if not b.get("ignore_eos"):
        n = max(1, int(n * random.uniform(0.3, 1.0)))
    n = max(n, int(b.get("min_tokens") or 0)) if b.get("ignore_eos") else n
    if not b.get("stream"):
        return web.json_response({
            "id": "x", "object": "chat.completion", "model": "bench", "system_fingerprint": FINGERPRINT(),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": 1, "total_tokens": n_in + 1}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(r)
    t0 = asyncio.get_running_loop().time()
    await asyncio.sleep(0.02 + (n_in - c) / 2_000_000)
    t1 = None
    tool = b.get("tools") and random.random() < 0.7
    if tool:
        name = random.choice(b["tools"])["function"]["name"]
        ch = {"choices": [{"index": 0, "delta": {"tool_calls": [
                  {"index": 0, "id": "c1", "type": "function",
                   "function": {"name": name, "arguments": "{\"target\":\"x\"}"}}]}, "finish_reason": None}],
              "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "bench"}
        await resp.write(f"data: {json.dumps(ch)}\n\n".encode())
        n = min(n, 20)
    else:
        for i in range(n):
            ch = {"choices": [{"index": 0, "delta": {"content": f" w{i % 50}"}, "finish_reason": None}],
                  "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "bench"}
            await resp.write(f"data: {json.dumps(ch)}\n\n".encode())
            if t1 is None:
                t1 = asyncio.get_running_loop().time()
            await asyncio.sleep(ARGS.token_delay)
    OUTPUT_TOKENS += n
    INPUT_TOKENS += n_in
    REQUESTS += 1
    HTTP["2xx"] += 1
    t2 = asyncio.get_running_loop().time()
    if t1 is not None:
        H["ttft"].append(t1 - t0)
        H["tpot"].append((t2 - t1) / (n - 1) if n > 1 else 0.0)
    alias = (r.headers.get("Authorization", "").removeprefix("Bearer ").strip() or "without-key")[:40]
    meta = str(((b.get("metadata") or {}).get("agent")) or "")[:40]
    d = LL.setdefault((alias, meta), {"input": 0, "output": 0, "ttft": [], "latency": []})
    d["input"] += n_in
    d["output"] += n
    d["latency"].append(t2 - t0)
    if t1 is not None:
        d["ttft"].append(t1 - t0)
    end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else "length"}],
           "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "bench"}
    await resp.write(f"data: {json.dumps(end)}\n\n".encode())
    if (b.get("stream_options") or {}).get("include_usage"):
        u = {"choices": [], "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "bench",
             "system_fingerprint": FINGERPRINT(),
             "usage": {"prompt_tokens": n_in, "completion_tokens": n, "total_tokens": n_in + n,
                       "prompt_tokens_details": {"cached_tokens": c}}}
        await resp.write(f"data: {json.dumps(u)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def histogram_ll(name, key, vals):
    """Prometheus text of a LiteLLM-style histogram for the series key = (api_key_alias, metadata_agent)."""
    lab = f'api_key_alias="{key[0]}",metadata_agent="{key[1]}",model="bench"'
    t = "".join(f'{name}_bucket{{{lab},le="{b}"}} {sum(1 for v in vals if v <= b)}.0\n' for b in B_LL)
    return (t + f'{name}_bucket{{{lab},le="+Inf"}} {len(vals)}.0\n'
            f'{name}_sum{{{lab}}} {sum(vals)}\n{name}_count{{{lab}}} {len(vals)}.0\n')


async def metrics_litellm(r):
    """GET /litellm/metrics/: LiteLLM-style counters and histograms per (api_key_alias, agent)."""
    t = ""
    for a, d in LL.items():
        lab = f'api_key_alias="{a[0]}",metadata_agent="{a[1]}",model="bench"'
        t += (f'litellm_input_tokens_metric_total{{{lab}}} {d["input"]}.0\n'
              f'litellm_output_tokens_metric_total{{{lab}}} {d["output"]}.0\n')
        t += (histogram_ll("litellm_llm_api_time_to_first_token_metric", a, d["ttft"])
              + histogram_ll("litellm_request_total_latency_metric", a, d["latency"]))
    return web.Response(text=t, content_type="text/plain")


async def metrics(r):
    """GET /metrics: vLLM-style Prometheus text (counters derived from the traffic seen, fixed gauges)."""
    t = (
        "# TYPE vllm:generation_tokens_total counter\n"
        f"vllm:generation_tokens_total{{model_name=\"bench\"}} {OUTPUT_TOKENS}\n"
        "# TYPE vllm:num_preemptions_total counter\n"
        "vllm:num_preemptions_total{model_name=\"bench\"} 0\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        "vllm:kv_cache_usage_perc{model_name=\"bench\"} 0.5\n"
        "# TYPE vllm:prefix_cache_queries_total counter\n"
        f"vllm:prefix_cache_queries_total{{model_name=\"bench\"}} {OUTPUT_TOKENS * 10}\n"
        "# TYPE vllm:prefix_cache_hits_total counter\n"
        f"vllm:prefix_cache_hits_total{{model_name=\"bench\"}} {OUTPUT_TOKENS}\n"
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
        f"vllm:spec_decode_num_draft_tokens_total{{model_name=\"bench\"}} {OUTPUT_TOKENS * 3}\n"
        "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
        f"vllm:spec_decode_num_accepted_tokens_total{{model_name=\"bench\"}} {OUTPUT_TOKENS * 2}\n"
        f"# TYPE vllm:prompt_tokens counter\n"
        f"vllm:prompt_tokens_total{{engine=\"0\",model_name=\"bench\"}} {INPUT_TOKENS}\n"
        f"# TYPE vllm:request_success counter\n"
        f"vllm:request_success_total{{engine=\"0\",finished_reason=\"length\",model_name=\"bench\"}} {REQUESTS}\n"
        + "".join(f'http_requests_total{{handler="/v1/chat/completions",method="POST",status="{k}"}} {v}.0\n'
                  for k, v in HTTP.items())
        + "# TYPE vllm:num_requests_waiting gauge\n"
        "vllm:num_requests_waiting{engine=\"0\",model_name=\"bench\"} 0.0\n"
        + histogram("vllm:time_to_first_token_seconds", B_TTFT, H["ttft"])
        + histogram("vllm:inter_token_latency_seconds", ITL_BOUNDS, H["tpot"])
        + histogram("vllm:request_time_per_output_token_seconds", ITL_BOUNDS, H["tpot"])
        + "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter\n"
        + "".join(f"vllm:spec_decode_num_accepted_tokens_per_pos_total"
                  f"{{model_name=\"bench\",engine=\"0\",position=\"{k}\"}} {OUTPUT_TOKENS}\n" for k in range(3))
        + "# TYPE vllm:cache_config_info gauge\n"
        "vllm:cache_config_info{block_size=\"16\",cache_dtype=\"auto\",enable_prefix_caching=\"True\","
        "gpu_memory_utilization=\"0.9\",num_gpu_blocks=\"1000\"} 1.0\n")
    return web.Response(text=t, content_type="text/plain")


def FINGERPRINT():
    """system_fingerprint of the fake engine (parallelism depends on --dev-mode, like /server_info)."""
    return "vllm-0.0.0-tp4-dp2-ep-0a1b2c3d" if ARGS.dev_mode else "vllm-0.0.0-tp8-ep-0a1b2c3d"


async def version(r):
    """GET /version."""
    return web.json_response({"version": "0.0.0-fake"})


async def server_info(r):
    """GET /server_info: engine configuration (only with --dev-mode, else 404)."""
    if not ARGS.dev_mode:
        return web.Response(status=404)
    return web.json_response({
        "vllm_config": {
            "parallel_config": {"tensor_parallel_size": 4, "pipeline_parallel_size": 1, "data_parallel_size": 2,
                                "enable_expert_parallel": True, "decode_context_parallel_size": 1},
            "model_config": {"model": "/weights/fake", "served_model_name": "bench", "max_model_len": 32768},
            "cache_config": {"enable_prefix_caching": True, "block_size": 16},
            "scheduler_config": {"max_num_seqs": 256, "max_num_batched_tokens": 8192,
                                 "enable_chunked_prefill": True},
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}}})


a = argparse.ArgumentParser()
a.add_argument("--token-delay", type=float, default=0.003, help="seconds between two tokens (sets the simulated speed)")
a.add_argument("--itl-bounds", default="",
               help="JSON bounds of the inter_token_latency histogram, like vLLM's --custom-histogram-buckets")
a.add_argument("--dev-mode", action="store_true", help="mimics VLLM_SERVER_DEV_MODE=1 (/server_info)")
a.add_argument("--port", type=int, default=8001)
a.add_argument("--error-rate", type=float, default=0.0)
ARGS = a.parse_args()
ITL_BOUNDS = json.loads(ARGS.itl_bounds) if ARGS.itl_bounds else B_ITL
PORT = ARGS.port
ERROR_RATE = ARGS.error_rate
app = web.Application(client_max_size=200 * 1024 * 1024)
app.add_routes([web.get("/v1/models", models), web.post("/v1/chat/completions", chat),
                web.get("/metrics", metrics), web.get("/version", version), web.get("/server_info", server_info),
                web.get("/litellm/metrics/", metrics_litellm),
                web.get("/litellm/metrics", lambda r: web.HTTPTemporaryRedirect("/litellm/metrics/"))])
web.run_app(app, host="127.0.0.1", port=PORT, print=None)
