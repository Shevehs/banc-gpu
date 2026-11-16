"""Snapshot of the vLLM server configuration, to be saved with each trial.

Reads, without modifying anything:
  - on the target: /v1/models (model, maximum context length if the target is vLLM) and, if the model
    is given, a one-token completion to read system_fingerprint. In vLLM's default mode
    (--fingerprint-mode full), this fingerprint is vllm-<version>[-tpN][-ppN][-dpN][-ep]-<hash8>:
    it gives the parallelism without dev mode; the hash changes with the configuration.
    In --fingerprint-mode hash, the degrees are masked: the fingerprint then looks like that of a
    server without parallelism ("1 or masked").
  - on each vLLM server (metrics URL without /metrics): /version; in /metrics, vllm:cache_config_info
    (prefix cache, cache type and size) and the number of positions of
    vllm:spec_decode_num_accepted_tokens_per_pos (= speculative tokens per step); /server_info, which exists
    only if vLLM was started with VLLM_SERVER_DEV_MODE=1 (speculative decoding method, scheduler, etc.).
A time budget (45 s by default) caps the timeout of each request and stops the following ones once
exhausted; a server that answers drip by drip can still exceed it. The one-token completion comes
last, so that a slow target does not prevent the rest of the snapshot.
The API key is sent only to the target: /metrics, /version and /server_info do not require it
(vLLM only protects /v1, /v2, /inference and /cohere). Redirects are not followed.

With --check-model, the script sends only the one-token completion. If the server refuses the model as unknown, it
prints the server's own message and the models it serves on stderr and exits with code 3. In every other case it
exits with 0, also when the server cannot be asked (see unknown_model_message).

Usage: python tools/server_config.py --url http://host:8000 [--model bench]
       [--metrics "http://h1:8000/metrics …"] [--output server_config.json]
       python tools/server_config.py --url http://host:8000 --model bench --check-model
API key: environment variable BENCH_API_KEY. Standard library only.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

_INFO = re.compile(r'^vllm:cache_config_info\{([^}]*)\}')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_POSITION = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{[^}]*position="(\d+)"')
_FINGERPRINT = re.compile(r'^vllm-(?P<version>.+?)(?P<per>(?:-(?:tp|pp|dp)\d+|-ep)*)-(?P<hash>[0-9a-zA-Z]{8}|nohash)$')
_UNKNOWN_MODEL = re.compile(r"invalid model|unknown model|model\b.{0,80}(?:not found|does not exist)", re.IGNORECASE)
_INVALID_MODEL = re.compile(r"invalid model|unknown model", re.IGNORECASE)   # the narrower wording accepted with a 400


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # a redirect is treated as an error
        """Refuse every redirect (returning None makes urllib raise an HTTPError)."""
        return None


# No proxy: the target and the /metrics endpoints are on the private network, and the key must not go through a proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def _request(url, key=None, body=None, timeout_s=5):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with _OPENER.open(req, timeout=timeout_s) as r:
        return r.status, r.read(2_000_000).decode("utf-8", "replace")


def _get_json(url, key=None, body=None, timeout_s=5):
    try:
        s, t = _request(url, key, body, timeout_s)
        j = json.loads(t) if s == 200 else None
        return j if isinstance(j, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _as_dict(x):
    return x if isinstance(x, dict) else {}


def _error_body(text):
    """(message, error object) of an OpenAI-style error body {"error": {...}}; else the start of the text and {}.
    The message is cut to 300 characters and its white space is collapsed, so it fits on one line of a log."""
    try:
        error = json.loads(text).get("error")
    except Exception:  # noqa: BLE001
        error = None
    if isinstance(error, dict):
        message, fields = str(error.get("message") or ""), error
    else:
        message, fields = (error if isinstance(error, str) else text), {}
    return " ".join(message.split())[:300], fields


def _refusal(code, text):
    """The server's message if this error answer (HTTP status `code`, body `text`) refuses the model as unknown.

    A 404 counts when the error names the model (vLLM, OpenAI). A 400 counts only for the explicit wording of LiteLLM
    ("Invalid model name") or the code model_not_found: other 400 answers can mention a model for another reason."""
    message, error = _error_body(text)
    if error.get("code") == "model_not_found":
        return message if code in (400, 404) else None
    if code == 404 and (error.get("param") == "model" or _UNKNOWN_MODEL.search(message)):
        return message
    if code == 400 and _INVALID_MODEL.search(message):
        return message
    return None


def _chat_probe(url, model, key=None, timeout_s=15):
    """The one-token completion: (the response object or None, the server's message if it refuses `model` as
    unknown or None). Any other failure (unreachable, key refused, wrong address, server error) gives (None, None)."""
    body = {"model": model, "max_tokens": 1, "temperature": 0, "messages": [{"role": "user", "content": "ok"}]}
    try:
        status, text = _request(url.rstrip("/") + "/v1/chat/completions", key, body, timeout_s)
    except urllib.error.HTTPError as e:
        try:
            text = e.read(20_000).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            text = ""
        finally:
            e.close()
        return None, _refusal(e.code, text)
    except Exception:  # noqa: BLE001
        return None, None
    try:
        response = json.loads(text) if status == 200 else None
    except ValueError:
        response = None
    return (response if isinstance(response, dict) else None), None


def unknown_model_message(url, model, key=None, timeout_s=15):
    """The server's own message if it refuses `model` as unknown, otherwise None.

    A server refuses an unknown model with HTTP 404 (vLLM, OpenAI) or HTTP 400 "Invalid model name" (LiteLLM).
    Everything else gives None, because other checks report it: the model is accepted (including by servers that
    accept any name), the server is unreachable, the key is refused (401), the address is wrong (a 404 that does not
    speak about the model), or another error."""
    return _chat_probe(url, model, key, timeout_s)[1]


def served_model_ids(url, key=None):
    """Ids listed by the target's /v1/models; empty if it cannot say."""
    models = _get_json(url.rstrip("/") + "/v1/models", key, timeout_s=5)
    data = models.get("data") if models else None
    return [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)] if isinstance(data, list) else []


def check_model(url, model, key=None):
    """--check-model: returns 3 and prints a message on stderr if the target refuses `model` as unknown, 0 otherwise
    (also when the target cannot be asked: the later steps report it)."""
    message = unknown_model_message(url, model, key)
    if message is None:
        return 0
    print(f'[config] ERROR: the server does not know the model "{model}": {message}', file=sys.stderr)
    served = served_model_ids(url, key)
    if served:
        print("[config] Models served: " + ", ".join(served[:10]) + (" …" if len(served) > 10 else ""), file=sys.stderr)
    return 3


def read_fingerprint(e):
    """Decode system_fingerprint (vLLM full mode). Missing degree = 1 (or masked in hash mode)."""
    if not isinstance(e, str) or len(e) > 300:
        return None
    m = _FINGERPRINT.match(e)
    if not m:
        return None
    per = m.group("per")
    deg = lambda k: int(re.search(rf"-{k}(\d+)", per).group(1)) if re.search(rf"-{k}(\d+)", per) else 1
    return {"version": m.group("version"), "tensor_parallel_size": deg("tp"), "pipeline_parallel_size": deg("pp"),
            "data_parallel_size": deg("dp"), "enable_expert_parallel": "-ep" in per, "hash_config": m.group("hash"),
            "masking_possible": not per}  # no segment: TP=PP=DP=1 without EP, or hash mode which masks everything


def _configuration(cfg):
    """Extract the useful fields from vllm_config (JSON format of /server_info)."""
    if not isinstance(cfg, dict):
        return None
    per, mod, cache, sched = (_as_dict(cfg.get(k)) for k in ("parallel_config", "model_config", "cache_config",
                                                         "scheduler_config"))
    spec = _as_dict(cfg.get("speculative_config"))
    guard = lambda d, ks: {k: d.get(k) for k in ks if k in d}
    return {
        "parallelism": guard(per, ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                                    "enable_expert_parallel", "decode_context_parallel_size",
                                    "prefill_context_parallel_size"]),
        "model": guard(mod, ["model", "served_model_name", "dtype", "quantization", "max_model_len"]),
        "cache": guard(cache, ["cache_dtype", "enable_prefix_caching", "gpu_memory_utilization", "block_size",
                               "num_gpu_blocks"]),
        "scheduler": guard(sched, ["max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill",
                                      "async_scheduling"]),
        "speculative_decoding": guard(spec, ["method", "num_speculative_tokens"]) if spec else None,
    }


def collect(url, metrics=None, key=None, model=None, budget_s=45):
    """Dictionary describing the target and each vLLM server. Does not raise on an unexpected response."""
    end = time.time() + budget_s
    remaining = lambda d: max(1.0, min(d, end - time.time()))
    url = url.rstrip("/")
    urls_m = list(metrics or []) or [url + "/metrics"]
    res = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "target": url, "given_metrics": bool(metrics),
           "servers": [], "fingerprint": None}
    models = _get_json(url + "/v1/models", key, timeout_s=remaining(5))
    data = models.get("data") if models else None
    res["target_models"] = [{"id": m.get("id"), "max_model_len": m.get("max_model_len")}
                            for m in data if isinstance(m, dict)] if isinstance(data, list) else None
    for um in urls_m:
        base = um[: -len("/metrics")] if um.endswith("/metrics") else um
        srv = {"metrics": um, "version": None, "cache_config": None, "speculative_tokens_metrics": None,
               "configuration": None, "vllm_lines": None, "metrics_status": None}
        if time.time() < end:
            v = _get_json(base + "/version", timeout_s=remaining(5))
            srv["version"] = v.get("version") if v else None
        try:
            if time.time() >= end:
                srv["metrics_status"] = "budget"
            s, t = _request(um, timeout_s=remaining(5)) if time.time() < end else (0, "")
            if s:
                srv["metrics_status"] = s
            positions, vllm_lines = set(), 0
            for row in t.splitlines() if s == 200 else []:
                vllm_lines += row.startswith("vllm:")
                m = _INFO.match(row)
                if m and srv["cache_config"] is None:
                    srv["cache_config"] = {k: v for k, v in _LABEL.findall(m.group(1))
                                           if k in ("cache_dtype", "enable_prefix_caching", "gpu_memory_utilization",
                                                    "block_size", "num_gpu_blocks", "engine")}
                p = _POSITION.match(row)
                if p:
                    positions.add(p.group(1))
            srv["vllm_lines"] = vllm_lines if s == 200 else None
            if s == 200 and vllm_lines:
                srv["speculative_tokens_metrics"] = len(positions)  # 0 = no speculative decoding (INFERRED)
        except urllib.error.HTTPError as e:
            srv["metrics_status"] = e.code  # 401/403: address protected by a key (LiteLLM), not vLLM's
        except Exception:  # noqa: BLE001
            pass
        # First call sometimes slow (vLLM collects the system environment): longer timeout.
        info = _get_json(base + "/server_info?config_format=json", timeout_s=remaining(20)) if time.time() < end else None
        srv["configuration"] = _configuration(info.get("vllm_config")) if info else None
        res["servers"].append(srv)
    if model and time.time() >= end:
        res["fingerprint"] = {"raw": None, "decoded": None, "response": None, "budget_exhausted": True}
    elif model:  # a one-token completion, last and before any measurement
        r, refused = _chat_probe(url, model, key, timeout_s=remaining(15))
        e = r.get("system_fingerprint") if r else None
        res["fingerprint"] = {"raw": e if isinstance(e, str) else None,
                            "decoded": read_fingerprint(e),
                            "response": r is not None}
        if refused:
            res["fingerprint"]["unknown_model"] = refused
    return res


def deduced_gpus(r):
    """Number of GPUs behind the target, deduced from a collected `r` (the content of server_config.json):
    TP × PP × DP read from the target fingerprint, times the number of vLLM servers found (one metrics URL per
    replica; all assumed identical to the one that answered). Prefill context parallelism (PCP), absent from
    the fingerprint, is multiplied in when /server_info was read.

    Returns (count or None, explanation, masking_possible). `masking_possible` is True when the fingerprint
    carries no parallelism degree (--fingerprint-mode hash would hide it): the count then assumes 1 GPU per
    server. None is returned, with the reason, when nothing allows deducing it."""
    try:
        dec = (r.get("fingerprint") or {}).get("decoded")
        if not dec:
            return None, "fingerprint absent (one-token completion not done, or --fingerprint-mode none or custom)", False
        # A vLLM server is recognised by vllm: lines in /metrics (older files: version or metrics read)
        servers = [s for s in (r.get("servers") or [])
                   if ((s.get("vllm_lines") or 0) > 0 if "vllm_lines" in s
                       else s.get("speculative_tokens_metrics") is not None or bool(s.get("cache_config")))]
        if not servers:
            return None, "no metrics URL answers like a vLLM server (BENCH_METRICS empty or wrong)", False
        tp, pp, dp = (int(dec[k]) for k in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"))
        par = ((servers[0].get("configuration") or {}).get("parallelism") or {})
        pcp = int(par.get("prefill_context_parallel_size") or 1)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return None, f"server_config.json has an unexpected shape ({type(e).__name__})", False
    n = tp * pp * dp * pcp * len(servers)
    return n, (f"{n} (TP {tp} × PP {pp} × DP {dp}" + (f" × PCP {pcp}" if pcp > 1 else "")
               + f" × {len(servers)} vLLM server(s))"), bool(dec.get("masking_possible"))


def run_gpus(directory):
    """deduced_gpus() for the server_config.json of a run directory; (None, reason, False) if missing or unreadable."""
    f = pathlib.Path(directory) / "server_config.json"
    if not f.exists():
        return None, "server_config.json missing", False
    try:
        return deduced_gpus(json.loads(f.read_text(encoding="utf-8")))
    except (ValueError, OSError) as e:
        return None, f"server_config.json unreadable ({e})", False


def summary(r):
    """One line per server, for the logs."""
    dec = (r.get("fingerprint") or {}).get("decoded")
    rows = []
    if (r.get("fingerprint") or {}).get("budget_exhausted"):
        rows.append("[config] fingerprint not collected: time budget exhausted (slow servers)")
    elif r.get("fingerprint"):
        rows.append(f"[config] target fingerprint: {r['fingerprint'].get('raw') or 'absent'}"
                      + (f" -> TP {dec['tensor_parallel_size']}, PP {dec['pipeline_parallel_size']}, "
                         f"DP {dec['data_parallel_size']}, EP {'yes' if dec['enable_expert_parallel'] else 'no'}"
                         + (" (or masked if --fingerprint-mode hash)" if dec["masking_possible"] else "")
                         if dec else "")
                      + ("" if r["fingerprint"].get("response") else " (no response: check the model, key or target)"))
        if r["fingerprint"].get("unknown_model"):
            rows.append(f"[config] ERROR: the server does not know the model: {r['fingerprint']['unknown_model']}")
    n, expl, masked = deduced_gpus(r)
    if n:
        rows.append(f"[config] GPUs deduced: {expl}" + ("; 1 GPU per server unless --fingerprint-mode hash hides "
                                                         "the parallelism" if masked else ""))
    for s in r["servers"]:
        cc = s["cache_config"] or {}
        sp = (s["configuration"] or {}).get("speculative_decoding")
        js = s.get("speculative_tokens_metrics")
        rows.append(f"[config] {s['metrics']}: vLLM {s['version'] or '?'}; prefix cache "
                      f"{cc.get('enable_prefix_caching', '?')}; speculative tokens (metrics) "
                      f"{js if js is not None else '?'}; /server_info "
                      + (f"read (speculative decoding: {sp or 'none'})" if s["configuration"] else
                         "absent or not read (VLLM_SERVER_DEV_MODE not enabled?)"))
        st = s.get("metrics_status")
        if s.get("vllm_lines") == 0 or st in (401, 403, 307, 308):
            rows.append(f"[config] WARNING: {s['metrics']} is not vLLM's /metrics ("
                        + ("no vllm: metric" if s.get("vllm_lines") == 0 else f"HTTP {st}: key required")
                        + ", LiteLLM address?): preemptions, cache and KV cache will not be read; put vLLM's "
                        "/metrics address in BENCH_METRICS (e.g. http://<vllm-host>:8000/metrics).")
        elif st == "budget":
            rows.append(f"[config] WARNING: {s['metrics']} not read (time budget exhausted, slow servers).")
        elif s.get("vllm_lines") is None:
            rows.append(f"[config] WARNING: {s['metrics']} unreachable"
                        + (f" (HTTP {st})" if isinstance(st, int) else "")
                        + ": preemptions, cache and KV cache will not be read.")
    return "\n".join(rows)


def main():
    """Command-line entry point: collect the server configuration, print it as JSON (or write it to --output)
    and print the summary on stderr."""
    a = argparse.ArgumentParser()
    a.add_argument("--url", required=True); a.add_argument("--metrics", default="")
    a.add_argument("--model", default=None, help="if given, a one-token completion to read the fingerprint")
    a.add_argument("--output", default=None)
    a.add_argument("--check-model", action="store_true",
                   help="only check that the target knows --model: exit code 3 and its message if not, 0 otherwise")
    p = a.parse_args()
    key = os.environ.get("BENCH_API_KEY") or None  # standalone: run as a file by the .sh scripts
    if p.check_model:
        if not p.model:
            a.error("--check-model needs --model")
        sys.exit(check_model(p.url, p.model, key))
    r = collect(p.url, p.metrics.split(), key, p.model)
    text = json.dumps(r, indent=1, ensure_ascii=False)
    if p.output:
        with open(p.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    print(summary(r), file=sys.stderr)


if __name__ == "__main__":
    main()
