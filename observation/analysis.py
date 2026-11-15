"""Reading /metrics (Prometheus format) and indicators for an observation window.
Standard library only. Used by observer.py and report/observation.py."""
import math
import re
import urllib.error
import urllib.parse
import urllib.request

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# Names as of vLLM commit 7f855f6 (vllm/v1/metrics/loggers.py); the _total suffix of counters is stripped.
COUNTERS = {
    "output_tokens": "vllm:generation_tokens",
    "input_tokens": "vllm:prompt_tokens",
    "completed_requests": "vllm:request_success",
    "preemptions": "vllm:num_preemptions",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "mtp_accepted": "vllm:spec_decode_num_accepted_tokens",
    "mtp_proposed": "vllm:spec_decode_num_draft_tokens",
    "failed_requests": "http_requests",
}
HISTOGRAMS = {
    "ttft": "vllm:time_to_first_token_seconds",
    # per-request mean of the time between tokens, first token excluded (vllm/v1/metrics/stats.py)
    "tpot": "vllm:request_time_per_output_token_seconds",
}
GAUGES = {"kv": "vllm:kv_cache_usage_perc", "waiting": "vllm:num_requests_waiting",
          "running": "vllm:num_requests_running"}


class _BlockRedirects(urllib.request.HTTPRedirectHandler):
    """Follows only one kind of redirect: same host, same path plus a trailing "/" (LiteLLM serves /metrics/
    and redirects /metrics to /metrics/). Any other redirect is refused, so the key is never sent elsewhere."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Follow only the same-host redirect that adds a trailing "/".

        Raises:
            urllib.error.HTTPError: for any other redirect."""
        a, b = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl))
        if (a.scheme, a.netloc) == (b.scheme, b.netloc) and b.path == a.path.rstrip("/") + "/":
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


# No proxy: /metrics are on the private network, and the LiteLLM key must not go through a proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _BlockRedirects)
HTTP_ERRORS = "http_requests"   # counter from prometheus_fastapi_instrumentator, exposed by vLLM


def fetch_series(url, timeout_s=5, prefix=("vllm:", HTTP_ERRORS), key=None):
    """List of (name, labels, value) from a /metrics endpoint, for names that start with `prefix`.
    `key` is only given for LiteLLM's /metrics (protected by default)."""
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with _OPENER.open(req, timeout=timeout_s) as r:
        text = r.read().decode("utf-8", "replace")
    res = []
    for row in text.splitlines():
        m = _LINE.match(row)
        if not m or not m.group(1).startswith(prefix):
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        name = m.group(1)
        if name.endswith("_total"):
            name = name[:-6]
        res.append((name, dict(_LABEL.findall(m.group(2) or "")), v))
    return res


def summarize(series):
    """Counters summed by name, histograms summed by bound, gauges kept per series."""
    counters, histograms, gauges = {}, {k: {} for k in HISTOGRAMS}, {k: [] for k in GAUGES}
    names_c = set(COUNTERS.values()) - {HTTP_ERRORS}
    for name, lab, v in series:
        if name in names_c:
            counters[name] = counters.get(name, 0.0) + v
        # Failed inference requests (4xx, 5xx): http_requests_total{handler="/v1/…", status="4xx"|"5xx"}
        # (prometheus_fastapi_instrumentator, status codes grouped by default; INFERRED, check on the installed version).
        # An error series only appears at the first error: as soon as a /v1 series exists, the count starts at 0.
        if name == HTTP_ERRORS and lab.get("handler", "").startswith("/v1/"):
            counters.setdefault(HTTP_ERRORS, 0.0)
            if lab.get("status", "")[:1] in ("4", "5"):
                counters[HTTP_ERRORS] += v
        for k, h in HISTOGRAMS.items():
            if name == h + "_bucket" and "le" in lab:
                histograms[k][lab["le"]] = histograms[k].get(lab["le"], 0.0) + v
        for k, j in GAUGES.items():
            if name == j:
                gauges[k].append(v)
    return counters, histograms, gauges


def _bound(le):
    return math.inf if le in ("+Inf", "Inf", "inf") else float(le)


def cumulative(h):
    """Sorted [(bound, cumulative count)] from a {le: count} dict."""
    return sorted(((_bound(k), v) for k, v in h.items()), key=lambda x: x[0])


def quantile(h, q):
    """Upper bound of the bucket that contains quantile q (conservative: the true quantile is at most there).
    Returns (bound, beyond); beyond = the quantile exceeds the last finite bound (which is returned)."""
    c = cumulative(h)
    if not c or c[-1][1] <= 0:
        return None, False
    total = c[-1][1]; target = q * total; low = 0.0
    for b, n in c:
        if n >= target:
            return (low, True) if math.isinf(b) else (b, False)
        low = b
    return low, True


def share_below(h, x):
    """Share of observations <= x: (estimate, exact, minimum, maximum).
    Exact if a bound falls within 1 % of x; otherwise interpolated, between minimum (cumulative count at the
    lower bound) and maximum (cumulative count at the upper bound). (None, None, None, None) if no observations."""
    c = cumulative(h)
    if not c or c[-1][1] <= 0:
        return None, None, None, None
    total = c[-1][1]
    for b, n in c:
        if not math.isinf(b) and abs(b - x) <= 0.01 * x:
            return n / total, True, n / total, n / total
    low, c_low = 0.0, 0.0
    for b, n in c:
        if b >= x:
            if math.isinf(b):
                return c_low / total, False, c_low / total, 1.0
            return (c_low + (n - c_low) * (x - low) / (b - low)) / total, False, c_low / total, n / total
        low, c_low = b, n
    return 1.0, False, 1.0, 1.0


def indicators(obs, target, gpu, ttft_max_ms=None):
    """Indicators for a window (dict read from observation.json)."""
    d = obs["end"] - obs["start"]; c = obs["counters"]; h = obs["histograms"]
    ratio = lambda a, b: a / b if a is not None and b else None
    output = c.get("output_tokens")
    tpot90, beyond = quantile(h["tpot"], 0.90)  # upper bound -> P10 interactivity is at least 1/bound
    share, exact, share_min, share_max = share_below(h["tpot"], 1.0 / target)
    ttft50, _ = quantile(h["ttft"], 0.50); ttft95, ttft_beyond = quantile(h["ttft"], 0.95)
    r = {
        "completed_requests": c.get("completed_requests"),
        "output_throughput_per_gpu": output / d / gpu if output is not None and d > 0 else None,
        "share_of_requests_at_target": share, "share_exact": exact, "share_min": share_min, "share_max": share_max,
        "failed_requests": c.get("failed_requests"),
        "estimated_interactivity_p10": (1.0 / tpot90) if tpot90 else None, "interactivity_bound": beyond,
        "estimated_ttft_p50_ms": ttft50 * 1000 if ttft50 is not None else None,
        "estimated_ttft_p95_ms": ttft95 * 1000 if ttft95 is not None else None, "ttft_p95_beyond": ttft_beyond,
        "preemptions": c.get("preemptions"),
        "prefix_cache": ratio(c.get("prefix_cache_hits"), c.get("prefix_cache_queries")),
        "mtp_acceptance": ratio(c.get("mtp_accepted"), c.get("mtp_proposed")),
        "kv_cache_max_fraction": obs["gauges"].get("kv_max"),
        "max_waiting_requests": obs["gauges"].get("waiting_max"),
    }
    # Three-valued verdict: True (holds), False (does not hold), None (undetermined: the target falls in a
    # bucket that straddles 90 %, or there were no requests). Errors and preemptions fail it; TTFT only if requested.
    if (r["preemptions"] or 0) > 0 or (r["failed_requests"] or 0) > 0:
        r["holds"] = False
    elif share is None:
        r["holds"] = None
    elif share_max < 0.90:
        r["holds"] = False
    elif ttft_max_ms is not None and (r["estimated_ttft_p95_ms"] is None or ttft_beyond
                                      or r["estimated_ttft_p95_ms"] > ttft_max_ms):
        r["holds"] = False
    else:
        r["holds"] = True if share_min >= 0.90 else None
    return r


def verdict(t):
    """Text of a three-valued verdict: True -> "holds", False -> "does not hold", None -> "undetermined"."""
    return {True: "holds", False: "does not hold", None: "undetermined"}[t]


# ---- Per-agent view: LiteLLM Prometheus metrics ("prometheus" callback), split by a label
# (api_key_alias = one virtual key per agent, or metadata_<field_name> = custom_prometheus_metadata_labels).
# Names from https://docs.litellm.ai/docs/proxy/prometheus; checked against LiteLLM 1.109.1 during validation.
LITELLM = {
    "output_tokens": "litellm_output_tokens_metric",
    "input_tokens": "litellm_input_tokens_metric",
    "ttft": "litellm_llm_api_time_to_first_token_metric",     # histogram (s), streaming requests only
    "latency": "litellm_request_total_latency_metric",        # histogram (s), end to end
}


def summarize_litellm(series, label):
    """{agent: {output_tokens, input_tokens, ttft{le}, ttft_sum, ttft_count, latency{le}, latency_sum, latency_count}}"""
    res = {}
    for name, lab, v in series:
        ag = lab.get(label)
        if ag is None:
            continue
        a = res.setdefault(ag, {"output_tokens": 0.0, "input_tokens": 0.0, "ttft": {}, "ttft_sum": 0.0,
                                "ttft_count": 0.0, "latency": {}, "latency_sum": 0.0, "latency_count": 0.0})
        for k in ("output_tokens", "input_tokens"):
            if name == LITELLM[k]:
                a[k] += v
        for k in ("ttft", "latency"):
            if name == LITELLM[k] + "_bucket" and "le" in lab:
                a[k][lab["le"]] = a[k].get(lab["le"], 0.0) + v
            elif name == LITELLM[k] + "_sum":
                a[k + "_sum"] += v
            elif name == LITELLM[k] + "_count":
                a[k + "_count"] += v
    return res


def delta_litellm(start, end):
    """Per-agent change between two summaries; an agent absent at the start counts from zero."""
    res = {}
    for ag, b in end.items():
        a = start.get(ag) or {}
        d = {}
        for k, v in b.items():
            if isinstance(v, dict):
                d[k] = {le: n - (a.get(k) or {}).get(le, 0.0) for le, n in v.items()}
            else:
                d[k] = v - (a.get(k) or 0.0)
        if d["latency_count"] > 0 or d["output_tokens"] > 0:
            res[ag] = d
    return res


def agent_indicators(agents):
    """Per-agent indicators. Mean speed = sum of output / (sum of latency - sum of TTFT), only if all the
    agent's requests are streamed (otherwise TTFT is missing): a mean, not a percentile (INFERRED)."""
    total_output = sum(a["output_tokens"] for a in agents.values()) or None
    res = {}
    for ag, a in sorted(agents.items()):
        n = a["latency_count"]
        t50, _ = quantile(a["ttft"], 0.50); t95, t95_beyond = quantile(a["ttft"], 0.95)
        l50, _ = quantile(a["latency"], 0.50); l95, l95_beyond = quantile(a["latency"], 0.95)
        decode = a["latency_sum"] - a["ttft_sum"]
        res[ag] = {
            "requests": n,
            "output_token_share": a["output_tokens"] / total_output if total_output else None,
            "input_tokens_per_request": a["input_tokens"] / n if n else None,
            "output_tokens_per_request": a["output_tokens"] / n if n else None,
            "ttft_p50_s": t50, "ttft_p95_s": t95, "ttft_p95_beyond": t95_beyond,
            "latency_p50_s": l50, "latency_p95_s": l95, "latency_p95_beyond": l95_beyond,
            "mean_speed": (a["output_tokens"] / decode) if n and a["ttft_count"] == n and decode > 0 else None,
        }
    return res
