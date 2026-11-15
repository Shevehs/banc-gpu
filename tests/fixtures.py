"""Builds small deterministic data: result trees for the report kinds, canned Prometheus answers for
observation.query."""
import calendar
import json
import pathlib
import re


def _w(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _stat(**kw):
    return kw


def aiperf_tree(root):
    """conf-a (2 repetitions, 3 levels: c1 and c4 pass, c8 misses the target), conf-b (preemption at c4), conf-c (no server metrics)."""
    root = pathlib.Path(root)
    spec = {
        # (run, concurrency): (throughput, p10, ttft p95, errors, preemptions, kv max, max waiting)
        ("conf-a-r1", 1): (100.0, 60.0, 200.0, 0, 0, 0.30, 0),
        ("conf-a-r1", 4): (320.0, 40.0, 400.0, 0, 0, 0.96, 3),
        ("conf-a-r1", 8): (480.0, 22.0, 900.0, 0, 0, 0.96, 3),
        ("conf-a-r2", 1): (104.0, 61.0, 210.0, 0, 0, 0.31, 0),
        ("conf-a-r2", 4): (340.0, 38.0, 420.0, 0, 0, 0.72, 0),
        ("conf-a-r2", 8): (500.0, 21.0, 950.0, 0, 0, 0.97, 4),
        ("conf-b-r1", 1): (90.0, 55.0, 180.0, 0, 0, 0.20, 0),
        ("conf-b-r1", 4): (300.0, 45.0, 380.0, 0, 2, 0.60, 0),
        ("conf-c-r1", 1): (80.0, 50.0, 150.0, 0, 0, 0.10, 0),  # no server metrics; single (hence highest) level
    }
    for (run, concurrency), (thr, p10, ttft95, err, pre, kv, wait) in spec.items():
        d = root / run / f"c{concurrency}"
        _w(d / "profile_export_aiperf.json", {
            "output_token_throughput": _stat(avg=thr),
            "request_count": _stat(avg=200),
            "error_request_count": _stat(avg=err),
            "output_sequence_length": _stat(avg=512.0),
            "output_token_throughput_per_user": _stat(p10=p10, p50=p10 + 10),
            "time_to_first_token": _stat(p50=ttft95 / 2, p95=ttft95, p99=ttft95 * 1.2),
            "metadata": {"submission_valid": True},
        })
        if run == "conf-c-r1":
            continue
        series = lambda s: {"series": [{"stats": s}]}
        _w(d / "server_metrics_export.json", {"metrics": {
            "vllm:num_preemptions": series({"total": pre}),
            "vllm:kv_cache_usage_perc": series({"max": kv}),
            "vllm:num_requests_waiting": series({"max": wait}),
            "vllm:prefix_cache_hits": series({"total": 50}),
            "vllm:prefix_cache_queries": series({"total": 100}),
        }})
    return root


def agents_tree(root):
    """conf-r1: levels n1 and n2; n2 has a failed task and preemptions."""
    root = pathlib.Path(root)
    for run, level, preempt in (("conf-r1", 1, 0), ("conf-r1", 2, 3)):
        d = root / run / f"n{level}"
        d.mkdir(parents=True)
        t0, warm, t1 = 1000.0, 10.0, 1100.0
        _w(d / "level.json", {"concurrent_tasks": level, "t_start": t0, "t_end": t1, "warmup": warm,
                               "agents": ["code_review", "pentest"],
                               "metrics": {"preemptions": preempt, "prefix_cache_hits": 30.0,
                                             "prefix_cache_queries": 60.0, "mtp_accepted": 1.0,
                                             "mtp_proposed": 4.0, "kv_cache_max_fraction": 0.5,
                                             "max_waiting_requests": 0}})
        calls, tasks = [], []
        for i in range(8):
            agent = "code_review" if i % 2 == 0 else "pentest"
            calls.append({"agent": agent, "t_start": t0 + 20 + i * 5, "duration_s": 2.0, "output_tokens": 100 + i,
                          "input_tokens": 1000, "cached_tokens": 400, "speed": 30.0 + i, "ttft_s": 0.2 + i / 100,
                          "error": None, "tool_call_count": 1})
            tasks.append({"agent": agent, "t_start": t0 + 15 + i * 5, "t_end": t0 + 60 + i * 5,
                          "error": "boom" if (level == 2 and i == 7) else None})
        (d / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
        (d / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    return root


# ---------------------------------------------------------------- canned Prometheus data (observation.query)
DAY = calendar.timegm((2024, 5, 14, 0, 0, 0))   # 2024-05-14 00:00 UTC (a past date: --end is capped at "now")
SCRAPE = 15


def cumulative_histogram(total, shares):
    """{le: cumulative count} from a request count and the cumulative share at each bound ("+Inf" gets the total)."""
    return {le: total * share for le, share in shares.items()}


# Cumulative shares of the per-request mean inter-token time (one bucket scheme, none at 30 tokens/s = 0.0333 s,
# so the share at the target is interpolated between the bounds 0.025 and 0.05).
TPOT_FAST = {"0.01": 0.10, "0.025": 0.93, "0.05": 0.99, "0.1": 1.0, "+Inf": 1.0}       # at least 93 %: holds
TPOT_SLOW = {"0.01": 0.0, "0.025": 0.20, "0.05": 0.60, "0.1": 1.0, "+Inf": 1.0}        # at most 60 %: does not hold
TPOT_STRADDLE = {"0.01": 0.10, "0.025": 0.75, "0.05": 0.97, "0.1": 1.0, "+Inf": 1.0}   # 75 to 97 %: undetermined
TTFT_OK = {"0.1": 0.20, "0.5": 0.96, "1": 1.0, "+Inf": 1.0}                # P95 at 0.5 s


def prom_window(w, requests=0, tpot=TPOT_FAST, **kw):
    """Canned data of one window of `w` seconds (the usable defaults: one target scraped all along, no restart).

    `requests` completed requests: tokens, histograms and cache counters follow from it unless given in `kw`
    (output_tokens, input_tokens, preemptions, failed, kv, waiting, running_avg, running_max, targets,
    always_up, at_start, boundary, at_end, resets, scrapes, accepted, proposed, hits, queries, ttft, agents,
    litellm_resets, litellm_up, litellm_count). failed=None: no http_requests_total series."""
    d = {"completed": requests, "output_tokens": requests * 400.0, "input_tokens": requests * 1000.0,
         "preemptions": 0.0, "failed": 0.0, "kv": 0.40, "waiting": 0.0, "running_avg": 0.0, "running_max": 0.0,
         "targets": 1, "always_up": 1, "at_start": 1, "boundary": 1, "at_end": 1, "resets": 0,
         "scrapes": w / SCRAPE, "accepted": requests * 100.0, "proposed": requests * 150.0,
         "hits": requests * 600.0, "queries": requests * 1000.0, "agents": None,
         "tpot": cumulative_histogram(requests, tpot), "ttft": cumulative_histogram(requests, TTFT_OK),
         "litellm_up": 1, "litellm_count": 1, "litellm_resets": 0}
    d.update(kw)
    return d


def agent_window(output=0.0, input_tokens=0.0, requests=0.0, ttft_seconds=0.0, latency_seconds=0.0,
                 ttft_shares=None, latency_shares=None, streamed=None):
    """LiteLLM data of one agent in a window; every request streamed unless `streamed` says otherwise."""
    ttft_shares = ttft_shares or {"0.1": 0.5, "0.5": 1.0, "+Inf": 1.0}
    latency_shares = latency_shares or {"1": 0.25, "5": 1.0, "+Inf": 1.0}
    return {"output_tokens": output, "input_tokens": input_tokens,
            "ttft": cumulative_histogram(requests, ttft_shares), "ttft_sum": ttft_seconds,
            "ttft_count": requests if streamed is None else streamed,
            "latency": cumulative_histogram(requests, latency_shares), "latency_sum": latency_seconds,
            "latency_count": requests}


WINDOW_3H = 3 * 3600


def observation_day():
    """Eight windows of 3 h over 2024-05-14 (UTC), one scenario per window: idle night, a few requests, then
    increasing load that holds up to 12.5 requests in flight, a slow window, a vLLM restart (discarded),
    preemptions with a full KV cache (LiteLLM restart too) and an undetermined verdict."""
    w = WINDOW_3H
    coder = lambda n: agent_window(output=n * 400.0, input_tokens=n * 1500.0, requests=n, ttft_seconds=n * 0.2,
                                   latency_seconds=n * 5.0)
    reviewer = lambda n: agent_window(output=n * 200.0, input_tokens=n * 6000.0, requests=n, ttft_seconds=n * 0.4,
                                      latency_seconds=n * 3.0, streamed=n - 100)
    return [
        prom_window(w, 0, kv=0.02),
        prom_window(w, 40, kv=0.05, running_avg=0.1, running_max=1.0),
        prom_window(w, 4000, running_avg=4.2, running_max=7.0, agents={"coder": coder(2400), "reviewer": reviewer(600)}),
        prom_window(w, 9000, running_avg=12.5, running_max=18.0, waiting=2.0,
                    agents={"coder": coder(5400), "reviewer": reviewer(1200), "": coder(100)}),
        prom_window(w, 12000, tpot=TPOT_SLOW, running_avg=20.0, running_max=26.0, kv=0.90,
                    agents={"coder": coder(7000), "reviewer": reviewer(1500)}),
        prom_window(w, 7000, resets=1, running_avg=8.0, running_max=11.0, agents={"coder": coder(4000)}),
        prom_window(w, 8000, preemptions=3.0, kv=0.97, running_avg=5.0, running_max=9.0, litellm_resets=1,
                    agents={"coder": coder(4500)}),
        prom_window(w, 6000, tpot=TPOT_STRADDLE, running_avg=15.0, running_max=19.0),
    ]


def prometheus_responder(first_end, w, windows):
    """Responder for helpers.PromStub: answers the PromQL queries of observation.query from `windows`, the
    prom_window() dicts of the successive windows ending at first_end, first_end + w, ... (a missing window
    reads as no data). Any other query is an error, which makes a changed query visible."""
    def window_at(t):
        i = round((t - first_end) / w)
        return windows[i] if 0 <= i < len(windows) else None

    def series(metric, t_values):
        values = [[t, str(v)] for t, v in t_values if v is not None]
        return [{"metric": metric, "values": values}] if values else []

    def respond(q, start, end, step):
        ts = []
        t = start
        while t <= end + 1e-6:
            ts.append(t)
            t += step

        def one(key):
            return series({}, [(t, None if window_at(t) is None else window_at(t)[key]) for t in ts])

        def grouped(key, label, extra=None):
            """One series per (agent[, le]) from windows[..]["agents"][agent][key]."""
            out = {}
            for t in ts:
                for agent, a in ((window_at(t) or {}).get("agents") or {}).items():
                    value = a.get(key)
                    items = value.items() if isinstance(value, dict) else [(None, value)]
                    for le, v in items:
                        out.setdefault((agent, le), []).append((t, v))
            return [r for (agent, le), vals in sorted(out.items(), key=str)
                    for r in series({label: agent, **({"le": le} if le is not None else {})}, vals)]

        def histogram(key):
            les = {le for t in ts for le in ((window_at(t) or {}).get(key) or {})}
            return [r for le in sorted(les, key=lambda x: float(x.replace("+Inf", "1e9")))
                    for r in series({"le": le}, [(t, (window_at(t) or {}).get(key, {}).get(le)) for t in ts])]

        litellm = "litellm" in q
        if q.startswith("(sum(resets("):
            return one("litellm_resets" if "litellm_" in q else "resets")
        if q.startswith("sum(count_over_time(up"):
            return one("scrapes")
        if q.startswith("count(http_requests_total"):
            return series({}, [(t, 1.0 if window_at(t) and window_at(t)["failed"] is not None else None)
                               for t in ts])
        m = re.match(r"sum by \(([^)]*)\) \(", q)
        if m:
            labels = [x.strip() for x in m.group(1).split(",")]
            name = re.search(r"(vllm:|litellm_)\w+", q).group(0)
            if labels == ["le"]:
                return histogram("ttft" if "first_token" in name else "tpot")
            suffix = "ttft" if "first_token" in name else "latency"
            if "le" in labels:
                return grouped(suffix, labels[0])
            if name.endswith("_sum") or name.endswith("_count"):
                return grouped(f"{suffix}_{name.rsplit('_', 1)[1]}", labels[0])
            return grouped("output_tokens" if "output_tokens" in name else "input_tokens", labels[0])
        if q.startswith("sum("):
            for key, name in (("output_tokens", "vllm:generation_tokens_total"), ("input_tokens", "vllm:prompt_tokens_total"),
                              ("completed", "vllm:request_success_total"), ("preemptions", "vllm:num_preemptions_total"),
                              ("hits", "vllm:prefix_cache_hits_total"), ("queries", "vllm:prefix_cache_queries_total"),
                              ("accepted", "vllm:spec_decode_num_accepted_tokens_total"),
                              ("proposed", "vllm:spec_decode_num_draft_tokens_total"),
                              ("failed", "http_requests_total{")):
                if name in q:
                    return one(key)
        for prefix, key in (("max_over_time(max(vllm:kv_cache_usage_perc", "kv"),
                            ("max_over_time(sum(vllm:num_requests_waiting", "waiting"),
                            ("avg_over_time(sum(vllm:num_requests_running", "running_avg"),
                            ("max_over_time(sum(vllm:num_requests_running", "running_max")):
            if q.startswith(prefix):
                return one(key)
        if q.startswith("min_over_time(sum(up"):
            return one("litellm_up" if litellm else "always_up")
        if q.startswith("max_over_time(count(up"):
            return one("litellm_count" if litellm else "targets")
        if q.startswith("sum(up") and "offset" in q:
            return one("at_start")
        if q.startswith("count(count_over_time(up"):
            return one("boundary" if "offset" in q else "at_end")
        raise AssertionError(f"unexpected PromQL: {q}")
    return respond
