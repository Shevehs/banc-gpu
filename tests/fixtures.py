"""Builds small deterministic result trees for the three report types."""
import json
import pathlib


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


def _histograms(ttft, tpot):
    return {"ttft": ttft, "tpot": tpot}


def observation_tree(root):
    """Label prod: levels 2 and 4 (two repetitions each), level 4 is slower; one window with a server restart."""
    root = pathlib.Path(root)
    fast = {"0.01": 10.0, "0.02": 95.0, "0.05": 100.0, "+Inf": 100.0}
    slow = {"0.01": 0.0, "0.02": 10.0, "0.05": 60.0, "+Inf": 100.0}
    ttft = {"0.1": 20.0, "0.5": 90.0, "1": 98.0, "+Inf": 100.0}
    base = 1_700_000_000.0
    specs = [("prod-n2-r1", 2, fast, 0, False), ("prod-n2-r2", 2, fast, 0, False),
             ("prod-n4-r1", 4, slow, 0, False), ("prod-n4-r2", 4, fast, 5, False),
             ("prod-n8-r1", 8, fast, 0, True)]
    for i, (label, level, tpot, preempt, reset) in enumerate(specs):
        start = base + i * 4000
        obs = {"label": label, "level": level, "start": start, "end": start + 900, "urls": ["http://v:8000/metrics"],
               "counters": {"output_tokens": 90000.0, "input_tokens": 200000.0, "completed_requests": 100.0,
                             "preemptions": float(preempt), "prefix_cache_hits": 50.0,
                             "prefix_cache_queries": 100.0, "mtp_accepted": None, "mtp_proposed": None,
                             "failed_requests": 0.0},
               "histograms": _histograms(ttft, tpot),
               "gauges": {"kv_max": 0.5, "waiting_max": 0.0, "running_max": 4.0},
               "counter_reset": reset}
        if level == 2:
            obs["agents_by"] = "metadata_agent"
            obs["agents"] = {"coder": {"output_tokens": 100.0, "input_tokens": 500.0, "ttft": {"0.5": 5.0, "+Inf": 10.0},
                                       "ttft_sum": 2.0, "ttft_count": 10.0, "latency": {"1": 8.0, "+Inf": 10.0},
                                       "latency_sum": 9.0, "latency_count": 10.0}}
        _w(root / label / f"observation-{i:02d}.json", obs)
    return root
