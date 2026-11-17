"""Processes the levels of the agent launcher: one row per level, plus the per-agent detail.

The server metrics (preemptions, server-side prefix cache, MTP acceptance, maximum KV cache) come from the
/metrics reading done by the launcher (level.json); they appear on the "all" row.
Window used: from the level start + warmup to the planned end of the level; only calls completed within this
window count (the tail, where concurrency drops, is excluded).
Selected level: the largest number of simultaneous tasks whose P10 interactivity (all agents) reaches
the target, with no error or preemption -- median of the repetitions (runs <conf>-rN).
--gpu is optional: without it, the GPU count is deduced from the server_config.json of each run
(TP x PP x DP of the fingerprint x number of vLLM servers).
Writes <root>/table_agents.csv."""
import json
import pathlib

from report.common import (GpuResolver, warn_missing_metrics, warn_rejected_lower_levels, warn_saturation,
                                  choose, configurations, write_csv, format_number, format_pct, group_rows,
                                  median_of, strip_repetition)

HELP = "levels of the agent launcher (agents_test.launcher): <root>/<run>/n<N>/level.json"

NO_SERVER = {"preemptions": None, "server_prefix_cache": None, "mtp_acceptance": None,
                "kv_cache_max_fraction": None, "max_waiting_requests": None}


def add_arguments(parser):
    """Add the --require-metrics option to the sub-command parser."""
    parser.add_argument("--require-metrics", action="store_true", help="reject a level without server metrics")


def percentile(xs, q):
    """Percentile `q` (0-1) of `xs` by linear interpolation, ignoring None; None if no value."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    i = int(k)
    fraction = k - i
    return xs[i] if i + 1 >= len(xs) else xs[i] * (1 - fraction) + xs[i + 1] * fraction


def read_jsonl(file):
    """Rows of a JSON-lines file; an empty list if the file does not exist."""
    return [json.loads(row) for row in open(file, encoding="utf-8") if row.strip()] if file.exists() else []


def summary(calls, t0, t1, n_gpu):
    """Indicators of the calls that completed within [t0, t1] (epoch seconds), over `n_gpu` GPUs.

    Call and error counts, output throughput per GPU (tokens/s), P10 and P50 interactivity (tokens/s, calls with
    at least 16 output tokens), TTFT P50 and P95 (s), cache rate, tool calls and maximum context (tokens)."""
    window = [x for x in calls if t0 <= x["t_start"] + x["duration_s"] <= t1]
    ok = [x for x in window if not x.get("error")]
    output = sum(x["output_tokens"] or 0 for x in ok)
    with_cache = [x for x in ok if x["input_tokens"] and x["cached_tokens"] is not None]
    speeds = [x["speed"] for x in ok if (x["output_tokens"] or 0) >= 16]
    return {
        "calls": len(window), "call_errors": len(window) - len(ok),
        "output_throughput_per_gpu": output / (t1 - t0) / n_gpu if t1 > t0 else None,
        "interactivity_p10": percentile(speeds, 0.10), "interactivity_p50": percentile(speeds, 0.50),
        "ttft_p50_s": percentile([x["ttft_s"] for x in ok], 0.50), "ttft_p95_s": percentile([x["ttft_s"] for x in ok], 0.95),
        "cache_rate": (sum(x["cached_tokens"] for x in with_cache) / sum(x["input_tokens"] for x in with_cache)) if with_cache else None,
        "tool_call_count": sum(x.get("tool_call_count") or 0 for x in ok),
        "max_context": max((x["input_tokens"] or 0) + (x["output_tokens"] or 0) for x in ok) if ok else None,
    }


def ratio_of(a, b):
    """a / b, or None if `a` is None or `b` is zero or None."""
    return (a / b) if a is not None and b else None


def level_row(run, level, agent, tasks, summarize_calls, server):
    """Table row of one agent (or "all") at one level: task counts and median task duration (minutes),
    followed by the call summary and the server metrics."""
    finished = [t for t in tasks if not t.get("error")]
    return {"run": run, "concurrent_tasks": level, "agent": agent,
            "tasks_finished": len(finished), "tasks_failed": len(tasks) - len(finished),
            "task_duration_p50_min": (percentile([t["t_end"] - t["t_start"] for t in finished], .5) or 0) / 60,
            **summarize_calls, **server}


def read_level(level_file, gpus):
    """Rows of one level.json: the "all" row, then one row per agent.

    Calls and tasks are read from the jsonl files next to it; the window runs from start + warmup to the
    planned end. The server metrics are attached to the "all" row only."""
    run_dir = level_file.parent
    meta = json.loads(level_file.read_text())
    t0 = meta["t_start"] + meta["warmup"]
    t1 = meta["t_end"]
    calls = read_jsonl(run_dir / "calls.jsonl")
    tasks = read_jsonl(run_dir / "tasks.jsonl")
    m = meta.get("metrics") or {}
    server = {"preemptions": m.get("preemptions"),
               "server_prefix_cache": ratio_of(m.get("prefix_cache_hits"), m.get("prefix_cache_queries")),
               "mtp_acceptance": ratio_of(m.get("mtp_accepted"), m.get("mtp_proposed")),
               "kv_cache_max_fraction": m.get("kv_cache_max_fraction"),
               "max_waiting_requests": m.get("max_waiting_requests")}
    run, level = run_dir.parent.name, meta["concurrent_tasks"]
    rows = [level_row(run, level, "all", tasks, {**summary(calls, t0, t1, gpus), "gpu": gpus}, server)]
    for agent in meta["agents"]:
        rows.append(level_row(run, level, agent, [t for t in tasks if t["agent"] == agent],
                                   summary([x for x in calls if x["agent"] == agent], t0, t1, gpus), NO_SERVER))
    return rows


def print_table(rows):
    """Print one line per level and agent (throughput in tokens/s per GPU, TTFT P95 in s, duration in minutes,
    err = failed tasks + failed calls)."""
    print(f"{'run':<12}{'N':>3} {'agent':<11}{'tasks':>7}{'err':>5}{'tput/GPU':>11}{'inter P10':>10}{'TTFT P95 s':>11}"
          f"{'cache':>7}{'dur min':>10}{'preempt':>9}{'MTP':>6}{'KV max':>8}")
    for l in rows:
        print(f"{l['run']:<12}{l['concurrent_tasks']:>3} {l['agent']:<11}{l['tasks_finished']:>7}{l['tasks_failed'] + l['call_errors']:>5}"
              f"{format_number(l['output_throughput_per_gpu']):>11}{format_number(l['interactivity_p10']):>10}{format_number(l['ttft_p95_s'], 2):>11}"
              f"{format_pct(l['cache_rate']):>7}{format_number(l['task_duration_p50_min']):>10}{format_number(l['preemptions'], 0):>9}"
              f"{format_pct(l['mtp_acceptance']):>6}{format_pct(l['kv_cache_max_fraction']):>8}")


def make_evaluator(args):
    """Build the `evaluate(rows) -> (accepted, other_reason)` function used by `choose`.

    A level is accepted if the median P10 interactivity reaches args.target, with no failed task or call,
    no preemption, and metrics present if --require-metrics. other_reason is always False."""
    def evaluate(rows):
        missing = any(l["preemptions"] is None for l in rows)
        ok = (median_of(rows, "interactivity_p10") >= args.target
              and sum(l["tasks_failed"] + l["call_errors"] for l in rows) == 0
              and sum(l["preemptions"] or 0 for l in rows) == 0
              and not (missing and args.require_metrics))
        return ok, False
    return evaluate


def print_selected_points(rows, args):
    """Print, per configuration, the largest number of simultaneous tasks that reaches the target (from the
    "all" rows), followed by the warnings."""
    groups = group_rows([l for l in rows if l["agent"] == "all"],
                             lambda l: strip_repetition(l["run"]), lambda l: l["concurrent_tasks"])
    print()
    for configuration in configurations(groups):
        choice = choose(groups, configuration, make_evaluator(args), prefer="level",
                        missing=lambda ls: any(l["preemptions"] is None for l in ls))
        if choice.selected is None:
            print(f"{configuration}: no level reaches {args.target} tokens/s per thread (P10) without errors or preemptions.")
            continue
        print(f"{configuration}: selected level = {choice.selected} simultaneous task(s), {choice.throughput:.1f} tokens/s per GPU "
              f"(median of {choice.repetitions} repetition(s), from {choice.low:.1f} to {choice.high:.1f}).")
        if choice.selected == choice.levels[-1]:
            print(f"   WARNING: this is the highest level tested; add levels above {choice.selected}.")
        warn_rejected_lower_levels(choice)
        warn_missing_metrics(choice)
        warn_saturation(choice.rows)


def early_stop_warnings(files):
    """Warnings for the runs in autonomous mode whose agents stopped far short of their steps.

    In autonomous mode the model decides whether to call a tool, and an agent stops at the first call without one.
    For each run and agent, the mean number of calls per finished task is compared with the number the profile
    makes when it goes through all its steps; under half of it gives a warning (the contexts are then much shorter
    than the profile). Runs in scripted mode, where the tool is forced, are not concerned."""
    from agents_test.profiles import PROFILES, calls_per_task
    seen, messages = {}, []
    for f in files:
        meta = json.loads(f.read_text())
        if meta.get("mode") != "autonomous":
            continue
        calls = read_jsonl(f.parent / "calls.jsonl")
        for task in read_jsonl(f.parent / "tasks.jsonl"):
            if task.get("error") or task["agent"] not in PROFILES:
                continue
            n = sum(1 for c in calls if c.get("task") == task["task"])
            total = seen.setdefault((f.parent.parent.name, task["agent"]), [0, 0])
            total[0] += n
            total[1] += 1
    for (run, agent), (n_calls, n_tasks) in sorted(seen.items()):
        expected = calls_per_task(agent)
        if expected > 2 and n_calls / n_tasks < expected / 2:
            messages.append(f"WARNING: {run}: autonomous mode, {agent} made {n_calls / n_tasks:.0f} calls per task instead of "
                            f"{expected}: the model stopped calling tools (output cap too small for a tool call at this "
                            "scale, or the server does not parse tool calls). Contexts are shorter than the profile; "
                            "use scripted mode for a stress test.")
    return messages


def execute(args):
    """Entry point of `report agents`: read the levels under args.root, write table_agents.csv, print the table
    and the selected levels.

    Raises:
        SystemExit: if no level.json is found, or if the GPU count of a run is unknown (no --gpu and nothing
        to deduce it from)."""
    root = pathlib.Path(args.root)
    files = sorted(root.glob("*/n*/level.json"), key=lambda f: (f.parent.parent.name, int(f.parent.name[1:])))
    resolver = GpuResolver(args.gpu)
    rows = []
    for f in files:
        gpus = resolver.for_run(f.parent.parent)
        if gpus is not None:
            rows.extend(read_level(f, gpus))
    resolver.finish(rows)
    if not rows:
        raise SystemExit(f"No level under {args.root}/<run>/n<N>/")
    write_csv(root / "table_agents.csv", rows)
    print_table(rows)
    for message in early_stop_warnings(files):
        print(f"   {message}")
    print_selected_points(rows, args)
