"""Processes the AIPerf runs: one table row per level and the selected operating point.

err column: failed requests, plus requests rejected for too-long context (AIPerf removes them from its
metrics; a level that has any is rejected).
  --target : target interactivity per thread (tokens/s); the 10th percentile must reach it
            (interactivity of a request = 1 / mean inter-token delay, first token excluded)
  --gpu   : number of GPUs behind the target, all replicas included (throughput per GPU = total throughput / gpu)
  --require-metrics : reject a level whose server metrics (preemptions) were not read
  --ttft-max : reject a level whose P95 TTFT exceeds this value (ms); set according to your use (ASSUMPTION)
Writes <root>/table.csv and prints the selected point per configuration (runs <conf>-r1, -r2...)."""
import json
import pathlib

from report.common import (warn_missing_metrics, warn_rejected_lower_levels, warn_saturation,
                                  choose, configurations, write_csv, format_number, format_pct, group_rows,
                                  median_of, strip_repetition)

HELP = "AIPerf runs (sweep.sh, agentx.sh): <root>/<run>/c<N>/profile_export_aiperf.json"


def add_arguments(parser):
    """Add the AIPerf options (--require-metrics, --ttft-max) to the sub-command parser."""
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--ttft-max", type=float, default=None, help="maximum accepted P95 TTFT (ms)")


def read_stat(data, key, field_name):
    """Statistic `field_name` ("avg", "p95"...) of metric `key` in an AIPerf export `data`; None if absent."""
    v = data.get(key) or {}
    return v.get(field_name) if isinstance(v, dict) else None


def count_or_mean(data, key):
    """Value of `key`: its "avg" if it is a statistics dict, else the raw value; 0 if absent."""
    v = data.get(key)
    v = v.get("avg") if isinstance(v, dict) else v
    return v or 0


def read_server_metrics(run_dir):
    """vLLM server counters over the measurement phase (deltas computed by AIPerf), and maximum KV cache usage."""
    file = run_dir / "server_metrics_export.json"
    if not file.exists():
        return {}
    metrics = json.loads(file.read_text()).get("metrics", {})

    def total(name):
        v = metrics.get(name)
        if not v:
            return None
        return sum((s.get("stats") or {}).get("total") or 0 for s in v.get("series", []))

    def maximum(name):
        v = metrics.get(name)
        if not v:
            return None
        values = [(s.get("stats") or {}).get("max") for s in v.get("series", [])]
        values = [x for x in values if x is not None]
        return max(values) if values else None

    r = {"preemptions": total("vllm:num_preemptions"), "kv_cache_max": maximum("vllm:kv_cache_usage_perc"),
         "waiting_max": maximum("vllm:num_requests_waiting")}
    hits, requests = total("vllm:prefix_cache_hits"), total("vllm:prefix_cache_queries")
    r["prefix_cache"] = hits / requests if hits is not None and requests else None
    accepted, proposed = total("vllm:spec_decode_num_accepted_tokens"), total("vllm:spec_decode_num_draft_tokens")
    r["mtp_acceptance"] = accepted / proposed if accepted is not None and proposed else None
    return r


def error_count(data):
    """Number of failed requests of a run: error_request_count, else the sum of error_summary; 0 if none."""
    n = read_stat(data, "error_request_count", "avg")
    if n is None:
        n = 0 if not data.get("error_summary") else sum(e.get("count", 1) for e in data["error_summary"])
    return int(n)


def read_run(profile_file, gpus):
    """Table row (dict) of one profile_export_aiperf.json; throughput per GPU = total throughput / `gpus`.

    The run is the parent directory name, the concurrency comes from the c<N> directory; server metrics
    are read from server_metrics_export.json next to the profile."""
    data = json.loads(profile_file.read_text())
    run_dir = profile_file.parent
    run = run_dir.parent.name
    server = read_server_metrics(run_dir)
    throughput = read_stat(data, "output_token_throughput", "avg")
    return {
        "configuration": strip_repetition(run), "run": run, "concurrency": int(run_dir.name[1:]),
        "requests": read_stat(data, "request_count", "avg"), "failed_requests": error_count(data),
        "mean_output_tokens": read_stat(data, "output_sequence_length", "avg"),
        # Requests rejected for too-long context: AIPerf removes them from the metrics (AgentX scenario)
        "context_overflows": int(count_or_mean(data, "context_overflow_count")),
        # Outputs shorter than requested (vLLM reduces max_tokens to the room left in the context)
        "shortened_outputs": int(count_or_mean(data, "osl_mismatch_count")),
        "aiperf_run_valid": (data.get("metadata") or {}).get("submission_valid"),
        "invalid_reasons": ";".join((data.get("metadata") or {}).get("submission_invalid_reasons") or []),
        "total_output_throughput": throughput,
        "output_throughput_per_gpu": throughput / gpus if throughput else None,
        "interactivity_p10": read_stat(data, "output_token_throughput_per_user", "p10"),
        "interactivity_p50": read_stat(data, "output_token_throughput_per_user", "p50"),
        "ttft_p50_ms": read_stat(data, "time_to_first_token", "p50"),
        "ttft_p95_ms": read_stat(data, "time_to_first_token", "p95"),
        "ttft_p99_ms": read_stat(data, "time_to_first_token", "p99"),
        "preemptions": server.get("preemptions"), "kv_cache_max_fraction": server.get("kv_cache_max"),
        "prefix_cache": server.get("prefix_cache"), "mtp_acceptance": server.get("mtp_acceptance"),
        "max_waiting_requests": server.get("waiting_max")}


def warn_runs(rows, root):
    """Runs that AIPerf does not consider valid (AgentX scenario) and the context filter recorded by agentx.sh."""
    for run in sorted({row["run"] for row in rows}):
        run_rows = [row for row in rows if row["run"] == run]
        if any(row["aiperf_run_valid"] is False for row in run_rows):
            reasons = sorted({x for row in run_rows for x in row["invalid_reasons"].split(";") if x})
            print(f"WARNING: {run}: AIPerf marks the run submission_valid=false ({', '.join(reasons) or '?'}): "
                  "not comparable to the published AgentX results.")
        shortened = [str(row["concurrency"]) for row in run_rows if row["shortened_outputs"]]
        if shortened:
            print(f"WARNING: {run}: outputs shorter than requested at concurrencies {', '.join(shortened)} "
                  "(AIPerf osl_mismatch_count: context almost full, vLLM reduces max_tokens): throughput and "
                  "interactivity are based on truncated outputs; raise --max-model-len or lower MAX_CONTEXT.")
        meta = pathlib.Path(root) / run / "agentx.txt"
        if meta.exists():
            infos = dict(x.split("=", 1) for x in meta.read_text().splitlines() if "=" in x)
            if infos.get("max_context"):
                print(f"WARNING: {run}: traces filtered to {infos['max_context']} tokens (MAX_CONTEXT): subset only.")


def print_table(rows):
    """Print one line per run and concurrency (throughput in tokens/s per GPU, TTFT in ms;
    err = failed requests + context overflows)."""
    print(f"{'run':<18}{'conc':>5}{'tput/GPU':>11}{'inter P10':>11}{'TTFT P95 ms':>13}{'preempt':>9}{'err':>5}{'cache':>7}{'MTP':>6}{'output avg':>12}")
    for l in rows:
        print(f"{l['run']:<18}{l['concurrency']:>5}{format_number(l['output_throughput_per_gpu']):>11}{format_number(l['interactivity_p10']):>11}"
              f"{format_number(l['ttft_p95_ms'], 0):>13}{format_number(l['preemptions'], 0):>9}{l['failed_requests'] + l['context_overflows']:>5}"
              f"{format_pct(l['prefix_cache']):>7}{format_pct(l['mtp_acceptance']):>6}{format_number(l['mean_output_tokens'], 0):>12}")


def make_evaluator(args):
    """Build the `evaluate(rows) -> (accepted, other_reason)` function used by `choose`.

    A level is accepted if the median P10 interactivity reaches args.target and it is healthy: no error,
    no preemption, metrics present if --require-metrics, median P95 TTFT <= --ttft-max if given.
    other_reason is True when the level is rejected for a reason other than interactivity."""
    def evaluate(rows):
        errors = sum(l["failed_requests"] + l["context_overflows"] for l in rows)  # overflows count as errors
        missing = any(l["preemptions"] is None for l in rows)
        preemptions = sum(l["preemptions"] or 0 for l in rows)
        ttft = median_of(rows, "ttft_p95_ms")
        healthy = (errors == 0 and preemptions == 0 and not (missing and args.require_metrics)
                and (args.ttft_max is None or ttft <= args.ttft_max))
        return median_of(rows, "interactivity_p10") >= args.target and healthy, not healthy
    return evaluate


def print_selected_points(rows, args):
    """Median of the repetitions per (configuration, concurrency), then the level with the highest throughput
    that meets the target, with no failed request and no preemption."""
    groups = group_rows(rows, lambda l: l["configuration"], lambda l: l["concurrency"])
    print()
    for configuration in configurations(groups):
        choice = choose(groups, configuration, make_evaluator(args), prefer="throughput",
                        missing=lambda ls: any(l["preemptions"] is None for l in ls))
        if choice.selected is None:
            print(f"{configuration}: no level meets the target of {args.target} tokens/s without errors or preemptions.")
            continue
        print(f"{configuration}: selected point = concurrency {choice.selected}, {choice.throughput:.1f} tokens/s per GPU "
              f"(median of {choice.repetitions} repetition(s), from {choice.low:.1f} to {choice.high:.1f}).")
        if choice.selected == choice.levels[-1]:
            print(f"   WARNING: this is the highest level tested. The limit may not have been reached yet: "
                  f"add levels above {choice.selected}.")
        print_interpolation(groups, choice, args.target)
        warn_rejected_lower_levels(choice)
        warn_missing_metrics(choice)
        warn_saturation(choice.rows)


def print_interpolation(groups, choice, target):
    """Linear interpolation between the selected level and the next_rows one (which misses the target),
    in the P10 interactivity -> throughput per GPU plane: estimate of the throughput exactly at the target (INFERRED)."""
    later = [c for c in choice.levels if c > choice.selected]
    if not later or later[0] in choice.other_reason:  # only if the next_rows level fails on the target alone
        return
    next_rows = groups[(choice.configuration, later[0])]
    x1, y1 = median_of(choice.rows, "interactivity_p10"), choice.throughput
    x2, y2 = median_of(next_rows, "interactivity_p10"), median_of(next_rows, "output_throughput_per_gpu")
    if x1 > target > x2 and x1 != x2 and y2 > y1:
        yi = y1 + (y2 - y1) * (x1 - target) / (x1 - x2)
        print(f"   Estimate at exactly {target} tokens/s (interpolation between concurrencies {choice.selected} and "
              f"{later[0]}): {yi:.1f} tokens/s per GPU. To be confirmed with an intermediate level.")


def execute(args):
    """Entry point of `report aiperf`: read the runs under args.root, write table.csv, print the table and
    the selected points.

    Raises:
        SystemExit: if no profile_export_aiperf.json is found."""
    root = pathlib.Path(args.root)
    rows = [read_run(f, args.gpu) for f in sorted(root.glob("*/c*/profile_export_aiperf.json"))]
    rows.sort(key=lambda l: (l["run"], l["concurrency"]))
    warn_runs(rows, args.root)
    if not rows:
        raise SystemExit(f"No profile_export_aiperf.json under {args.root}/<run>/c<N>/")
    output = root / "table.csv"
    write_csv(output, rows)
    print(f"Table: {output}\n")
    print_table(rows)
    print_selected_points(rows, args)
