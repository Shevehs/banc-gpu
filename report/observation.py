"""Processes the observation windows: one row per window, then, if levels were noted,
the highest level that holds the target.

  --target : target interactivity per request (tokens/s). A window "holds" if at least 90 % of the completed
            requests reach it (same threshold as the P10 interactivity of the stress tests, measured here by vLLM),
            with no preemption and no failed request (and a P95 TTFT <= --ttft-max if given). "Undetermined":
            the target falls in a histogram bucket that straddles 90 % (bounds to add, README section 8.1).
  --gpu   : number of GPUs behind the metrics URLs, all replicas included
  --view period : per configuration, groups the windows by hour (--period hour) or by 6-hour range
                  (night 0-6 h, morning 6-12 h, afternoon 12-18 h, evening 18-24 h; --period range), local time
  --view agents : tokens, requests, TTFT and latency per agent, read from the LiteLLM metrics
                  (observer --litellm); all the windows of a configuration added together
Grouping: label without its -n<level> and -r<repetition> suffixes (e.g. prod-n4-r1 -> prod).
Writes <root>/table_observation.csv."""
import json
import pathlib
import re
import statistics
import time

from report.common import (warn_saturation, choose, configurations, write_csv, format_number, format_pct,
                                  group_rows)
from observation.analysis import indicators, agent_indicators, verdict

HELP = "observation windows (observation.observer): <root>/<label>/observation-*.json"

RANGES = [(0, "night (0-6 h)"), (6, "morning (6-12 h)"), (12, "afternoon (12-18 h)"), (18, "evening (18-24 h)")]


def add_arguments(parser):
    """Add --ttft-max, --view and --period to the sub-command parser."""
    parser.add_argument("--ttft-max", type=float, default=None, help="maximum accepted estimated P95 TTFT (ms)")
    parser.add_argument("--view", choices=["windows", "period", "agents"], default="windows")
    parser.add_argument("--period", choices=["hour", "range"], default="range")


def period_label(t, mode):
    """Label of the period that contains the epoch time `t` (local time): "HH h" if `mode` is "hour",
    else the name of the 6-hour range."""
    hour = time.localtime(t).tm_hour
    return f"{hour:02d} h" if mode == "hour" else [name for start, name in RANGES if hour >= start][-1]


def sum_histograms(histograms):
    """Sum several {le: count} histograms bound by bound."""
    total_sum = {}
    for h in histograms:
        for le, n in h.items():
            total_sum[le] = total_sum.get(le, 0.0) + n
    return total_sum


def merge(windows):
    """Adds successive windows (counters, histograms; gauges: maximum) into one pseudo-window."""
    counters = {}
    for f in windows:
        for k, v in f["counters"].items():
            if k == "failed_requests":  # None = no /v1 series created yet: nothing to add
                counters[k] = counters.get(k) if v is None else (counters.get(k) or 0.0) + v
            else:  # other counter unreadable in a window: total unknown
                counters[k] = None if v is None or (k in counters and counters[k] is None) else counters.get(k, 0.0) + v
    return {"start": 0.0, "end": sum(f["end"] - f["start"] for f in windows), "counters": counters,
            "histograms": {k: sum_histograms([f["histograms"][k] for f in windows]) for k in windows[0]["histograms"]},
            "gauges": {k: max((f["gauges"][k] for f in windows if f["gauges"][k] is not None), default=None)
                       for k in windows[0]["gauges"]}}


def overlaps(windows):
    """True if two windows overlap in time (same server observed twice, or another server under the same
    label): their sum would count requests twice, or mix servers."""
    ordered = sorted(windows, key=lambda x: x["start"])
    return any(b["start"] < a["end"] - 1 for a, b in zip(ordered, ordered[1:]))


def share_text(r):
    """Share of requests at the target as a percentage; "*" if interpolated, an em dash if unknown."""
    if r["share_of_requests_at_target"] is None:
        return "—"
    return format_pct(r["share_of_requests_at_target"]) + ("" if r["share_exact"] else "*")


def headers():
    """Header of the indicator columns, shared by the windows and period views."""
    return (f"{'requests':>9}{'err':>5}{'tput/GPU':>11}{'at target':>11}{'inter P10≥':>11}"
            f"{'TTFT P95≤ ms':>13}{'preempt':>8}{'cache':>7}{'MTP':>6}{'KV max':>8}  verdict")


def format_row(r):
    """Indicator columns of row `r` (one window or merged period), aligned with `headers`."""
    return (f"{format_number(r['completed_requests'], 0):>9}{format_number(r['failed_requests'], 0):>5}{format_number(r['output_throughput_per_gpu']):>11}"
            f"{share_text(r):>11}{('<' if r['interactivity_bound'] else '') + format_number(r['estimated_interactivity_p10']):>11}"
            f"{('>' if r['ttft_p95_beyond'] else '') + format_number(r['estimated_ttft_p95_ms'], 0):>13}{format_number(r['preemptions'], 0):>8}"
            f"{format_pct(r['prefix_cache']):>7}{format_pct(r['mtp_acceptance']):>6}{format_pct(r['kv_cache_max_fraction']):>8}"
            f"  {r.get('verdict') or verdict(r['holds'])}")


def period_view(valid, args):
    """Print, per configuration and period (args.period), the indicators of the merged windows.

    `valid` is a list of (window, configuration). A period whose windows overlap or come from different
    servers gets the verdict "mixed" and a warning."""
    groups = {}
    for window, configuration in valid:
        groups.setdefault((configuration, period_label(window["start"], args.period)), []).append(window)
    print(f"{'configuration':<16}{'period':<22}{'windows':>9}" + headers())
    for (configuration, period_range) in sorted(groups):
        windows = groups[(configuration, period_range)]
        r = indicators(merge(windows), args.target, args.gpu, args.ttft_max)
        mixed = overlaps(windows) or len({tuple(sorted(f.get("urls") or [])) for f in windows}) > 1
        print(f"{configuration[:16]:<16}{period_range:<22}{len(windows):>9}" + (format_row(r) if not mixed else
              format_row({**r, "holds": None}).replace("undetermined", "mixed")))
        if mixed:
            print(f"   WARNING: {configuration}, {period_range}: windows that overlap or different servers under the same label: "
                  "no verdict (sum distorted); give one label per server or per set of replicas.")
    print("\nThroughput/GPU: mean over the period (output tokens / observed time / GPU). For the peak of a period: "
          "look at its windows one by one (--view windows).")


def agents_view(valid):
    """Print the per-agent view (LiteLLM metrics) per configuration, all its windows added together;
    a message if no window carries per-agent data."""
    per_config = {}
    for window, configuration in valid:
        if window.get("agents") is not None:
            per_config.setdefault(configuration, []).append(window)
    if not per_config:
        print("No per-agent data: run observer with --litellm (LiteLLM prometheus callback).")
        return
    for configuration, windows in sorted(per_config.items()):
        totals = {}
        for f in windows:
            for agent, a in f["agents"].items():
                t = totals.setdefault(agent, {k: ({} if isinstance(v, dict) else 0.0) for k, v in a.items()})
                for k, v in a.items():
                    t[k] = sum_histograms([t[k], v]) if isinstance(v, dict) else t[k] + v
        per = windows[0].get("agents_by", "api_key_alias")
        print(f"{configuration} ({len(windows)} window(s), agents distinguished by {per})")
        print(f"  {'agent':<24}{'requests':>9}{'output share':>13}{'input/req':>11}{'output/req':>11}"
              f"{'TTFT P50≤ s':>12}{'TTFT P95≤ s':>12}{'latency P95≤ s':>15}{'avg speed':>12}")
        for agent, r in agent_indicators(totals).items():
            print(f"  {agent[:24]:<24}{format_number(r['requests'], 0):>9}{format_pct(r['output_token_share']):>13}"
                  f"{format_number(r['input_tokens_per_request'], 0):>11}{format_number(r['output_tokens_per_request'], 0):>11}"
                  f"{format_number(r['ttft_p50_s'], 2):>12}{('>' if r['ttft_p95_beyond'] else '') + format_number(r['ttft_p95_s'], 2):>12}"
                  f"{('>' if r['latency_p95_beyond'] else '') + format_number(r['latency_p95_s'], 2):>15}{format_number(r['mean_speed']):>12}")
        if "None" in totals or "" in totals:
            print("   \"None\" or empty: requests without an agent identity (no virtual key, or no metadata.agent).")
    print("\nPer-agent view read from LiteLLM: TTFT only for streamed requests; average speed = output tokens "
          "/ (latency - TTFT), a mean and not a percentile, proxy time included, empty if some of the "
          "requests are not streamed. Bounds <=: buckets of the LiteLLM histograms.")


def load_windows(args):
    """Returns (rows, alerts, valid) from all the observation-*.json files in the results directory."""
    rows, alerts, valid = [], [], []
    for file in sorted(pathlib.Path(args.root).glob("*/observation*.json")):
        o = json.loads(file.read_text())
        label = o["label"]
        configuration = re.sub(r"(-n\d+)?(-r\d+)?$", "", label)
        start = time.strftime("%Y-%m-%d %H:%M", time.localtime(o["start"]))
        if o.get("incomplete"):
            alerts.append(f"{label} {start}: incomplete window ({o.get('error')}): discarded.")
            continue
        r = indicators(o, args.target, args.gpu, args.ttft_max)
        discarded = bool(o.get("counter_reset"))
        if discarded:
            alerts.append(f"{label} {start}: server restarted during the window: window discarded.")
            r = {k: (None if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items()}
            r["holds"] = False
        else:
            valid.append((o, configuration))
        if o.get("failed_reads"):
            alerts.append(f"{label} {start}: {o['failed_reads']} failed /metrics read(s): incomplete gauges.")
        if not discarded and (r["completed_requests"] or 0) < 50:
            alerts.append(f"{label} {start}: fewer than 50 completed requests: percentiles unreliable.")
        if o.get("agents_error"):
            alerts.append(f"{label} {start}: LiteLLM metrics not read ({o['agents_error']}).")
        rows.append({"configuration": configuration, "window": label, "start": start, "level": o.get("level"),
                       "duration_s": round(o["end"] - o["start"]), **r,
                       "verdict": "discarded" if discarded else verdict(r["holds"]), "discarded": discarded})
    return rows, alerts, valid


def make_evaluator(args):
    """Build the `evaluate(rows) -> (accepted, other_reason)` function used by `choose`.

    A level holds if the median lower bound of the share of requests at the target is at least 90 %, with no
    discarded window, preemption or failed request, and a median estimated P95 TTFT <= --ttft-max if given.
    other_reason is always False."""
    def evaluate(rows):
        shares = [l["share_min"] for l in rows if l["share_min"] is not None]
        ttft95 = [l["estimated_ttft_p95_ms"] for l in rows if l["estimated_ttft_p95_ms"] is not None]
        ok = bool(shares and statistics.median(shares) >= 0.90 and not any(l["discarded"] for l in rows)
                  and sum(l["preemptions"] or 0 for l in rows) == 0 and sum(l["failed_requests"] or 0 for l in rows) == 0
                  and (args.ttft_max is None or (ttft95 and statistics.median(ttft95) <= args.ttft_max)))
        return ok, False
    return evaluate


def windows_view(rows, args):
    """Print one line per window, then, per configuration, the highest level that holds the target
    (windows with a noted level only)."""
    print(f"Table: {pathlib.Path(args.root) / 'table_observation.csv'}\n")
    print(f"{'window':<20}{'start':>17}{'level':>7}{'dur s':>8}" + headers())
    for l in rows:
        print(f"{l['window'][:20]:<20}{l['start']:>17}{format_number(l['level'], 0):>7}{l['duration_s']:>8}" + format_row(l))
    if any(l["share_exact"] is False for l in rows):
        print(f"\n* interpolated share: no histogram bound at {1000 / args.target:.1f} ms; start vLLM with "
              "--custom-histogram-buckets (README, section 8.1). \"undetermined\": the target falls in a bucket that straddles 90 %.")
    print("≥ and ≤: bounds read from the histogram buckets (< or >: beyond the last bound). "
          "err: /v1 requests with 4xx or 5xx seen by vLLM (\"—\" if not exposed).")
    # Highest level that holds, per configuration: median of the repetitions for the share at target and
    # TTFT; no preemption or error; no discarded window.
    print()
    groups = group_rows([l for l in rows if l["level"] is not None], lambda l: l["configuration"], lambda l: l["level"])
    for configuration in configurations(groups):
        choice = choose(groups, configuration, make_evaluator(args), prefer="level")
        if choice.selected is None:
            print(f"{configuration}: no level holds {args.target:g} tokens/s for 90 % of requests without preemptions "
                  "or errors (or the result is undetermined: see the verdicts).")
            continue
        print(f"{configuration}: highest level that holds = {choice.selected} ({choice.throughput:.1f} tokens/s per GPU, "
              f"{choice.repetitions} window(s)).")
        if choice.selected == choice.levels[-1]:
            print("   WARNING: this is the highest level observed; the limit has not been reached.")
        warn_saturation(choice.rows, "level")


def execute(args):
    """Entry point of `report observation`: write table_observation.csv, print the chosen view, then the alerts.

    Raises:
        SystemExit: if no usable window is found."""
    rows, alerts, valid = load_windows(args)
    if not rows:
        for a in alerts:
            print("   WARNING: " + a)
        raise SystemExit(f"No usable window under {args.root}/<label>/")
    write_csv(pathlib.Path(args.root) / "table_observation.csv", rows)
    if args.view == "period":
        period_view(valid, args)
    elif args.view == "agents":
        agents_view(valid)
    else:
        windows_view(rows, args)
    for a in alerts:
        print("   WARNING: " + a)
