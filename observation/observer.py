"""Observation mode: measures real usage without generating any load.

Your agents (or your users) call vLLM directly or through LiteLLM; this script reads /metrics from each
vLLM server at the start and end of every window (and the gauges every --step seconds), then records the
deltas: tokens, requests, preemptions, cache, MTP, and the histograms of TTFT and of the time per token
of each request.

  --duration  : total observation time (s)
  --interval  : splits the observation into successive windows of this length (per-period view);
                without --interval, a single window
  --litellm   : LiteLLM /metrics URL ("prometheus" callback) for the per-agent view;
                --by : label that tells agents apart (api_key_alias by default, or metadata_<field_name>)

Example (one day in 15-minute windows, with the per-agent view):
  python -m observation.observer --metrics "http://10.0.0.5:8000/metrics" --duration 86400 --interval 900 \
      --litellm http://10.0.0.9:4000/metrics/ --label prod-r1 --gpu 8 --target 30
Results: <output>/<label>/observation-<YYYYMMDD-HHMMSS>.json (one per window) and server_config.json.
--url and --model (optional): also record the server fingerprint (parallelism) through a one-token completion,
sent BEFORE the first window. Key: BENCH_API_KEY for the target (with --url);
BENCH_LITELLM_KEY (otherwise the same key) for LiteLLM's /metrics/, protected by default; no key is ever sent to vLLM /metrics.
For exact interactivity, start vLLM with --custom-histogram-buckets (see the README).
Standard library only."""
import argparse
import json
import os
import re
import shutil
import signal
import sys
import time

from tools import settings
from observation.analysis import (COUNTERS, HISTOGRAMS, delta_litellm, indicators, fetch_series, summarize,
                     summarize_litellm, verdict)

NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
BY_OK = re.compile(r"^(api_key_alias|team_alias|user|end_user|metadata_[A-Za-z0-9_]{1,40})$")


class StopState(BaseException):
    """Stop requested (SIGTERM): a BaseException, so that an `except Exception` does not swallow it."""


ZONE, REQUESTED = [False], [False]  # final read in progress; stop requested during it


def snapshot(urls):
    """Sum of the series from all vLLM URLs, plus one marker per URL (output tokens, completed requests)
    to detect a restart; raises an exception if a URL does not respond."""
    all_series, samples = [], {}
    for u in urls:
        series = fetch_series(u)
        c, _, _ = summarize(series)
        samples[u] = (c.get(COUNTERS["output_tokens"], 0.0), c.get(COUNTERS["completed_requests"], 0.0))
        all_series += series
    return summarize(all_series) + (samples,)


def persistent_snapshot(urls, runs=4, pause_s=3.0):
    """snapshot() with a few retries: an outage of a few seconds does not lose the window."""
    for i in range(runs):
        try:
            return snapshot(urls)
        except Exception:  # noqa: BLE001
            if i == runs - 1:
                raise
            time.sleep(pause_s)


def counter_dropped(before, after):
    """True if a marker went down on some URL (server restarted)."""
    return any(u in before and (after[u][0] < before[u][0] or after[u][1] < before[u][1]) for u in after)


def observe_window(p, urls, key, start, end_planned, litellm0):
    """Observes one window; returns (obs, final LiteLLM summary, stop)."""
    c0, h0, _, t_prev = persistent_snapshot(urls)
    kv_max = waiting_max = running_max = None; reads = failures = 0; stop = False; restarted = False
    try:
        while time.time() < end_planned:
            time.sleep(max(0.0, min(p.step, end_planned - time.time())))
            try:
                _, _, j, t = snapshot(urls)
                reads += 1
                restarted |= counter_dropped(t_prev, t); t_prev = t
                if j["kv"]:
                    kv_max = max(j["kv"]) if kv_max is None else max(kv_max, max(j["kv"]))
                if j["waiting"]:
                    v = sum(j["waiting"]); waiting_max = v if waiting_max is None else max(waiting_max, v)
                if j["running"]:
                    v = sum(j["running"]); running_max = v if running_max is None else max(running_max, v)
            except Exception:  # noqa: BLE001 — a missed read does not stop the window
                failures += 1
    except (KeyboardInterrupt, StopState):
        stop = True
    ZONE[0] = True  # SIGTERM during the final reads (vLLM then LiteLLM): noted, applied afterwards
    try:
        c1, h1, _, t1 = persistent_snapshot(urls)
        litellm_end, litellm_err = None, None
        if p.litellm:
            try:
                litellm_end = summarize_litellm(fetch_series(p.litellm, prefix="litellm_", key=key), p.by)
            except Exception as e:  # noqa: BLE001
                litellm_err = f"{type(e).__name__}: {e}"[:200]
    finally:
        ZONE[0] = False
    end = time.time()
    counters, counter_reset = {}, restarted or counter_dropped(t_prev, t1)
    for key_c, name in COUNTERS.items():
        if name in c0 and name in c1:
            counters[key_c] = c1[name] - c0[name]; counter_reset |= counters[key_c] < 0
        elif name in c1 and key_c == "failed_requests":
            counters[key_c] = c1[name]  # series that appeared during the window: started at 0
        else:
            counters[key_c] = None
    histograms = {}
    for k in HISTOGRAMS:
        histograms[k] = {le: h1[k][le] - h0[k].get(le, 0.0) for le in h1[k]}
        counter_reset |= any(v < 0 for v in histograms[k].values())
    obs = {"label": p.label, "level": p.level, "start": start, "end": end, "urls": urls,
           "counters": counters, "histograms": histograms,
           "gauges": {"kv_max": kv_max, "waiting_max": waiting_max, "running_max": running_max},
           "reads": reads, "failed_reads": failures, "counter_reset": counter_reset, "interrupted": stop}
    stop = stop or REQUESTED[0]
    obs["interrupted"] = stop
    litellm1 = litellm_end
    if p.litellm:
        if litellm_err:
            obs["agents_error"] = litellm_err
        elif litellm0 is not None:
            try:
                obs["agents"] = delta_litellm(litellm0, litellm1); obs["agents_by"] = p.by
            except Exception as e:  # noqa: BLE001
                obs["agents_error"] = f"{type(e).__name__}: {e}"[:200]
    return obs, litellm1, stop


def main():
    """Command-line entry point: observe /metrics in successive windows of --interval seconds for --duration.

    Writes observation-*.json under <output>/<label>/ and prints one summary line per window.

    Raises:
        SystemExit: on invalid options or an existing label without --overwrite."""
    settings.warn_ignored_settings()
    a = argparse.ArgumentParser()
    a.add_argument("--metrics", required=True, help="vLLM /metrics URL(s), separated by spaces")
    a.add_argument("--duration", type=float, default=900, help="total observation time in seconds")
    a.add_argument("--interval", type=float, default=None, help="seconds per window (per-period view)")
    a.add_argument("--label", required=True)
    a.add_argument("--level", type=int, default=None, help="number of active agents or sessions (noted by hand)")
    a.add_argument("--litellm", default=None, help="LiteLLM /metrics URL, for the per-agent view")
    a.add_argument("--by", default="api_key_alias", help="LiteLLM label that tells agents apart")
    a.add_argument("--output", default="results-observation")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--step", type=float, default=5.0, help="seconds between two gauge reads")
    a.add_argument("--url", default=None, help="target (vLLM or LiteLLM) for the configuration; default: first metrics URL")
    a.add_argument("--model", default=None, help="if given, one-token completion before the first window (fingerprint)")
    a.add_argument("--target", type=float, default=None, help="for the summary printed after each window")
    a.add_argument("--gpu", type=int, default=None, help="for the summary printed after each window")
    p = a.parse_args()
    if not NAME_OK.match(p.label) or ".." in p.label:
        raise SystemExit("--label: letters, digits, dot, hyphen, underscore; no / and no ..")
    if not BY_OK.match(p.by):
        raise SystemExit("--by: api_key_alias, team_alias, user, end_user or metadata_<field_name>")
    interval = p.interval or p.duration
    if interval < 60 or p.duration < interval:
        raise SystemExit("--interval (or --duration): at least 60 s, and --duration >= --interval")
    urls = p.metrics.split()
    out_dir = os.path.join(p.output, p.label)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not p.overwrite:
            raise SystemExit(f"{out_dir} already exists: use another label, or add --overwrite.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    key = settings.api_key()
    # Key for LiteLLM's /metrics/: BENCH_LITELLM_KEY if set (target = vLLM), otherwise the target's key.
    key_ll = os.environ.get("BENCH_LITELLM_KEY") or key
    try:
        from tools.server_config import collect, summary
        base = (p.url or urls[0]).rstrip("/")
        base = base[: -len("/metrics")] if base.endswith("/metrics") else base
        cfg = collect(base, urls, key if p.url else None, p.model)  # without --url: no key sent to vLLM
        with open(os.path.join(out_dir, "server_config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1, ensure_ascii=False)
        print(summary(cfg), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(server configuration not recorded: {e})", file=sys.stderr)

    def _on_signal(*_):
        if ZONE[0]:
            REQUESTED[0] = True
            return
        raise StopState()
    signal.signal(signal.SIGTERM, _on_signal)  # a stop from the interface ends the current window cleanly
    litellm = None
    if p.litellm:
        try:
            litellm = summarize_litellm(fetch_series(p.litellm, prefix="litellm_", key=key_ll), p.by)
            print(f"[litellm] {len(litellm)} value(s) of {p.by} seen in {p.litellm}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[litellm] cannot read ({type(e).__name__}: {e}): no per-agent view", flush=True)
    t_end = time.time() + p.duration
    print(f"== {p.label}: observing for {p.duration:.0f} s in window(s) of {interval:.0f} s"
          + (f", level {p.level}" if p.level is not None else "") + " (Ctrl-C to stop) ==", flush=True)
    n_windows = 0
    while True:
        if n_windows and t_end - time.time() < min(60.0, interval) / 2:  # no leftover window that is too short
            break
        n_windows += 1
        start = time.time()
        end_planned = min(start + interval, t_end)
        try:
            obs, litellm_end, stop = observe_window(p, urls, key_ll, start, end_planned, litellm)
        except StopState:
            print("Stop requested.", flush=True)
            break
        except Exception as e:  # noqa: BLE001 — /metrics unreachable at the start or end: incomplete window
            obs = {"label": p.label, "level": p.level, "start": start, "end": time.time(), "urls": urls,
                   "incomplete": True, "error": f"{type(e).__name__}: {e}"[:200]}
            print(f"   window {time.strftime('%H:%M:%S', time.localtime(start))} incomplete: /metrics unreachable "
                  f"({obs['error']}); continuing.", flush=True)
            with open(os.path.join(out_dir, "observation-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(start))
                                   + ".json"), "w", encoding="utf-8") as fh:
                json.dump(obs, fh, indent=1)
            try:
                time.sleep(max(0.0, end_planned - time.time()))
            except (KeyboardInterrupt, StopState):
                break
            litellm = None  # LiteLLM starting point lost: the per-agent view resumes at the next window
            continue
        litellm = litellm_end if litellm_end is not None else litellm
        name = "observation-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(start)) + ".json"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obs, fh, indent=1)
        msg = f"   window {time.strftime('%H:%M:%S', time.localtime(start))} ({obs['end'] - start:.0f} s) -> {name}"
        if p.target and p.gpu and obs["counter_reset"]:
            msg += "; discarded (server restarted)"
        elif p.target and p.gpu:
            r = indicators(obs, p.target, p.gpu)
            f = lambda x, n=1: "—" if x is None else f"{x:.{n}f}"
            share = r["share_of_requests_at_target"]
            msg += (f"; requests {f(r['completed_requests'], 0)}; throughput/GPU {f(r['output_throughput_per_gpu'])}; "
                    f"at {p.target:g} tokens/s {f(None if share is None else 100 * share, 0)} %"
                    f"{'' if r['share_exact'] is not False else ' (interpolated)'}; {verdict(r['holds'])}")
        print(msg, flush=True)
        if obs["counter_reset"]:
            print("   WARNING: counters went down (server restarted during the window?): window unusable.")
        if stop:
            print("Stop requested: last window shortened.", flush=True)
            break
    print(f"Done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
