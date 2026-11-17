"""Observation of real usage: queries Prometheus, which scrapes /metrics from vLLM (and LiteLLM), and gives a
verdict per window, per period or per agent. Generates no load and has nothing to start: Prometheus keeps the
history, so any past range can be queried.

Usage: python -m observation.query --prometheus http://127.0.0.1:9090 --target 30 --gpu 8
       [--vllm 'job="vllm"'] [--litellm 'job="litellm"' --by metadata_agent]
       [--since 2h | --start "2026-11-14 09:00" [--end "2026-11-14 17:00"]] [--window 5m]
       [--view windows|period|agents] [--period range|hour] [--ttft-max ms]
       [--exclude "2026-11-14 10:00/2026-11-14 11:00" ...] [--scrape 15s] [--output table.csv]
       [--view-output view.csv]
  --vllm, --litellm : Prometheus label selectors of the targets (job of prometheus.yml). All the matching
            series are added together: one selector = one configuration (all replicas).
  --target : target interactivity per request (tokens/s). A window "holds" if at least 90 % of the completed
            requests reach it (same threshold as the P10 interactivity of the stress tests, measured here by vLLM),
            with no preemption and no failed request (and a P95 TTFT <= --ttft-max if given). "Undetermined":
            the target falls in a histogram bucket that straddles 90 % (bounds to add, README section 7.1).
  --gpu   : number of GPUs behind the vLLM selector, all replicas included (required here: Prometheus does not
            see the server fingerprint)
  --exclude : ranges to discard (stress test on the same server: the synthetic load would mix with real usage)
  --scrape : Prometheus scrape interval (scrape_interval), for the gauges. Optional: by default it is read from Prometheus
            (the interval of the vLLM and LiteLLM targets, else the global one), and 15 s if Prometheus cannot say.
  --output : CSV of the windows, whatever the view (one row per window, discarded ones included)
  --view-output : CSV of the rows of the chosen view: the windows, or one row per period (with the number of windows
            merged and the hour it starts), or one row per agent
Key: BENCH_PROMETHEUS_KEY environment variable if Prometheus requires a token (Bearer). Standard library only.
"""
import argparse
import csv
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from observation.analysis import COUNTERS
from observation.analysis import HISTOGRAMS
from observation.analysis import LITELLM
from observation.analysis import agent_indicators
from observation.analysis import cumulative
from observation.analysis import indicators
from observation.analysis import verdict

RANGES = [(0, "night (0-6 h)"), (6, "morning (6-12 h)"), (12, "afternoon (12-18 h)"), (18, "evening (18-24 h)")]
SELECTOR = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*"'
                      r'(\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*")*\s*$')
LABEL = re.compile(r"^(api_key_alias|team_alias|user|end_user|metadata_[A-Za-z0-9_]{1,40})$")

# Counters that fall back to zero or change identity when a server restarts (restart signs, see restart_signs)
VLLM_WITNESSES = ["vllm:request_success_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
                  "vllm:request_time_per_output_token_seconds_count", "http_requests_total"]
LITELLM_WITNESSES = ["litellm_request_total_latency_metric_count", "litellm_output_tokens_metric_total",
                     "litellm_input_tokens_metric_total"]


def fmt(x, n=1):
    """Number with `n` decimals; an em dash if the value is missing."""
    return "—" if x is None or x == "" else f"{float(x):.{n}f}"


def pct(x):
    """Fraction as a whole percentage; an em dash if the value is missing."""
    return "—" if x is None or x == "" else f"{100 * float(x):.0f}%"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # a redirect is an error: the key must not be sent elsewhere
        return None


# No proxy: Prometheus is on the private network, and its key must not go through a proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def parse_duration(v):
    """'300', '300s', '5m', '2h', '1d' -> seconds.

    Raises:
        SystemExit: if `v` is not a duration."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", str(v))
    if not m:
        raise SystemExit(f"invalid duration: {v} (e.g. 300, 5m, 2h, 1d)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def parse_instant(v):
    """'YYYY-MM-DD HH:MM' (local time) or a Unix timestamp -> seconds.

    Raises:
        SystemExit: if `v` is neither."""
    v = str(v).strip()
    if re.match(r"^\d+(\.\d+)?$", v):
        return float(v)
    for fmt_ in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return time.mktime(time.strptime(v, fmt_))
        except ValueError:
            pass
    raise SystemExit(f"invalid date: {v} (form YYYY-MM-DD HH:MM, local time)")


def check_selector(v, what):
    """Return the label selector `v` (without braces) if it is well formed.

    Raises:
        SystemExit: otherwise; `what` names the option in the message."""
    v = (v or "").strip()
    if not SELECTOR.match(v):
        raise SystemExit(f"{what}: label selector expected, e.g. job=\"vllm\" or job=\"vllm\",instance=~\"10.0.0.*\"")
    return v


DEFAULT_SCRAPE_S = 15
_PROMETHEUS_DURATION = re.compile(r"^(?:\d+(?:\.\d+)?(?:ms|s|m|h|d|w|y))+$")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d|w|y)")
_MATCHER = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*(=~|!~|!=|=)\s*"([^"\\]*)"')


def parse_prometheus_duration(v):
    """A duration as Prometheus writes it ("15s", "1m30s", "500ms", "2h") -> seconds, or None if it is not one."""
    v = str(v or "").strip()
    if not _PROMETHEUS_DURATION.match(v):
        return None
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
    return sum(float(n) * units[u] for n, u in _DURATION_PART.findall(v))


def matches_selector(labels, selector):
    """True if the target labels (a dict) satisfy the label selector (matchers separated by commas, no braces)."""
    for name, op, value in _MATCHER.findall(selector or ""):
        have = labels.get(name, "")
        if op == "=":
            ok = have == value
        elif op == "!=":
            ok = have != value
        else:
            try:
                ok = re.fullmatch(value, have) is not None
            except re.error:
                return False
            ok = ok if op == "=~" else not ok
        if not ok:
            return False
    return True


def detect_scrape_interval(prom, selectors):
    """(seconds, where it comes from): the scrape interval of Prometheus, whole seconds, rounded up.

    Taken from the active targets that match one of the `selectors` (the longest interval if they differ, which is the
    safe side), else from the global `scrape_interval`, else DEFAULT_SCRAPE_S."""
    data = prom.get_data("/api/v1/targets?state=active") or {}
    found = []
    for target in data.get("activeTargets") or []:
        labels = dict(target.get("labels") or {})
        labels.setdefault("job", target.get("scrapePool", ""))
        seconds = parse_prometheus_duration(target.get("scrapeInterval"))
        if seconds and any(matches_selector(labels, s) for s in selectors if s):
            found.append(seconds)
    if found:
        return max(1, math.ceil(max(found))), "read from Prometheus: the interval of the selected targets"
    config = (prom.get_data("/api/v1/status/config") or {}).get("yaml") or ""
    m = re.search(r"^global:\n(?:[ \t]+.*\n)*?[ \t]+scrape_interval:[ \t]*(\S+)", config, re.MULTILINE)
    seconds = parse_prometheus_duration(m.group(1)) if m else None
    if seconds:
        return max(1, math.ceil(seconds)), "read from Prometheus (global scrape_interval)"
    return DEFAULT_SCRAPE_S, "default: Prometheus did not say"


class Prometheus:
    """Minimal client of the Prometheus HTTP API (query_range, and a tolerant read of the other endpoints)."""

    def __init__(self, base, key=None, timeout_s=30):
        self.base, self.key, self.timeout_s = base.rstrip("/"), key, timeout_s

    def get_data(self, path):
        """The `data` of a GET endpoint (path and query string), or None when Prometheus cannot answer.

        Meant for optional information: an unreachable server, a refusal or an unexpected body give None."""
        req = urllib.request.Request(self.base + path)
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        try:
            with _OPENER.open(req, timeout=min(self.timeout_s, 10)) as r:
                j = json.loads(r.read(5_000_000).decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
            return None
        data = j.get("data") if isinstance(j, dict) and j.get("status") == "success" else None
        return data if isinstance(data, dict) else None

    def query_range(self, q, start, end, step):
        """[(labels, {timestamp: value})] for a PromQL query evaluated from `start` to `end`, every `step` seconds.

        Raises:
            SystemExit: if Prometheus is unreachable or answers with an error."""
        url = self.base + "/api/v1/query_range?" + urllib.parse.urlencode(
            {"query": q, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": f"{step:g}"})
        req = urllib.request.Request(url)
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        try:
            with _OPENER.open(req, timeout=self.timeout_s) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
            raise SystemExit(f"Prometheus answered HTTP {e.code} to \"{q[:120]}\": {body}") from e
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"Prometheus unreachable ({self.base}): {e}") from e
        if j.get("status") != "success":
            raise SystemExit(f"Prometheus: {j.get('error') or j}")
        return [(s.get("metric") or {}, {round(float(t)): float(v) for t, v in s.get("values") or []
                                         if v not in ("NaN", "+Inf", "-Inf")})
                for s in (j.get("data") or {}).get("result") or []]


def increase(series, w, by=""):
    """PromQL for the increase of a counter over each window ]t-w, t], added up (grouped by labels `by`).
    - Value at t minus value at t-w (last scrapes before those instants): exact, and successive windows add up
      without loss or double counting.
    - Series that appeared during the window (first 5xx status code, first call of an agent in LiteLLM): its whole
      value, if its target was already scraped at the start of the window (otherwise, for instance at the first
      start of Prometheus, the value accumulated since the server started would distort the window).
    A reset to zero (restart) makes the difference wrong: those windows are spotted by restart_signs()
    and discarded."""
    W = f"{int(w)}s"
    exact = f"(({series} - {series} offset {W}) >= 0)"
    new = f"(({series} unless {series} offset {W}) and on(job, instance) (up offset {W} == 1))"
    q = f"{exact} or {new}"
    return f"sum by ({by}) ({q})" if by else f"sum({q})"


def restart_signs(sel, names, w, scrape):
    """PromQL counting the signs of a restart in the window (plus one scrape before, to see the boundary being
    crossed): drop of a witness counter (resets) or change of process_start_time_seconds (the only sign when the
    counter went past its previous value between two scrapes; exposed by prometheus_client on Linux)."""
    R = f"{int(w + scrape)}s"
    terms = [f"(sum(resets({n}{sel}[{R}])) or vector(0))" for n in names]
    terms.append(f"(sum(changes(process_start_time_seconds{sel}[{R}])) or vector(0))")
    return " + ".join(terms)


def collect(prom, sel_v, start, end, w, scrape, sel_l=None, by=None):
    """One observation per window: {start, end, counters, histograms, gauges, targets, agents, ...}.

    Raises:
        SystemExit: if the range holds no whole window, or too many of them."""
    ends = []
    t = start + w
    while t <= end + 1e-6:
        ends.append(round(t))
        t += w
        if len(ends) > 10_000:
            raise SystemExit("More than 10,000 windows (Prometheus refuses more than 11,000 points per query): "
                             "lengthen --window or shorten the range.")
    if not ends:
        raise SystemExit("Range shorter than one window: lengthen --since or shorten --window.")
    first, last = ends[0], ends[-1]
    W, C = f"{int(w)}s", f"{int(scrape)}s"
    one = lambda q: (prom.query_range(q, first, last, w) or [({}, {})])[0][1]
    by_le = lambda q: {s.get("le"): v for s, v in prom.query_range(q, first, last, w)}
    sel = "{" + sel_v + "}"
    resets = one(restart_signs(sel, VLLM_WITNESSES, w, scrape))
    scrapes = one(f"sum(count_over_time(up{sel}[{W}]))")
    err_sel = "{" + sel_v + ',handler=~"/v1/.*",status=~"[45].*"}'
    counters = {k: one(increase(n + (err_sel if k == "failed_requests" else sel), w)) for k, n in COUNTERS.items()}
    exposes_errors = one(f"count(http_requests_total{sel})")
    histograms = {k: by_le(increase(n + "_bucket" + sel, w, "le")) for k, n in HISTOGRAMS.items()}
    gauges = {
        "kv_max": one(f"max_over_time(max(vllm:kv_cache_usage_perc{sel})[{W}:{C}])"),
        "waiting_max": one(f"max_over_time(sum(vllm:num_requests_waiting{sel})[{W}:{C}])"),
        "running_avg": one(f"avg_over_time(sum(vllm:num_requests_running{sel})[{W}:{C}])"),
        "running_max": one(f"max_over_time(sum(vllm:num_requests_running{sel})[{W}:{C}])"),
    }
    targets_min = one(f"min_over_time(sum(up{sel})[{W}:{C}])")
    targets = one(f"max_over_time(count(up{sel})[{W}:{C}])")
    targets_at_start = one(f"sum(up{sel} offset {W})")  # targets reachable at the last scrape before the start
    # Targets scraped in the scrape that precedes the start (]t-w-1.5C, t-w]): without it, "X offset W" would go
    # back up to 5 min (Prometheus lookback) and the window would count the outage.
    R0 = f"{int(1500 * scrape)}ms"
    targets_at_boundary = one(f"count(count_over_time(up{sel}[{R0}] offset {W}))")
    # Same at the end (]t-1.5C, t]): without a recent scrape, the end value would be old (truncated window).
    targets_at_end = one(f"count(count_over_time(up{sel}[{R0}]))")
    agents_t, litellm_targets = None, None
    if sel_l:
        sl = "{" + sel_l + "}"
        litellm_targets = (one(f"min_over_time(sum(up{sl})[{W}:{C}])"), one(f"max_over_time(count(up{sl})[{W}:{C}])"),
                           one(restart_signs(sl, LITELLM_WITNESSES, w, scrape)))
        agents_t = {}
        for k in ("output_tokens", "input_tokens"):
            for lab, vals in prom.query_range(increase(LITELLM[k] + sl, w, by), first, last, w):
                for t, v in vals.items():
                    agents_t.setdefault(t, {}).setdefault(lab.get(by, ""), {})[k] = v
        for k in ("ttft", "latency"):
            for lab, vals in prom.query_range(increase(LITELLM[k] + "_bucket" + sl, w, by + ", le"), first, last, w):
                for t, v in vals.items():
                    agents_t.setdefault(t, {}).setdefault(lab.get(by, ""), {}).setdefault(k, {})[lab.get("le")] = v
            for suffix in ("sum", "count"):
                for lab, vals in prom.query_range(increase(LITELLM[k] + "_" + suffix + sl, w, by), first, last, w):
                    for t, v in vals.items():
                        agents_t.setdefault(t, {}).setdefault(lab.get(by, ""), {})[f"{k}_{suffix}"] = v
    obs = []
    for t in ends:
        c = {k: v.get(t) for k, v in counters.items()}
        if c["failed_requests"] is None and exposes_errors.get(t):
            c["failed_requests"] = 0.0  # http_requests_total series present, none on /v1 with 4xx/5xx
        o = {"start": t - w, "end": t, "counters": c,
             "histograms": {k: {le: v[t] for le, v in h.items() if t in v} for k, h in histograms.items()},
             "gauges": {k: v.get(t) for k, v in gauges.items()},
             "targets": targets.get(t), "targets_always_up": targets_min.get(t),
             "targets_at_start": targets_at_start.get(t),
             "resets": resets.get(t) or 0, "scrapes": scrapes.get(t),
             "targets_at_boundary": targets_at_boundary.get(t) or 0, "targets_at_end": targets_at_end.get(t) or 0}
        if sel_l:
            o["litellm_targets"] = (litellm_targets[1].get(t), litellm_targets[0].get(t))
            o["litellm_resets"] = litellm_targets[2].get(t) or 0
            agents = {}
            for agent, d in (agents_t.get(t) or {}).items():
                d = {"output_tokens": 0.0, "input_tokens": 0.0, "ttft": {}, "ttft_sum": 0.0, "ttft_count": 0.0,
                     "latency": {}, "latency_sum": 0.0, "latency_count": 0.0, **d}
                if d["latency_count"] > 0 or d["output_tokens"] > 0:
                    agents[agent] = d
            o["agents"] = agents
        obs.append(o)
    return obs


def period_label(t, mode):
    """Label of the period that contains the epoch time `t` (local time): "HH h" if `mode` is "hour",
    else the name of the 6-hour range."""
    hour = time.localtime(t).tm_hour
    return f"{hour:02d} h" if mode == "hour" else [name for start, name in RANGES if hour >= start][-1]


def sum_histograms(histograms):
    """Sum several {le: count} histograms bound by bound."""
    total = {}
    for h in histograms:
        for le, n in h.items():
            total[le] = total.get(le, 0.0) + n
    return total


def merge(obs):
    """Add successive windows (counters, histograms; gauges: maximum, concurrency: mean) into one pseudo-window."""
    c = {}
    for o in obs:
        for k, v in o["counters"].items():
            if k == "failed_requests":  # None = no /v1 series created yet: nothing to add
                c[k] = c.get(k) if v is None else (c.get(k) or 0.0) + v
            else:  # other counter unreadable in a window: total unknown
                c[k] = None if v is None or (k in c and c[k] is None) else c.get(k, 0.0) + v
    g = {k: max((o["gauges"][k] for o in obs if o["gauges"][k] is not None), default=None) for k in obs[0]["gauges"]}
    means = [o["gauges"]["running_avg"] for o in obs if o["gauges"]["running_avg"] is not None]
    g["running_avg"] = sum(means) / len(means) if means else None
    return {"start": 0.0, "end": sum(o["end"] - o["start"] for o in obs), "counters": c, "gauges": g,
            "histograms": {k: sum_histograms([o["histograms"][k] for o in obs]) for k in obs[0]["histograms"]}}


def share_text(r):
    """Share of requests at the target as a percentage; "*" if interpolated, an em dash if unknown."""
    if r["share_of_requests_at_target"] is None:
        return "—"
    return pct(r["share_of_requests_at_target"]) + ("" if r["share_exact"] else "*")


def headers():
    """Header of the indicator columns, shared by the windows and period views."""
    return (f"{'avg conc':>9}{'requests':>9}{'err':>5}{'tput/GPU':>11}{'at target':>11}{'inter P10≥':>11}"
            f"{'TTFT P95≤ ms':>13}{'preempt':>8}{'cache':>7}{'MTP':>6}{'KV max':>8}  verdict")


def format_row(r):
    """Indicator columns of row `r` (one window or merged period), aligned with `headers`."""
    return (f"{fmt(r['mean_concurrency']):>9}{fmt(r['completed_requests'], 0):>9}{fmt(r['failed_requests'], 0):>5}"
            f"{fmt(r['output_throughput_per_gpu']):>11}{share_text(r):>11}"
            f"{('<' if r['interactivity_bound'] else '') + fmt(r['estimated_interactivity_p10']):>11}"
            f"{('>' if r['ttft_p95_beyond'] else '') + fmt(r['estimated_ttft_p95_ms'], 0):>13}"
            f"{fmt(r['preemptions'], 0):>8}"
            f"{pct(r['prefix_cache']):>7}{pct(r['mtp_acceptance']):>6}{pct(r['kv_cache_max_fraction']):>8}"
            f"  {r['verdict']}")


def is_increasing(h):
    """True if the cumulative counts of a histogram grow with the bound (otherwise: inconsistent data)."""
    c = cumulative(h)
    return all(b[1] >= a[1] - 1e-9 for a, b in zip(c, c[1:], strict=False))


def evaluate(o, args):
    """Indicators and verdict of one window or merged period `o`, for the target and GPU count in `args`."""
    r = indicators(o, args.target, args.gpu, args.ttft_max)
    if not all(is_increasing(h) for h in o["histograms"].values()):
        for k in ("share_of_requests_at_target", "share_min", "share_max", "estimated_interactivity_p10",
                  "estimated_ttft_p50_ms", "estimated_ttft_p95_ms"):
            r[k] = None
        r["holds"], r["inconsistent_histogram"] = None, True
    r["mean_concurrency"] = o["gauges"].get("running_avg")
    r["max_concurrency"] = o["gauges"].get("running_max")
    r["verdict"] = "no requests" if r["completed_requests"] == 0 else verdict(r["holds"])
    return r


def discard_reason(o, w, scrape, excluded):
    """Why window `o` must be discarded, or None if it is usable."""
    if not o["targets"]:
        return "no vLLM target scraped (selector, or Prometheus not started yet)"
    if (o["targets_at_start"] or 0) < o["targets"]:
        return "target not yet scraped or unreachable at the start of the window"
    if o["targets_at_boundary"] < o["targets"] or o["targets_at_end"] < o["targets"]:
        return ("no scrape just before the start or the end of the window (Prometheus stopped, or --scrape "
                "shorter than its scrape_interval): difference distorted")
    if (o["targets_always_up"] or 0) < o["targets"]:
        return "a vLLM target unreachable during the window (missed scrape)"
    if o["resets"] > 0:
        return "vLLM restarted during the window (counters reset to zero)"
    expected = o["targets"] * w / scrape
    if (o["scrapes"] or 0) < 0.8 * expected:
        return (f"missing scrapes ({o['scrapes'] or 0:.0f} for ≈ {expected:.0f} expected; "
                "Prometheus stopped, or --scrape shorter than its scrape_interval)")
    if any(d0 < o["end"] and o["start"] < d1 for d0, d1 in excluded):
        return "excluded range (stress test)"
    return None


def build_rows(obs, args, sel_l, w, scrape, excluded):
    """Evaluate every window. Returns (rows, alerts, valid): one row per window, the warnings to print, and the
    windows that are usable (not discarded)."""
    rows, alerts, valid = [], [], []
    for o in obs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(o["start"]))
        r = evaluate(o, args)
        reason = discard_reason(o, w, scrape, excluded)
        if reason:
            r = {k: (None if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items()}
            r["verdict"] = "run in progress" if reason.startswith("excluded") else "discarded"
            if not reason.startswith("no vLLM target"):
                alerts.append(f"{when}: {reason}: window discarded.")
        else:
            valid.append(o)
            if o["scrapes"] and o["targets"] and o["scrapes"] > 1.2 * o["targets"] * w / scrape:
                actual = o["targets"] * w / o["scrapes"]
                alerts.append(f"--scrape ({scrape:g} s) is longer than the actual Prometheus interval "
                              f"(≈ {actual:.0f} s): scrape checks and gauges distorted; give --scrape {actual:.0f}s.")
            if r.get("inconsistent_histogram"):
                alerts.append(f"{when}: inconsistent histogram (non-increasing buckets): verdict undetermined.")
            if (r["completed_requests"] or 0) and r["completed_requests"] < 50:
                alerts.append(f"{when}: fewer than 50 completed requests: percentiles unreliable.")
        if sel_l:
            alerts.extend(check_litellm(o, when, sel_l))
        rows.append({"start": when, "end": time.strftime("%H:%M", time.localtime(o["end"])),
                     "duration_s": round(o["end"] - o["start"]), "targets": o["targets"], **r,
                     "discarded": bool(reason), "discard_reason": reason or ""})
    return rows, alerts, valid


def check_litellm(o, when, sel_l):
    """Warnings about the LiteLLM data of window `o`; drops its per-agent data (o["agents"] = None) if unreliable."""
    alerts = []
    if o.get("litellm_targets") and (o["litellm_targets"][1] or 0) < (o["litellm_targets"][0] or 0):
        alerts.append(f"{when}: a LiteLLM target unreachable during the window: discarded from the per-agent view.")
        o["agents"] = None
    if o.get("litellm_resets"):
        alerts.append(f"{when}: LiteLLM restarted during the window: discarded from the per-agent view.")
        o["agents"] = None
    if o.get("agents") and not all(is_increasing(d[k]) for d in o["agents"].values() for k in ("ttft", "latency")):
        alerts.append(f"{when}: inconsistent LiteLLM histogram (restart not spotted?): "
                      "discarded from the per-agent view.")
        o["agents"] = None
    if o["targets"] and not (o.get("litellm_targets") or (None,))[0]:
        alerts.append(f"{when}: no LiteLLM target scraped (selector {sel_l}).")
    return alerts


def period_groups(valid, args):
    """[(label, number of windows, indicators of the merged windows)] per period (args.period), sorted by label."""
    groups = {}
    for o in valid:
        groups.setdefault(period_label(o["start"], args.period), []).append(o)
    return [(label, len(groups[label]), evaluate(merge(groups[label]), args)) for label in sorted(groups)]


def period_start_hour(label, mode):
    """Hour of the day (0-23, local time) at which the period `label` starts: to put the periods in order."""
    if mode == "hour":
        return int(label[:2])
    return next(start for start, name in RANGES if name == label)


def period_rows(valid, args):
    """Rows of the per-period view, in the order of the day: the period, the hour it starts, the number of windows
    merged, then the indicators of the merged windows (same columns as the windows)."""
    rows = [{"period": label, "start_hour": period_start_hour(label, args.period), "windows": n, **r}
            for label, n, r in period_groups(valid, args)]
    return sorted(rows, key=lambda r: r["start_hour"])


def period_view(valid, args):
    """Print the indicators of the merged windows per period (args.period)."""
    print(f"{'period':<22}{'windows':>9}" + headers())
    for label, n, r in period_groups(valid, args):
        print(f"{label:<22}{n:>9}" + format_row(r))
    print("\nThroughput/GPU: mean over the period; avg conc: requests in flight on average. Peak of a period: "
          "--view windows.")


def agent_totals(valid):
    """The metrics of every usable window added together, per agent."""
    totals = {}
    for o in valid:
        for agent, d in (o.get("agents") or {}).items():
            t = totals.setdefault(agent, {k: ({} if isinstance(v, dict) else 0.0) for k, v in d.items()})
            for k, v in d.items():
                t[k] = sum_histograms([t[k], v]) if isinstance(v, dict) else t[k] + v
    return totals


def agent_rows(valid):
    """Rows of the per-agent view: one per agent (the label as LiteLLM gives it, then the name shown)."""
    rows = []
    for agent, r in agent_indicators(agent_totals(valid)).items():
        rows.append({"agent": agent, "agent_name": agent if agent not in ("", "None") else "(no identity)", **r})
    return rows


def agents_view(valid, args, sel_l):
    """Print the per-agent view (LiteLLM metrics), all the usable windows added together."""
    totals = agent_totals(valid)
    if not totals:
        print(f"No per-agent request in the range (LiteLLM {{{sel_l}}}, label {args.by}).")
    else:
        print(f"  {'agent':<24}{'requests':>9}{'output share':>13}{'input/req':>11}{'output/req':>11}"
              f"{'TTFT P50≤ s':>12}{'TTFT P95≤ s':>12}{'latency P95≤ s':>15}{'avg speed':>12}")
        for agent, r in agent_indicators(totals).items():
            name = (agent if agent not in ("", "None") else "(no identity)")[:24]
            print(f"  {name:<24}{fmt(r['requests'], 0):>9}{pct(r['output_token_share']):>13}"
                  f"{fmt(r['input_tokens_per_request'], 0):>11}{fmt(r['output_tokens_per_request'], 0):>11}"
                  f"{fmt(r['ttft_p50_s'], 2):>12}{('>' if r['ttft_p95_beyond'] else '') + fmt(r['ttft_p95_s'], 2):>12}"
                  f"{('>' if r['latency_p95_beyond'] else '') + fmt(r['latency_p95_s'], 2):>15}"
                  f"{fmt(r['mean_speed']):>12}")
        if "" in totals or "None" in totals:
            print(f"   (no identity): requests without a value for the label {args.by}.")
    print("\nPer-agent view read from LiteLLM: TTFT only for streamed requests; average speed = output tokens "
          "/ (latency - TTFT), a mean and not a percentile, proxy time included, empty if some of the "
          "requests are not streamed. Bounds <=: buckets of the LiteLLM histograms.")


def windows_view(rows, args):
    """Print one line per window, then the highest mean concurrency that holds the target."""
    print(f"{'start':<17}{'end':>6}" + headers())
    for r in rows:
        print(f"{r['start']:<17}{r['end']:>6}" + format_row(r))
    if any(r["share_exact"] is False for r in rows):
        print(f"\n* interpolated share: no histogram bound at {1000 / args.target:.1f} ms; start vLLM with "
              "--custom-histogram-buckets (README, section 7.1). \"undetermined\": the target falls in a bucket "
              "that straddles 90 %.")
    print("≥ and ≤: bounds read from the histogram buckets (< or >: beyond the last bound). "
          "err: /v1 requests with 4xx or 5xx seen by vLLM (\"—\" if not exposed). avg conc: requests in flight "
          "on average.")
    held = [r for r in rows if r["verdict"] == "holds" and (r["completed_requests"] or 0) >= 50
            and (r["mean_concurrency"] or 0) > 0]
    failed = [r for r in rows if r["verdict"] == "does not hold" and r["mean_concurrency"] is not None]
    print()
    if held:
        m = max(held, key=lambda r: r["mean_concurrency"])
        print(f"Highest mean concurrency that holds: {m['mean_concurrency']:.1f} requests in flight "
              f"({m['start']}, {fmt(m['output_throughput_per_gpu'])} tokens/s per GPU).")
        lower = [r for r in failed if r["mean_concurrency"] <= m["mean_concurrency"]]
        if lower:
            print("   WARNING: windows with equal or lower concurrency do not hold "
                  f"({', '.join(r['start'] for r in lower[:5])}): see err, preemptions and TTFT.")
        undetermined = [r for r in rows if r["verdict"] == "undetermined"
                        and (r["mean_concurrency"] or 0) > m["mean_concurrency"]]
        if undetermined:
            print(f"   WARNING: {len(undetermined)} busier window(s) with an undetermined verdict "
                  "(histogram bounds, README section 7.1): the limit may be higher.")
        if not [r for r in failed if r["mean_concurrency"] > m["mean_concurrency"]]:
            print("   WARNING: no busier window fails: the limit is not reached in this range.")
    else:
        print(f"No window of at least 50 requests holds {args.target:g} tokens/s for 90 % of the requests "
              "without preemption or error (or the result is undetermined: see the verdicts).")
    if any((r["max_waiting_requests"] or 0) > 0 for r in held):
        print("   WARNING: requests waited in the server queue in windows that hold.")
    if any((r["kv_cache_max_fraction"] or 0) >= 0.95 for r in rows if not r["discarded"]):
        print("   WARNING: KV cache 95 % full or more in at least one window.")


def build_parser():
    """Argument parser of the command line."""
    a = argparse.ArgumentParser(description="Observation of real usage from Prometheus.")
    a.add_argument("--prometheus", required=True, help="Prometheus URL, e.g. http://127.0.0.1:9090")
    a.add_argument("--vllm", default='job="vllm"', help='selector of the vLLM targets (default job="vllm")')
    a.add_argument("--litellm", default=None, help='selector of the LiteLLM targets, e.g. job="litellm" (agents view)')
    a.add_argument("--by", default="metadata_agent", help="LiteLLM label that tells agents apart")
    a.add_argument("--since", default="2h", help="range ending now (e.g. 30m, 2h, 1d); ignored with --start")
    a.add_argument("--start", default=None, help='start of the range, "YYYY-MM-DD HH:MM" (local time)')
    a.add_argument("--end", default=None, help="end of the range (default: now)")
    a.add_argument("--window", default="5m", help="length of a window (e.g. 5m, 15m, 1h)")
    a.add_argument("--scrape", default=None,
                   help="Prometheus scrape_interval (default: read from Prometheus, else 15s)")
    a.add_argument("--target", type=float, required=True, help="target interactivity per request (tokens/s)")
    a.add_argument("--gpu", type=int, required=True, help="GPUs behind the vLLM selector, all replicas")
    a.add_argument("--ttft-max", type=float, default=None, help="maximum accepted estimated P95 TTFT (ms)")
    a.add_argument("--view", choices=["windows", "period", "agents"], default="windows")
    a.add_argument("--period", choices=["hour", "range"], default="range")
    a.add_argument("--exclude", action="append", default=[], help='range "start/end" to discard (stress test)')
    a.add_argument("--output", default=None, help="CSV of the windows (optional)")
    a.add_argument("--view-output", default=None,
                   help="CSV of the rows of the chosen view: windows, periods or agents (optional)")
    return a


def write_csv(path, rows):
    """Write `rows` (dictionaries) as a CSV file with a header: the columns are the keys of all the rows."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Command-line entry point: query Prometheus, print the chosen view and the warnings, write --output and
    --view-output."""
    p = build_parser().parse_args()
    sel_v = check_selector(p.vllm, "--vllm")
    sel_l = check_selector(p.litellm, "--litellm") if p.litellm else None
    if sel_l and not LABEL.match(p.by):
        raise SystemExit("--by: api_key_alias, team_alias, user, end_user or metadata_<field>")
    if p.view == "agents" and not sel_l:
        raise SystemExit("--view agents: give --litellm (selector of the LiteLLM targets in Prometheus).")
    w = parse_duration(p.window)
    scrape_given = parse_duration(p.scrape) if p.scrape else None
    end = parse_instant(p.end) if p.end else time.time()
    start = parse_instant(p.start) if p.start else end - parse_duration(p.since)
    if p.start and not p.end:
        end = time.time()
    end = min(end, time.time())
    excluded = []
    for x in p.exclude:
        if "/" not in x:
            raise SystemExit(f"--exclude: form \"start/end\" expected ({x})")
        d0, d1 = x.split("/", 1)
        excluded.append((parse_instant(d0), parse_instant(d1)))
    prom = Prometheus(p.prometheus, os.environ.get("BENCH_PROMETHEUS_KEY") or None)
    if scrape_given:
        scrape, scrape_source = scrape_given, "given by --scrape"
    else:
        scrape, scrape_source = detect_scrape_interval(prom, [sel_v, sel_l])
    if w < 4 * scrape:
        raise SystemExit(f"--window must be at least 4 x {'--scrape' if scrape_given else 'the scrape interval'} "
                         f"({4 * scrape:g} s" + ("" if scrape_given else f", {scrape_source}") + ").")
    obs = collect(prom, sel_v, start, end, w, scrape, sel_l, p.by if sel_l else None)
    rows, alerts, valid = build_rows(obs, p, sel_l, w, scrape, excluded)
    if all(not o["targets"] for o in obs):
        raise SystemExit(f"No vLLM target {{{sel_v}}} in Prometheus over this range: check the selector "
                         "(job of prometheus.yml) and the Status > Targets page of Prometheus.")
    if p.output:
        write_csv(p.output, rows)
    if p.view_output:
        write_csv(p.view_output, {"period": period_rows, "agents": lambda v, a: agent_rows(v)}.get(p.view, lambda v, a: rows)(valid, p))
    print(f"Prometheus {p.prometheus}, vLLM {{{sel_v}}}"
          + (f", LiteLLM {{{sel_l}}} by {p.by}" if sel_l else "")
          + f"; {time.strftime('%Y-%m-%d %H:%M', time.localtime(obs[0]['start']))} → "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(obs[-1]['end']))}, windows of {w:g} s")
    print(f"Scrape interval: {scrape:g} s ({scrape_source})\n")
    if p.view == "period":
        period_view(valid, p)
    elif p.view == "agents":
        agents_view(valid, p, sel_l)
    else:
        windows_view(rows, p)
    for al in dict.fromkeys(alerts):
        print("   WARNING: " + al)


if __name__ == "__main__":
    main()
