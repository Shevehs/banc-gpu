"""Sampling of vLLM's Prometheus metrics during one level of the test agents.

Polls one or more /metrics URLs every `step_s` seconds between `t_start` and `t_end`,
and computes, over that window: the change in the counters (preemptions, prefix cache,
speculative decoding) and the maximum KV cache occupancy. Several URLs (several replicas)
are summed; for gauges (KV cache), the maximum per series is kept (one engine per series when
DP > 1). No key is sent: vLLM does not protect /metrics. Standard library only.
"""
import re
import threading
import time
import urllib.request

COUNTERS = {
    "preemptions": "vllm:num_preemptions",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "mtp_accepted": "vllm:spec_decode_num_accepted_tokens",
    "mtp_proposed": "vllm:spec_decode_num_draft_tokens",
}
KV_GAUGE = "vllm:kv_cache_usage_perc"
WAITING_GAUGE = "vllm:num_requests_waiting"
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfa]+)")


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # private network: no proxy


def fetch(url, timeout_s=5, maxima=None):
    """Values of a /metrics endpoint, summed by name (the _total suffix of counters is removed).
    If `maxima` is a dict, it receives the per-series maximum of each name."""
    with _OPENER.open(urllib.request.Request(url), timeout=timeout_s) as r:
        if r.geturl() != url:
            raise ValueError("redirect refused")
        text = r.read().decode("utf-8", "replace")
    vals = {}
    for row in text.splitlines():
        m = _LINE.match(row)
        if not m:
            continue
        name = m.group(1)
        if name.endswith("_total"):
            name = name[:-6]
        try:
            x = float(m.group(3))
        except ValueError:
            continue
        vals[name] = vals.get(name, 0.0) + x
        if maxima is not None:
            maxima[name] = x if name not in maxima else max(maxima[name], x)
    return vals


class Sampler:
    """Background sampling; result() gives the changes over the window and the maximum KV cache."""

    def __init__(self, urls, t_start, t_end, step_s=2.0):
        self.urls, self.t_start, self.t_end, self.step = urls, t_start, t_end, step_s
        self.first, self.last, self.kv_max, self.errors, self.reads = {}, {}, None, 0, 0
        self.waiting_max = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        """Start the sampling thread (nothing if there are no URLs) and return `self`."""
        if self.urls:
            self._thread.start()
        return self

    def _snapshot(self):
        total, kv = {}, None
        for u in self.urls:
            mx = {}
            v = fetch(u, maxima=mx)
            for k, x in v.items():
                total[k] = total.get(k, 0.0) + x
            if KV_GAUGE in mx:
                kv = mx[KV_GAUGE] if kv is None else max(kv, mx[KV_GAUGE])
        return total, kv

    def _loop(self):
        while time.time() < self.t_start:
            time.sleep(min(self.step, max(0.0, self.t_start - time.time())))
        while time.time() <= self.t_end:
            try:
                total, kv = self._snapshot()
                if not self.first:
                    self.first = total
                self.last = total
                if kv is not None:
                    self.kv_max = kv if self.kv_max is None else max(self.kv_max, kv)
                if WAITING_GAUGE in total:
                    a = total[WAITING_GAUGE]
                    self.waiting_max = a if self.waiting_max is None else max(self.waiting_max, a)
                self.reads += 1
            except Exception:  # noqa: BLE001 — a missed read does not stop the level
                self.errors += 1
            time.sleep(self.step)

    def result(self):
        """Wait for the sampling to finish (at most three steps) and return the measurements as a dict.

        Keys: urls, reads, failed_reads, kv_cache_max_fraction, max_waiting_requests, and the change over the
        window of each counter in COUNTERS (None if fewer than two reads)."""
        if self.urls and self._thread.is_alive():
            self._thread.join(timeout=self.step * 3)
        r = {"urls": self.urls, "reads": self.reads, "failed_reads": self.errors,
             "kv_cache_max_fraction": self.kv_max, "max_waiting_requests": self.waiting_max}
        for key, name in COUNTERS.items():
            r[key] = (self.last[name] - self.first[name]) if self.reads >= 2 and name in self.last and name in self.first else None
        return r
