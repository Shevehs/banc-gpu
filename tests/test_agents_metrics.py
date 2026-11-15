"""agents_test/metrics.py: /metrics parsing and the background sampler, against a local HTTP stub."""
import time
import urllib.error

import pytest

from agents_test import metrics
from agents_test.metrics import Sampler, fetch

TEXT = """# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0"} 2.0
vllm:num_preemptions_total{engine="1"} 3.0
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.75
vllm:num_requests_waiting{engine="0"} 4
bad line here
vllm:weird{x="1"} +Inf
vllm:nanny NaN
vllm:broken{x="1"} abc
"""


def test_fetch_sums_by_name_and_strips_total(stub):
    s = stub({"/metrics": (200, TEXT, {})})
    v = fetch(s.url + "/metrics")
    assert v["vllm:num_preemptions"] == 5.0 and v["vllm:kv_cache_usage_perc"] == 1.0 and v["vllm:num_requests_waiting"] == 4.0
    assert "vllm:weird" not in v and "vllm:nanny" not in v and "vllm:broken" not in v     # unparsable values are skipped


def test_fetch_per_series_maxima(stub):
    s = stub({"/metrics": (200, TEXT, {})})
    maxima = {}
    fetch(s.url + "/metrics", maxima=maxima)
    assert maxima["vllm:kv_cache_usage_perc"] == 0.75 and maxima["vllm:num_preemptions"] == 3.0


def test_fetch_sends_no_credentials(stub):
    s = stub({"/metrics": (200, TEXT, {})})
    fetch(s.url + "/metrics")
    assert "Authorization" not in s.seen[0]["headers"]


def test_fetch_refuses_redirects(stub):
    s = stub({"/metrics": (302, "", {"Location": "/other"}), "/other": (200, TEXT, {})})
    with pytest.raises(ValueError, match="redirect"):
        fetch(s.url + "/metrics")


def test_fetch_http_error(stub):
    with pytest.raises(urllib.error.HTTPError):
        fetch(stub({}).url + "/metrics")


def test_counters_cover_what_the_report_reads():
    assert set(metrics.COUNTERS) == {"preemptions", "prefix_cache_hits", "prefix_cache_queries", "mtp_accepted", "mtp_proposed"}


def counting_stub(stub):
    """A /metrics endpoint whose preemption counter grows by 2 at each request, with a KV gauge that peaks once."""
    state = {"n": 0}

    def body():
        state["n"] += 1
        kv = 0.9 if state["n"] == 3 else 0.2
        return (f'vllm:num_preemptions_total{{engine="0"}} {2 * state["n"]}\nvllm:prefix_cache_hits_total 1\n'
                f"vllm:kv_cache_usage_perc {kv}\nvllm:num_requests_waiting {state['n']}\n")
    return stub({"/metrics": (200, body, {})})


def test_sampler_result_has_the_keys_the_report_reads(stub):
    s = counting_stub(stub)
    now = time.time()
    sampler = Sampler([s.url + "/metrics"], now, now + 0.8, step_s=0.1).start()
    r = sampler.result()
    assert r["reads"] >= 3 and r["failed_reads"] == 0 and r["urls"] == [s.url + "/metrics"]
    assert r["preemptions"] == 2 * (r["reads"] - 1)
    assert r["prefix_cache_hits"] == 0.0
    assert r["prefix_cache_queries"] is None and r["mtp_accepted"] is None and r["mtp_proposed"] is None
    assert r["kv_cache_max_fraction"] == 0.9 and r["max_waiting_requests"] == r["reads"]
    assert set(r) == {"urls", "reads", "failed_reads", "kv_cache_max_fraction", "max_waiting_requests", *metrics.COUNTERS}


def test_sampler_sums_replicas(stub):
    a, b = stub({"/metrics": (200, "vllm:kv_cache_usage_perc 0.3\nvllm:num_requests_waiting 1\n", {})}), \
        stub({"/metrics": (200, "vllm:kv_cache_usage_perc 0.6\nvllm:num_requests_waiting 2\n", {})})
    now = time.time()
    r = Sampler([a.url + "/metrics", b.url + "/metrics"], now, now + 0.3, step_s=0.1).start().result()
    assert r["kv_cache_max_fraction"] == 0.6 and r["max_waiting_requests"] == 3.0


def test_sampler_without_urls_does_not_start_a_thread():
    s = Sampler([], time.time(), time.time() + 1)
    assert s.start() is s
    r = s.result()
    assert r["reads"] == 0 and r["preemptions"] is None and r["kv_cache_max_fraction"] is None


def test_sampler_counts_failed_reads(stub):
    s = stub({})
    now = time.time()
    r = Sampler([s.url + "/metrics"], now, now + 0.4, step_s=0.1).start().result()
    assert r["reads"] == 0 and r["failed_reads"] >= 2 and r["preemptions"] is None


def test_sampler_needs_two_reads_for_a_delta(stub):
    s = counting_stub(stub)
    now = time.time()
    r = Sampler([s.url + "/metrics"], now, now + 0.01, step_s=0.3).start().result()
    assert r["reads"] == 1 and r["preemptions"] is None
