"""observation/analysis.py: LiteLLM per-agent indicators, indicators and verdict branches, metric names."""
import math

import pytest

from observation import analysis
from observation.analysis import agent_indicators, cumulative, indicators, quantile


def test_cumulative_sorts_numerically_with_infinity_last():
    assert [b for b, _ in cumulative({"+Inf": 9.0, "10": 5.0, "0.5": 1.0})] == [0.5, 10.0, math.inf]
    assert cumulative({"Inf": 1.0, "inf": 2.0}) == [(math.inf, 1.0), (math.inf, 2.0)]


def test_quantile_of_a_single_infinite_bucket():
    assert quantile({"+Inf": 4.0}, 0.5) == (0.0, True)


# ---------------------------------------------------------------- LiteLLM per-agent view
def agent(output=1000.0, inp=5000.0, ttft_count=10.0, ttft_sum=4.0, latency_count=10.0, latency_sum=14.0):
    return {"output_tokens": output, "input_tokens": inp, "ttft": {"0.5": 10.0, "+Inf": 10.0}, "ttft_sum": ttft_sum,
            "ttft_count": ttft_count, "latency": {"1": 8.0, "2": 10.0, "+Inf": 10.0}, "latency_sum": latency_sum,
            "latency_count": latency_count}


def test_agent_indicators_values():
    r = agent_indicators({"a": agent(), "b": agent(output=3000.0)})
    a = r["a"]
    assert a["requests"] == 10.0 and a["output_token_share"] == pytest.approx(0.25)
    assert a["input_tokens_per_request"] == 500.0 and a["output_tokens_per_request"] == 100.0
    assert (a["ttft_p50_s"], a["ttft_p95_s"], a["ttft_p95_beyond"]) == (0.5, 0.5, False)
    assert (a["latency_p50_s"], a["latency_p95_s"], a["latency_p95_beyond"]) == (1.0, 2.0, False)
    assert a["mean_speed"] == pytest.approx(1000.0 / 10.0)
    assert list(r) == ["a", "b"]


def test_agent_indicators_mean_speed_needs_every_request_streamed():
    assert agent_indicators({"a": agent(ttft_count=9.0)})["a"]["mean_speed"] is None
    assert agent_indicators({"a": agent(ttft_sum=14.0)})["a"]["mean_speed"] is None      # no decode time
    assert agent_indicators({"a": agent(latency_count=0.0, ttft_count=0.0)})["a"]["mean_speed"] is None


def test_agent_indicators_without_requests_or_tokens():
    a = agent_indicators({"a": agent(output=0.0, inp=0.0, latency_count=0.0)})["a"]
    assert a["output_token_share"] is None and a["input_tokens_per_request"] is None and a["output_tokens_per_request"] is None
    assert agent_indicators({}) == {}


def test_agent_indicators_beyond_the_last_bound():
    a = agent()
    a["latency"] = {"1": 1.0, "+Inf": 10.0}
    assert agent_indicators({"a": a})["a"]["latency_p95_beyond"] is True


# ---------------------------------------------------------------- indicators and verdict
FAST = {"0.01": 10.0, "0.0333": 95.0, "0.1": 100.0, "+Inf": 100.0}       # 95 % of requests at 30 tokens/s or better
SLOW = {"0.01": 0.0, "0.0333": 50.0, "0.1": 100.0, "+Inf": 100.0}
STRADDLE = {"0.01": 10.0, "0.05": 95.0, "+Inf": 100.0}                   # no bound near 1/30 s: interpolated
TTFT = {"0.1": 20.0, "0.5": 96.0, "+Inf": 100.0}                         # P95 at 0.5 s


def obs(tpot=FAST, ttft=TTFT, **counters):
    c = {"output_tokens": 6000.0, "input_tokens": 9000.0, "completed_requests": 100.0, "preemptions": 0.0,
         "prefix_cache_hits": 50.0, "prefix_cache_queries": 100.0, "mtp_accepted": 3.0, "mtp_proposed": 6.0,
         "failed_requests": 0.0}
    c.update(counters)
    return {"start": 1000.0, "end": 1100.0, "counters": c, "histograms": {"ttft": ttft, "tpot": tpot},
            "gauges": {"kv_max": 0.4, "waiting_max": 1.0, "running_max": 2.0}}


def test_indicators_values():
    r = indicators(obs(), 30, 2)
    assert r["output_throughput_per_gpu"] == pytest.approx(6000 / 100 / 2)
    assert r["completed_requests"] == 100.0 and r["prefix_cache"] == 0.5 and r["mtp_acceptance"] == 0.5
    assert r["share_of_requests_at_target"] == 0.95 and r["share_exact"] is True
    assert r["estimated_interactivity_p10"] == pytest.approx(1 / 0.0333) and r["interactivity_bound"] is False
    assert r["estimated_ttft_p50_ms"] == 500.0 and r["estimated_ttft_p95_ms"] == 500.0 and r["ttft_p95_beyond"] is False
    assert r["kv_cache_max_fraction"] == 0.4 and r["max_waiting_requests"] == 1.0


def test_verdict_holds():
    assert indicators(obs(), 30, 1)["holds"] is True


def test_verdict_fails_on_preemptions_or_failed_requests():
    assert indicators(obs(preemptions=1.0), 30, 1)["holds"] is False
    assert indicators(obs(failed_requests=2.0), 30, 1)["holds"] is False
    assert indicators(obs(preemptions=None, failed_requests=None), 30, 1)["holds"] is True    # not exposed: no failure seen


def test_verdict_fails_when_too_few_requests_are_at_target():
    r = indicators(obs(tpot=SLOW), 30, 1)
    assert r["share_of_requests_at_target"] == 0.5 and r["holds"] is False


def test_verdict_undetermined_when_the_bucket_straddles_90_percent():
    r = indicators(obs(tpot=STRADDLE), 30, 1)
    assert r["share_exact"] is False and r["holds"] is None


def test_verdict_undetermined_without_observations():
    r = indicators(obs(tpot={}), 30, 1)
    assert r["share_of_requests_at_target"] is None and r["holds"] is None
    assert r["estimated_interactivity_p10"] is None


def test_verdict_preemptions_win_over_missing_data():
    assert indicators(obs(tpot={}, preemptions=3.0), 30, 1)["holds"] is False


def test_verdict_with_ttft_limit():
    assert indicators(obs(), 30, 1, ttft_max_ms=600)["holds"] is True
    assert indicators(obs(), 30, 1, ttft_max_ms=400)["holds"] is False
    assert indicators(obs(ttft={}), 30, 1, ttft_max_ms=400)["holds"] is False                          # no TTFT data
    beyond = {"0.1": 10.0, "0.5": 20.0, "+Inf": 100.0}
    r = indicators(obs(ttft=beyond), 30, 1, ttft_max_ms=10_000)
    assert r["ttft_p95_beyond"] is True and r["holds"] is False
    assert indicators(obs(ttft=beyond), 30, 1)["holds"] is True                                         # limit not requested


def test_indicators_with_missing_counters():
    r = indicators(obs(output_tokens=None, prefix_cache_hits=None, mtp_proposed=0.0), 30, 1)
    assert r["output_throughput_per_gpu"] is None and r["prefix_cache"] is None and r["mtp_acceptance"] is None


def test_indicators_zero_duration():
    o = obs()
    o["end"] = o["start"]
    assert indicators(o, 30, 1)["output_throughput_per_gpu"] is None


def test_counters_and_histograms_constants_are_consistent():
    assert set(analysis.GAUGES) == {"kv", "waiting", "running"}
    assert set(analysis.HISTOGRAMS) == {"ttft", "tpot"}
    assert analysis.GAUGES["running"] == "vllm:num_requests_running"
    assert set(analysis.COUNTERS) == {"output_tokens", "input_tokens", "completed_requests", "preemptions",
                                      "prefix_cache_hits", "prefix_cache_queries", "mtp_accepted", "mtp_proposed",
                                      "failed_requests"}


def test_counter_names_are_the_ones_prometheus_stores():
    for key, name in analysis.COUNTERS.items():
        assert name.endswith("_total"), (key, name)
    assert analysis.COUNTERS["failed_requests"] == "http_requests_total"
    assert analysis.COUNTERS["completed_requests"] == "vllm:request_success_total"
    assert all(name.startswith("vllm:") for key, name in analysis.COUNTERS.items() if key != "failed_requests")


def test_litellm_names():
    assert set(analysis.LITELLM) == {"output_tokens", "input_tokens", "ttft", "latency"}
    assert analysis.LITELLM["output_tokens"] == "litellm_output_tokens_metric_total"
    assert analysis.LITELLM["input_tokens"] == "litellm_input_tokens_metric_total"
    assert not analysis.LITELLM["ttft"].endswith("_bucket") and not analysis.LITELLM["latency"].endswith("_total")


def test_removed_prometheus_text_helpers_are_gone():
    """Scraping /metrics directly was replaced by queries to Prometheus (observation.query)."""
    for name in ("fetch_series", "summarize", "summarize_litellm", "delta_litellm", "HTTP_ERRORS"):
        assert not hasattr(analysis, name)
