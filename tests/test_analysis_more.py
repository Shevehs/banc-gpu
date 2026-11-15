"""observation/analysis.py: fetching, LiteLLM per-agent view, indicators and verdict branches."""
import math
import urllib.error

import pytest

from observation import analysis
from observation.analysis import (agent_indicators, cumulative, delta_litellm, fetch_series, indicators, quantile,
                                  summarize_litellm)

TEXT = """# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="a\\"b"} 12.5
vllm:kv_cache_usage_perc 0.5
vllm:bad_value{x="1"} notanumber
http_requests_total{handler="/v1/chat/completions",status="5xx"} 2.0
litellm_output_tokens_metric_total{api_key_alias="k1"} 7.0
unrelated_metric 3
"""


def test_fetch_series_filters_by_prefix_and_strips_total(stub):
    s = stub({"/metrics": (200, TEXT, {})})
    series = fetch_series(s.url + "/metrics")
    assert ("vllm:generation_tokens", {"engine": "0", "model_name": 'a\\"b'}, 12.5) in series
    assert ("vllm:kv_cache_usage_perc", {}, 0.5) in series
    assert ("http_requests", {"handler": "/v1/chat/completions", "status": "5xx"}, 2.0) in series
    names = [n for n, _, _ in series]
    assert "vllm:bad_value" not in names and "unrelated_metric" not in names and "litellm_output_tokens_metric" not in names


def test_fetch_series_litellm_prefix_and_bearer_key(stub):
    s = stub({"/metrics/": (200, TEXT, {})})
    series = fetch_series(s.url + "/metrics/", prefix="litellm_", key="sk-1")
    assert series == [("litellm_output_tokens_metric", {"api_key_alias": "k1"}, 7.0)]
    assert s.seen[0]["headers"]["Authorization"] == "Bearer sk-1"


def test_fetch_series_sends_no_key_by_default(stub):
    s = stub({"/metrics": (200, TEXT, {})})
    fetch_series(s.url + "/metrics")
    assert "Authorization" not in s.seen[0]["headers"]


def test_fetch_series_follows_only_the_trailing_slash_redirect(stub):
    s = stub({"/metrics": (307, "", {"Location": "/metrics/"}), "/metrics/": (200, TEXT, {})})
    assert fetch_series(s.url + "/metrics")
    assert [r["path"] for r in s.seen] == ["/metrics", "/metrics/"]


@pytest.mark.parametrize("location", ["/elsewhere", "http://example.invalid/metrics/", "/metrics//", "/metrics/?x=1"])
def test_fetch_series_refuses_other_redirects(stub, location):
    s = stub({"/metrics": (302, "", {"Location": location}), "/elsewhere": (200, TEXT, {})})
    with pytest.raises(urllib.error.HTTPError):
        fetch_series(s.url + "/metrics", key="sk-1")
    assert all(r["path"] != "/elsewhere" for r in s.seen)


def test_fetch_series_http_error(stub):
    with pytest.raises(urllib.error.HTTPError):
        fetch_series(stub({}).url + "/metrics")


def test_cumulative_sorts_numerically_with_infinity_last():
    assert [b for b, _ in cumulative({"+Inf": 9.0, "10": 5.0, "0.5": 1.0})] == [0.5, 10.0, math.inf]
    assert cumulative({"Inf": 1.0, "inf": 2.0}) == [(math.inf, 1.0), (math.inf, 2.0)]


def test_quantile_of_a_single_infinite_bucket():
    assert quantile({"+Inf": 4.0}, 0.5) == (0.0, True)


# ---------------------------------------------------------------- LiteLLM per-agent view
def litellm_series():
    s = []
    for alias, out, inp, ttft, lat in (("a", 100.0, 500.0, 0.2, 2.0), ("b", 50.0, 100.0, 0.1, 1.0)):
        lab = {"api_key_alias": alias}
        s += [("litellm_output_tokens_metric", lab, out), ("litellm_input_tokens_metric", lab, inp),
              ("litellm_llm_api_time_to_first_token_metric_bucket", {**lab, "le": "0.5"}, 4.0),
              ("litellm_llm_api_time_to_first_token_metric_bucket", {**lab, "le": "+Inf"}, 4.0),
              ("litellm_llm_api_time_to_first_token_metric_sum", lab, ttft * 4),
              ("litellm_llm_api_time_to_first_token_metric_count", lab, 4.0),
              ("litellm_request_total_latency_metric_bucket", {**lab, "le": "5"}, 4.0),
              ("litellm_request_total_latency_metric_bucket", {**lab, "le": "+Inf"}, 4.0),
              ("litellm_request_total_latency_metric_sum", lab, lat * 4),
              ("litellm_request_total_latency_metric_count", lab, 4.0)]
    return s


def test_summarize_litellm_groups_by_label():
    res = summarize_litellm(litellm_series(), "api_key_alias")
    assert sorted(res) == ["a", "b"]
    a = res["a"]
    assert a["output_tokens"] == 100.0 and a["input_tokens"] == 500.0
    assert a["ttft"] == {"0.5": 4.0, "+Inf": 4.0} and a["ttft_count"] == 4.0 and a["ttft_sum"] == pytest.approx(0.8)
    assert a["latency"] == {"5": 4.0, "+Inf": 4.0} and a["latency_count"] == 4.0 and a["latency_sum"] == pytest.approx(8.0)


def test_summarize_litellm_sums_series_of_the_same_agent_and_skips_unlabelled():
    series = [("litellm_output_tokens_metric", {"api_key_alias": "a", "model": "m1"}, 1.0),
              ("litellm_output_tokens_metric", {"api_key_alias": "a", "model": "m2"}, 2.0),
              ("litellm_output_tokens_metric", {"model": "m1"}, 99.0)]
    assert summarize_litellm(series, "api_key_alias")["a"]["output_tokens"] == 3.0
    assert list(summarize_litellm(series, "api_key_alias")) == ["a"]
    assert summarize_litellm(series, "team_alias") == {}


def test_summarize_litellm_ignores_unknown_metric_names():
    res = summarize_litellm([("litellm_other", {"api_key_alias": "a"}, 5.0)], "api_key_alias")
    assert res["a"]["output_tokens"] == 0.0 and res["a"]["ttft"] == {}


def test_delta_litellm_subtracts_and_counts_new_agents_from_zero():
    start = summarize_litellm(litellm_series(), "api_key_alias")
    series_end = litellm_series() + [("litellm_output_tokens_metric", {"api_key_alias": "a"}, 10.0),
                                     ("litellm_request_total_latency_metric_count", {"api_key_alias": "a"}, 2.0),
                                     ("litellm_request_total_latency_metric_bucket", {"api_key_alias": "a", "le": "+Inf"}, 2.0),
                                     ("litellm_output_tokens_metric", {"api_key_alias": "new"}, 5.0)]
    end = summarize_litellm(series_end, "api_key_alias")
    d = delta_litellm(start, end)
    assert sorted(d) == ["a", "new"]                    # "b" had no activity in the window
    assert d["a"]["output_tokens"] == 10.0 and d["a"]["latency_count"] == 2.0
    assert d["a"]["latency"]["+Inf"] == 2.0 and d["a"]["latency"]["5"] == 0.0
    assert d["new"]["output_tokens"] == 5.0


def test_delta_litellm_empty():
    assert delta_litellm({}, {}) == {}
    assert delta_litellm({"a": {}}, {}) == {}


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
    assert analysis.COUNTERS["failed_requests"] == analysis.HTTP_ERRORS
    assert set(analysis.HISTOGRAMS) == {"ttft", "tpot"}
