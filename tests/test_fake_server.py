"""tools/fake_server.py endpoints, through a subprocess (the module starts a server when imported)."""
import json
import urllib.error
import urllib.request
import uuid

import pytest

from observation.analysis import COUNTERS, HISTOGRAMS, fetch_series, summarize
from tests import helpers

pytest.importorskip("aiohttp")

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url, **kw):
    with OPENER.open(url, timeout=10, **kw) as r:
        return r.status, r.read().decode()


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})})
    with OPENER.open(req, timeout=20) as r:
        return r.status, r.read().decode()


def chat(url, content="hello", **body):
    """Streaming chat completion; returns the decoded events (without [DONE])."""
    payload = {"model": "bench", "messages": [{"role": "user", "content": content}], "stream": True,
               "stream_options": {"include_usage": True}, "max_tokens": 8, **body}
    _, text = post(url + "/v1/chat/completions", payload)
    lines = [x[6:] for x in text.splitlines() if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(x) for x in lines[:-1]]


def usage_of(events):
    return [e for e in events if e.get("usage")][-1]["usage"]


def test_models_and_version(fake_url):
    assert json.loads(get(fake_url + "/v1/models")[1])["data"][0]["id"] == "bench"
    assert json.loads(get(fake_url + "/version")[1])["version"] == "0.0.0-fake"


def test_streaming_completion_with_forced_length(fake_url):
    events = chat(fake_url, max_tokens=7, min_tokens=7, ignore_eos=True)
    contents = [e["choices"][0]["delta"].get("content") for e in events if e["choices"]]
    assert len([c for c in contents if c]) == 7
    assert [e["choices"][0]["finish_reason"] for e in events if e["choices"] and e["choices"][0]["finish_reason"]] == ["length"]
    u = usage_of(events)
    assert u["completion_tokens"] == 7 and u["prompt_tokens"] > 0 and u["total_tokens"] == u["prompt_tokens"] + 7
    assert "cached_tokens" in u["prompt_tokens_details"]


def test_completion_without_usage_option(fake_url):
    events = chat(fake_url, stream_options=None)
    assert not any(e.get("usage") for e in events)


def test_non_streaming_completion(fake_url):
    _, text = post(fake_url + "/v1/chat/completions", {"model": "bench", "messages": [{"role": "user", "content": "x"}]})
    j = json.loads(text)
    assert j["choices"][0]["message"]["content"] == "ok" and j["usage"]["completion_tokens"] == 1
    assert j["system_fingerprint"] == "vllm-0.0.0-tp8-ep-0a1b2c3d"


def test_prefix_cache_reports_cached_tokens_on_a_repeated_prompt(fake_url):
    prompt = uuid.uuid4().hex * 120
    first = usage_of(chat(fake_url, prompt))["prompt_tokens_details"]["cached_tokens"]
    second = usage_of(chat(fake_url, prompt))["prompt_tokens_details"]["cached_tokens"]
    assert first == 0 and second > 0 and second % 16 == 0


def test_metrics_contain_what_the_observer_reads(fake_url):
    chat(fake_url, max_tokens=5, min_tokens=5, ignore_eos=True)
    counters, histograms, gauges = summarize(fetch_series(fake_url + "/metrics"))
    for key, name in COUNTERS.items():
        assert name in counters, f"{key}: {name} missing from the fake server's /metrics"
    assert counters["vllm:generation_tokens"] >= 5 and counters["vllm:request_success"] >= 1
    assert counters["http_requests"] == 0.0
    for k in HISTOGRAMS:
        assert histograms[k]["+Inf"] >= 1
    assert gauges["kv"] == [0.5] and gauges["waiting"] == [0.0]


def test_metrics_counters_grow_with_the_requests(fake_url):
    before = summarize(fetch_series(fake_url + "/metrics"))[0]
    chat(fake_url, max_tokens=9, min_tokens=9, ignore_eos=True)
    after = summarize(fetch_series(fake_url + "/metrics"))[0]
    assert after["vllm:generation_tokens"] - before["vllm:generation_tokens"] == 9
    assert after["vllm:request_success"] - before["vllm:request_success"] == 1


def test_metrics_have_the_cache_config_and_spec_positions(fake_url):
    text = get(fake_url + "/metrics")[1]
    assert 'vllm:cache_config_info{block_size="16"' in text
    assert text.count("vllm:spec_decode_num_accepted_tokens_per_pos_total") == 3


def test_server_info_is_absent_without_dev_mode(fake_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(fake_url + "/server_info")
    assert e.value.code == 404


def test_litellm_metrics_by_key_alias(fake_url):
    key = "alias-" + uuid.uuid4().hex[:8]
    post(fake_url + "/v1/chat/completions", {"model": "bench", "stream": True, "max_tokens": 4, "metadata": {"agent": "coder"},
                                           "messages": [{"role": "user", "content": "x"}]}, {"Authorization": f"Bearer {key}"})
    text = get(fake_url + "/litellm/metrics/")[1]
    assert f'litellm_output_tokens_metric_total{{api_key_alias="{key}",metadata_agent="coder"' in text
    assert f'litellm_request_total_latency_metric_count{{api_key_alias="{key}"' in text


def test_litellm_metrics_redirect_to_the_trailing_slash(fake_url):
    class Stay(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.build_opener(urllib.request.ProxyHandler({}), Stay).open(fake_url + "/litellm/metrics")
    assert e.value.code == 307 and e.value.headers["Location"] == "/litellm/metrics/"


def test_error_rate_makes_requests_fail_and_counts_them():
    with helpers.fake_server("--error-rate", "1.0") as url:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(url + "/v1/chat/completions", {"model": "bench", "messages": []})
        assert e.value.code == 500
        counters = summarize(fetch_series(url + "/metrics"))[0]
        assert counters["http_requests"] == 1.0


def test_dev_mode_exposes_server_info_and_a_parallel_fingerprint():
    with helpers.fake_server("--dev-mode") as url:
        info = json.loads(get(url + "/server_info")[1])["vllm_config"]
        assert info["parallel_config"]["tensor_parallel_size"] == 4 and info["speculative_config"]["method"] == "mtp"
        from tools.server_config import collect
        r = collect(url, [url + "/metrics"], model="bench")
        assert r["fingerprint"]["decoded"]["data_parallel_size"] == 2
        assert r["servers"][0]["configuration"]["speculative_decoding"]["num_speculative_tokens"] == 3
        assert r["servers"][0]["speculative_tokens_metrics"] == 3


def test_custom_itl_bounds():
    with helpers.fake_server("--itl-bounds", "[0.0333, 0.1]") as url:
        chat(url, max_tokens=4, min_tokens=4, ignore_eos=True)
        histograms = summarize(fetch_series(url + "/metrics"))[1]
        assert sorted(histograms["tpot"], key=lambda x: float(x.replace("+Inf", "1e9"))) == ["0.0333", "0.1", "+Inf"]
