"""tools/fake_server.py endpoints, through a subprocess (the module starts a server when imported)."""
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

from observation.analysis import COUNTERS, GAUGES, HISTOGRAMS, LITELLM
from observation.query import LITELLM_WITNESSES, VLLM_WITNESSES
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


def sample(text, name, **labels):
    """Sum of the values of the series `name` (with the given labels) in a Prometheus text; None if there is none."""
    found = []
    for m in re.finditer(rf"^{re.escape(name)}(?:\{{([^}}]*)\}})? (\S+)$", text, re.M):
        have = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1) or ""))
        if all(have.get(k) == v for k, v in labels.items()):
            found.append(float(m.group(2)))
    return sum(found) if found else None


def metrics(url):
    return get(url + "/metrics")[1]


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


def test_metrics_contain_every_series_observation_query_reads(fake_url):
    chat(fake_url, max_tokens=5, min_tokens=5, ignore_eos=True)
    text = metrics(fake_url)
    for key, name in COUNTERS.items():
        assert sample(text, name) is not None, f"{key}: {name} missing from the fake server's /metrics"
    for name in GAUGES.values():
        assert sample(text, name) is not None, f"{name} missing"
    assert set(HISTOGRAMS.values()) == {"vllm:time_to_first_token_seconds", "vllm:request_time_per_output_token_seconds"}
    for name in HISTOGRAMS.values():
        assert sample(text, name + "_bucket", le="+Inf") >= 1 and sample(text, name + "_count") >= 1
        assert sample(text, name + "_sum") is not None
    for name in VLLM_WITNESSES:
        assert re.search(rf"^{re.escape(name)}[{{ ]", text, re.M), f"restart witness {name} missing"
    assert sample(text, COUNTERS["output_tokens"]) >= 5 and sample(text, COUNTERS["completed_requests"]) >= 1
    assert sample(text, "http_requests_total", handler="/v1/chat/completions", status="2xx") >= 1
    assert sample(text, GAUGES["kv"]) == 0.5 and sample(text, GAUGES["waiting"]) == 0.0


def test_metrics_counters_are_named_with_total_like_prometheus_stores_them(fake_url):
    text = metrics(fake_url)
    assert "vllm:generation_tokens_total{" in text and "vllm:request_success_total{" in text
    assert re.search(r"^vllm:generation_tokens\{", text, re.M) is None


def test_metrics_counters_grow_with_the_requests(fake_url):
    before = metrics(fake_url)
    chat(fake_url, max_tokens=9, min_tokens=9, ignore_eos=True)
    after = metrics(fake_url)
    assert sample(after, COUNTERS["output_tokens"]) - sample(before, COUNTERS["output_tokens"]) == 9
    assert sample(after, COUNTERS["completed_requests"]) - sample(before, COUNTERS["completed_requests"]) == 1


def test_running_requests_gauge_is_zero_when_idle_and_counts_streams_in_flight():
    with helpers.fake_server("--token-delay", "0.05") as url:
        assert sample(metrics(url), GAUGES["running"]) == 0.0
        done = threading.Thread(target=lambda: chat(url, max_tokens=40, min_tokens=40, ignore_eos=True), daemon=True)
        done.start()
        deadline = time.time() + 10
        seen = 0.0
        while time.time() < deadline and not seen:
            seen = sample(metrics(url), GAUGES["running"]) or 0.0
            time.sleep(0.05)
        done.join(timeout=20)
        assert seen >= 1.0
        assert sample(metrics(url), GAUGES["running"]) == 0.0


def test_metrics_have_the_cache_config_and_spec_positions(fake_url):
    text = get(fake_url + "/metrics")[1]
    assert 'vllm:cache_config_info{block_size="16"' in text
    assert text.count("vllm:spec_decode_num_accepted_tokens_per_pos_total") == 3


def test_server_info_is_absent_without_dev_mode(fake_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(fake_url + "/server_info")
    assert e.value.code == 404


def test_litellm_metrics_have_the_names_the_agents_view_queries(fake_url):
    key = "alias-" + uuid.uuid4().hex[:8]
    post(fake_url + "/v1/chat/completions", {"model": "bench", "stream": True, "max_tokens": 4, "ignore_eos": True, "metadata": {"agent": "coder"},
                                           "messages": [{"role": "user", "content": "x"}]}, {"Authorization": f"Bearer {key}"})
    text = get(fake_url + "/litellm/metrics/")[1]
    assert sample(text, LITELLM["output_tokens"], api_key_alias=key, metadata_agent="coder") == 4.0
    assert sample(text, LITELLM["input_tokens"], api_key_alias=key) > 0
    for k in ("ttft", "latency"):
        assert sample(text, LITELLM[k] + "_bucket", api_key_alias=key, le="+Inf") == 1.0
        assert sample(text, LITELLM[k] + "_count", api_key_alias=key) == 1.0
        assert sample(text, LITELLM[k] + "_sum", api_key_alias=key) > 0
    for name in LITELLM_WITNESSES:
        assert re.search(rf"^{re.escape(name)}\{{", text, re.M), f"restart witness {name} missing"


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
        assert sample(metrics(url), "http_requests_total", status="5xx") == 1.0


def test_admin_changes_the_error_rate_while_the_server_runs():
    with helpers.fake_server() as url:
        assert json.loads(get(url + "/admin?rate=1")[1]) == {"rate": 1.0}
        with pytest.raises(urllib.error.HTTPError) as e:
            chat(url)
        assert e.value.code == 500
        assert json.loads(get(url + "/admin")[1]) == {"rate": 1.0}               # no parameter: unchanged
        assert json.loads(get(url + "/admin?rate=0")[1]) == {"rate": 0.0}
        assert usage_of(chat(url))["completion_tokens"] > 0
        text = metrics(url)
        assert sample(text, "http_requests_total", status="5xx") == 1.0 and sample(text, "http_requests_total", status="2xx") == 1.0


def test_http_series_appear_only_with_the_first_request():
    with helpers.fake_server() as url:
        assert sample(metrics(url), "http_requests_total", status="2xx") == 0.0
        assert sample(metrics(url), "http_requests_total", status="5xx") is None


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
        text = metrics(url)
        bounds = re.findall(r'vllm:request_time_per_output_token_seconds_bucket\{[^}]*le="([^"]+)"', text)
        assert sorted(bounds, key=lambda x: float(x.replace("+Inf", "1e9"))) == ["0.0333", "0.1", "+Inf"]
