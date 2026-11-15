"""observation/observer.py: pure helpers, window logic against a stub, and argument validation."""
import argparse
import json
import time

import pytest

from observation import analysis, observer

METRICS = """# HELP vllm:generation_tokens_total x
vllm:generation_tokens_total{engine="0",model_name="m"} 100.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 4.0
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:num_requests_waiting{engine="0"} 1.0
vllm:num_requests_running{engine="0"} 3.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 4.0
other_metric 9
"""


def test_counter_dropped():
    before = {"u": (10.0, 5.0)}
    assert observer.counter_dropped(before, {"u": (10.0, 5.0)}) is False
    assert observer.counter_dropped(before, {"u": (11.0, 6.0)}) is False
    assert observer.counter_dropped(before, {"u": (9.0, 5.0)}) is True       # output tokens went down
    assert observer.counter_dropped(before, {"u": (10.0, 4.0)}) is True      # completed requests went down
    assert observer.counter_dropped(before, {"u": (0.0, 0.0), "new": (0.0, 0.0)}) is True
    assert observer.counter_dropped(before, {"new": (0.0, 0.0)}) is False   # a URL absent before is not a restart
    assert observer.counter_dropped({}, {}) is False


def test_snapshot_sums_urls_and_records_markers(stub):
    a, b = stub({"/metrics": (200, METRICS, {})}), stub({"/metrics": (200, METRICS, {})})
    urls = [a.url + "/metrics", b.url + "/metrics"]
    counters, histograms, gauges, samples = observer.snapshot(urls)
    assert counters["vllm:generation_tokens"] == 200.0 and counters["vllm:request_success"] == 8.0
    assert histograms["ttft"] == {"0.5": 4.0, "+Inf": 8.0}
    assert gauges["kv"] == [0.25, 0.25] and gauges["waiting"] == [1.0, 1.0] and gauges["running"] == [3.0, 3.0]
    assert samples == {urls[0]: (100.0, 4.0), urls[1]: (100.0, 4.0)}


def test_snapshot_raises_when_a_url_is_down(stub):
    s = stub({"/metrics": (200, METRICS, {}), "/down": (500, "", {})})
    with pytest.raises(OSError):
        observer.snapshot([s.url + "/metrics", s.url + "/down"])


def test_persistent_snapshot_retries_then_gives_up(monkeypatch):
    calls = []

    def flaky(urls):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("down")
        return "ok"
    monkeypatch.setattr(observer, "snapshot", flaky)
    monkeypatch.setattr(observer.time, "sleep", lambda s: None)
    assert observer.persistent_snapshot(["u"]) == "ok" and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(observer, "snapshot", lambda urls: (_ for _ in ()).throw(OSError("still down")))
    with pytest.raises(OSError):
        observer.persistent_snapshot(["u"], runs=2)


def fake_snapshot(output, completed, ttft_inf=4.0, kv=0.5, failed=None):
    counters = {"vllm:generation_tokens": output, "vllm:request_success": completed}
    if failed is not None:
        counters["http_requests"] = failed
    return (counters, {"ttft": {"0.5": ttft_inf / 2, "+Inf": ttft_inf}, "tpot": {}}, {"kv": [kv], "waiting": [2.0], "running": [3.0]},
            {"u": (output, completed)})


def window_args(**kw):
    return argparse.Namespace(step=0.01, litellm=None, by="api_key_alias", label="L", level=2, **kw)


def run_window(monkeypatch, first, middle, last, **kw):
    """observe_window with scripted snapshots: `first` at the start, `middle` for the gauge reads, `last` at the end."""
    state = {"n": 0}

    def persistent(urls, **k):
        state["n"] += 1
        return first if state["n"] == 1 else last
    monkeypatch.setattr(observer, "persistent_snapshot", persistent)
    monkeypatch.setattr(observer, "snapshot", lambda urls: middle)
    now = time.time()
    return observer.observe_window(window_args(**kw), ["u"], None, now, now + 0.05, None)


def test_observe_window_computes_deltas(monkeypatch):
    obs, litellm, stop = run_window(monkeypatch, fake_snapshot(10, 1), fake_snapshot(15, 2, kv=0.9), fake_snapshot(30, 5, 10.0))
    assert stop is False and litellm is None
    assert obs["counters"]["output_tokens"] == 20 and obs["counters"]["completed_requests"] == 4
    assert obs["counters"]["input_tokens"] is None and obs["counters"]["preemptions"] is None
    assert obs["histograms"]["ttft"] == {"0.5": 5 - 2.0, "+Inf": 10.0 - 4.0}
    assert obs["gauges"] == {"kv_max": 0.9, "waiting_max": 2.0, "running_max": 3.0}
    assert obs["counter_reset"] is False and obs["interrupted"] is False and obs["reads"] >= 1
    assert set(obs) >= {"label", "level", "start", "end", "urls", "counters", "histograms", "gauges", "failed_reads"}
    assert set(obs["counters"]) == set(analysis.COUNTERS)


def test_observe_window_flags_a_restart(monkeypatch):
    obs, _, _ = run_window(monkeypatch, fake_snapshot(100, 10), fake_snapshot(120, 12), fake_snapshot(5, 1))
    assert obs["counter_reset"] is True and obs["counters"]["output_tokens"] < 0


def test_observe_window_failed_requests_series_appearing_during_the_window(monkeypatch):
    obs, _, _ = run_window(monkeypatch, fake_snapshot(1, 1), fake_snapshot(2, 2), fake_snapshot(3, 3, failed=7.0))
    assert obs["counters"]["failed_requests"] == 7.0


def test_observe_window_counts_failed_mid_window_reads(monkeypatch):
    def boom(urls):
        raise OSError("x")
    monkeypatch.setattr(observer, "persistent_snapshot", lambda urls, **k: fake_snapshot(1, 1))
    monkeypatch.setattr(observer, "snapshot", boom)
    now = time.time()
    obs, _, _ = observer.observe_window(window_args(), ["u"], None, now, now + 0.05, None)
    assert obs["failed_reads"] >= 1 and obs["reads"] == 0 and obs["gauges"]["kv_max"] is None


def test_observe_window_stop_requested_during_final_read(monkeypatch):
    monkeypatch.setattr(observer, "REQUESTED", [True])
    obs, _, stop = run_window(monkeypatch, fake_snapshot(1, 1), fake_snapshot(2, 2), fake_snapshot(3, 3))
    assert stop is True and obs["interrupted"] is True


def run_main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["observer", *argv])
    observer.main()


@pytest.mark.parametrize("argv,message", [
    (["--metrics", "http://x/metrics", "--label", "../x"], "--label"),
    (["--metrics", "http://x/metrics", "--label=-x"], "--label"),
    (["--metrics", "http://x/metrics", "--label", "ok", "--by", "bad label"], "--by"),
    (["--metrics", "http://x/metrics", "--label", "ok", "--duration", "30"], "at least 60"),
    (["--metrics", "http://x/metrics", "--label", "ok", "--duration", "120", "--interval", "30"], "at least 60"),
    (["--metrics", "http://x/metrics", "--label", "ok", "--duration", "60", "--interval", "120"], "--duration >= --interval"),
])
def test_main_rejects_invalid_arguments(monkeypatch, tmp_path, argv, message):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *argv, "--output", str(tmp_path))
    assert message in str(e.value)
    assert list(tmp_path.iterdir()) == []


def test_main_refuses_to_overwrite_a_non_empty_directory(monkeypatch, tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "observation-1.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "--metrics", "http://127.0.0.1:1/metrics", "--label", "ok", "--output", str(tmp_path))
    assert "already exists" in str(e.value) and (tmp_path / "ok" / "observation-1.json").read_text() == "{}"


def test_main_warns_about_ignored_variables_before_validating(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CLE_API", "sk-secret")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--metrics", "http://x/metrics", "--label", "../x", "--output", str(tmp_path))
    err = capsys.readouterr().err
    assert "CLE_API" in err and "BENCH_API_KEY" in err and "sk-secret" not in err


def test_json_roundtrip_of_an_observation_is_serializable(monkeypatch):
    obs, _, _ = run_window(monkeypatch, fake_snapshot(1, 1), fake_snapshot(2, 2), fake_snapshot(3, 3))
    assert json.loads(json.dumps(obs)) == obs
