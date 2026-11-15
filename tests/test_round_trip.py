"""Producer -> consumer round trips: what the producers write is what the reports read (catches renamed keys).

The producers run against tools/fake_server.py. The agents round trip takes about 15 s; the 60 s observer run only
happens with RUN_SLOW=1."""
import csv
import json
import os
import threading
import time

import pytest

from observation import observer
from observation.analysis import COUNTERS, HISTOGRAMS, fetch_series, summarize_litellm
from tests import helpers

pytest.importorskip("aiohttp")
pytest.importorskip("httpx")
pytest.importorskip("openai")

PY = [helpers.sys.executable, "-B"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(x):
    return None if x in ("", None) else float(x)


# ---------------------------------------------------------------- agents launcher -> report agents
@pytest.fixture(scope="module")
def agents_run(fake_url, tmp_path_factory):
    out = tmp_path_factory.mktemp("agents")
    launcher = helpers.run(PY + ["-m", "agents_test.launcher", "--url", fake_url, "--model", "bench", "--agents", "code_review,pentest",
                                 "--levels", "1 2", "--duration", "8", "--warmup", "1", "--scale", "0.05", "--latency-factor", "0.02",
                                 "--spread", "1", "--label", "rt-r1", "--output", str(out)], timeout=180)
    report = helpers.run(PY + ["-m", "report", "agents", str(out), "--target", "30", "--gpu", "1"])
    return out, launcher, report


def test_launcher_ran_cleanly(agents_run):
    _, launcher, _ = agents_run
    assert launcher.returncode == 0, launcher.stderr
    assert "[config] target fingerprint" in launcher.stdout and "2 concurrent task(s)" in launcher.stdout
    assert "WARNING: server metrics not read" not in launcher.stdout


def test_launcher_writes_the_files_the_report_reads(agents_run):
    out, _, _ = agents_run
    assert json.loads((out / "rt-r1" / "server_config.json").read_text())["fingerprint"]["response"] is True
    for n in (1, 2):
        d = out / "rt-r1" / f"n{n}"
        level = json.loads((d / "level.json").read_text())
        assert level["concurrent_tasks"] == n and level["agents"] == ["code_review", "pentest"]
        assert level["t_end"] - level["t_start"] == pytest.approx(8) and level["warmup"] == 1 and level["mode"] == "scripted"
        assert set(level["metrics"]) >= {"preemptions", "prefix_cache_hits", "prefix_cache_queries", "mtp_accepted", "mtp_proposed",
                                         "kv_cache_max_fraction", "max_waiting_requests", "reads", "failed_reads"}
        assert level["metrics"]["reads"] >= 2 and level["metrics"]["failed_reads"] == 0
        calls = [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
        tasks = [json.loads(x) for x in (d / "tasks.jsonl").read_text().splitlines()]
        assert calls and tasks
        for c in calls:
            assert set(c) >= {"agent", "task", "t_start", "duration_s", "ttft_s", "input_tokens", "output_tokens",
                              "cached_tokens", "speed", "tool_call_count", "error"}
            assert c["error"] is None and c["output_tokens"] > 0 and c["input_tokens"] > 0
        for t in tasks:
            assert set(t) >= {"agent", "task", "t_start", "t_end", "error"} and t["t_end"] >= t["t_start"]


def test_report_agents_reads_the_launcher_output(agents_run):
    out, _, report = agents_run
    assert report.returncode == 0, report.stderr
    rows = read_csv(out / "table_agents.csv")
    assert {(r["concurrent_tasks"], r["agent"]) for r in rows} == {(n, a) for n in ("1", "2") for a in ("all", "code_review", "pentest")}
    for r in rows:
        assert r["run"] == "rt-r1" and num(r["calls"]) > 0 and num(r["call_errors"]) == 0 and num(r["tasks_failed"]) == 0
        assert num(r["output_throughput_per_gpu"]) > 0
        assert 0 < num(r["interactivity_p10"]) < 2000 and num(r["interactivity_p50"]) >= num(r["interactivity_p10"])
        assert 0 < num(r["ttft_p50_s"]) <= num(r["ttft_p95_s"]) < 10
        assert 0 <= num(r["cache_rate"]) <= 1 and num(r["max_context"]) > 0
    for r in (r for r in rows if r["agent"] == "all"):
        assert num(r["preemptions"]) == 0 and num(r["kv_cache_max_fraction"]) == 0.5 and num(r["max_waiting_requests"]) == 0
        assert 0 < num(r["server_prefix_cache"]) <= 1 and num(r["mtp_acceptance"]) == pytest.approx(2 / 3, abs=0.05)
    for r in (r for r in rows if r["agent"] != "all"):
        assert r["preemptions"] == "" and r["mtp_acceptance"] == ""           # server metrics only on the "all" row
    assert "selected level" in report.stdout or "no level reaches" in report.stdout


def test_report_agents_throughput_matches_the_logged_tokens(agents_run):
    out, _, _ = agents_run
    level = json.loads((out / "rt-r1" / "n1" / "level.json").read_text())
    t0, t1 = level["t_start"] + level["warmup"], level["t_end"]
    calls = [json.loads(x) for x in (out / "rt-r1" / "n1" / "calls.jsonl").read_text().splitlines()]
    tokens = sum(c["output_tokens"] for c in calls if t0 <= c["t_start"] + c["duration_s"] <= t1)
    row = [r for r in read_csv(out / "table_agents.csv") if r["concurrent_tasks"] == "1" and r["agent"] == "all"][0]
    assert num(row["output_throughput_per_gpu"]) == pytest.approx(tokens / (t1 - t0), rel=1e-6)


def test_launcher_refuses_an_existing_label_without_overwrite(agents_run, fake_url):
    out, _, _ = agents_run
    r = helpers.run(PY + ["-m", "agents_test.launcher", "--url", fake_url, "--label", "rt-r1", "--output", str(out)])
    assert r.returncode != 0 and "already exists" in r.stderr


@pytest.mark.parametrize("args,message", [(["--label", "ok", "--warmup", "20", "--duration", "10"], "--warmup"),
                                          (["--label", "ok", "--agents", "nope"], "Unknown agents"),
                                          (["--label", "../x"], "--label")])
def test_launcher_validates_its_arguments(tmp_path, args, message):
    r = helpers.run(PY + ["-m", "agents_test.launcher", "--url", "http://127.0.0.1:1", "--output", str(tmp_path), *args])
    assert r.returncode != 0 and message in r.stderr
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- observer -> report observation
def load_for(url, key, stop):
    """Background traffic so that the counters move during an observation window."""
    def run():
        i = 0
        while not stop.is_set():
            i += 1
            try:
                body = {"model": "bench", "stream": True, "max_tokens": 6, "min_tokens": 6, "ignore_eos": True,
                        "metadata": {"agent": "coder"}, "messages": [{"role": "user", "content": f"hello {i}"}]}
                import urllib.request
                req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10).read()
            except OSError:
                pass
            time.sleep(0.02)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def check_observation(out, label, litellm=True):
    (file,) = sorted((out / label).glob("observation-*.json"))
    obs = json.loads(file.read_text())
    assert obs["label"] == label and obs["end"] > obs["start"] and obs["counter_reset"] is False and obs["interrupted"] is False
    assert set(obs["counters"]) == set(COUNTERS) and set(obs["histograms"]) == set(HISTOGRAMS)
    assert obs["counters"]["output_tokens"] > 0 and obs["counters"]["completed_requests"] > 0
    assert obs["counters"]["failed_requests"] == 0 and obs["counters"]["preemptions"] == 0
    assert obs["counters"]["mtp_accepted"] > 0 and obs["counters"]["prefix_cache_queries"] > 0
    for k in HISTOGRAMS:
        assert obs["histograms"][k]["+Inf"] > 0 and all(v >= 0 for v in obs["histograms"][k].values())
    assert obs["gauges"]["kv_max"] == 0.5
    if litellm:
        assert obs["agents_by"] == "api_key_alias" and "key-a" in obs["agents"]
        assert obs["agents"]["key-a"]["output_tokens"] > 0 and obs["agents"]["key-a"]["latency_count"] > 0
    return obs


def check_report_observation(out, label, litellm=True):
    r = helpers.run(PY + ["-m", "report", "observation", str(out), "--target", "30", "--gpu", "1"])
    assert r.returncode == 0, r.stderr
    (row,) = read_csv(out / "table_observation.csv")
    assert row["window"] == label and row["configuration"] == label and row["level"] == "2"
    assert num(row["completed_requests"]) > 0 and num(row["output_throughput_per_gpu"]) > 0
    assert num(row["duration_s"]) > 0 and row["discarded"] == "False"
    assert 0 <= num(row["share_of_requests_at_target"]) <= 1 and num(row["preemptions"]) == 0 and num(row["failed_requests"]) == 0
    assert num(row["kv_cache_max_fraction"]) == 0.5 and num(row["mtp_acceptance"]) == pytest.approx(2 / 3, abs=0.01)
    assert row["verdict"] in ("holds", "does not hold", "undetermined")
    assert num(row["estimated_ttft_p95_ms"]) > 0 and num(row["estimated_interactivity_p10"]) > 0
    if litellm:
        agents = helpers.run(PY + ["-m", "report", "observation", str(out), "--target", "30", "--gpu", "1", "--view", "agents"])
        assert agents.returncode == 0 and "key-a" in agents.stdout and "agents distinguished by api_key_alias" in agents.stdout
    period = helpers.run(PY + ["-m", "report", "observation", str(out), "--target", "30", "--gpu", "1", "--view", "period"])
    assert period.returncode == 0 and "verdict" in period.stdout


def test_observe_window_to_report_observation(fake_url, tmp_path):
    """Fast version of the observer round trip: one 3-second window driven in-process."""
    import argparse
    p = argparse.Namespace(step=0.5, litellm=fake_url + "/litellm/metrics/", by="api_key_alias", label="fast", level=2)
    litellm0 = summarize_litellm(fetch_series(fake_url + "/litellm/metrics/", prefix="litellm_", key="key-a"), "api_key_alias")
    stop = threading.Event()
    thread = load_for(fake_url, "key-a", stop)
    try:
        start = time.time()
        obs, _, interrupted = observer.observe_window(p, [fake_url + "/metrics"], "key-a", start, start + 3, litellm0)
    finally:
        stop.set()
        thread.join()
    assert interrupted is False
    (tmp_path / "fast").mkdir()
    (tmp_path / "fast" / "observation-20260101-000000.json").write_text(json.dumps(obs))
    check_observation(tmp_path, "fast")
    check_report_observation(tmp_path, "fast")


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="takes about 65 s: set RUN_SLOW=1")
def test_observer_process_to_report_observation(fake_url, tmp_path):
    stop = threading.Event()
    thread = load_for(fake_url, "key-a", stop)
    try:
        r = helpers.run(PY + ["-m", "observation.observer", "--metrics", fake_url + "/metrics", "--duration", "60", "--label", "slow",
                              "--level", "2", "--output", str(tmp_path), "--litellm", fake_url + "/litellm/metrics/",
                              "--url", fake_url, "--model", "bench", "--target", "30", "--gpu", "1"],
                        env=helpers.clean_environment(BENCH_API_KEY="key-a"), timeout=180)
    finally:
        stop.set()
        thread.join()
    assert r.returncode == 0, r.stderr
    assert "value(s) of api_key_alias seen" in r.stdout and "Done ->" in r.stdout
    assert json.loads((tmp_path / "slow" / "server_config.json").read_text())["fingerprint"]["response"] is True
    check_observation(tmp_path, "slow")
    check_report_observation(tmp_path, "slow")
