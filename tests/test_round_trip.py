"""Producer -> consumer round trips: what the producers write is what the reports read (catches renamed keys).

The producers run against tools/fake_server.py: the launcher (agents), tools/server_config.py (GPU deduction) and
the profile files of fixtures.aiperf_tree. The agents round trip takes about 15 s. The observation tool queries
Prometheus, which the fake server cannot stand in for: it is covered with a Prometheus stub in
tests/test_observation_query.py and tests/test_report_golden.py."""
import csv
import json
import shutil

import pytest

from tests import fixtures, helpers

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
    given = read_csv(out / "table_agents.csv") if report.returncode == 0 else None
    deduced_report = helpers.run(PY + ["-m", "report", "agents", str(out), "--target", "30"])
    return out, launcher, report, given, deduced_report


def test_launcher_ran_cleanly(agents_run):
    _, launcher, *_ = agents_run
    assert launcher.returncode == 0, launcher.stderr
    assert "[config] target fingerprint" in launcher.stdout and "2 concurrent task(s)" in launcher.stdout
    assert "[config] GPUs deduced: 8 (TP 8 × PP 1 × DP 1 × 1 vLLM server(s))" in launcher.stdout
    assert "WARNING: http" not in launcher.stdout
    assert "WARNING: server metrics not read" not in launcher.stdout


def test_launcher_writes_the_files_the_report_reads(agents_run):
    out, *_ = agents_run
    config = json.loads((out / "rt-r1" / "server_config.json").read_text())
    assert config["fingerprint"]["response"] is True and config["servers"][0]["vllm_lines"] > 0
    assert config["servers"][0]["metrics_status"] == 200
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
    out, _, report, *_ = agents_run
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


def test_report_agents_deduces_the_gpus_from_the_launcher_fingerprint(agents_run):
    out, _, _, given, deduced_report = agents_run
    assert deduced_report.returncode == 0, deduced_report.stderr
    assert deduced_report.stdout.startswith("GPUs: rt-r1: 8 (TP 8 × PP 1 × DP 1 × 1 vLLM server(s)), deduced from the fingerprint.\n")
    rows = read_csv(out / "table_agents.csv")                       # written by the run without --gpu
    assert [r["gpu"] for r in rows if r["agent"] == "all"] == ["8", "8"] and [r["gpu"] for r in rows if r["agent"] != "all"] == [""] * 4
    for deduced, one in zip(rows, given, strict=True):
        assert deduced["calls"] == one["calls"]
        if deduced["agent"] == "all":
            assert num(deduced["output_throughput_per_gpu"]) == pytest.approx(num(one["output_throughput_per_gpu"]) / 8, rel=1e-6)
    assert [r["gpu"] for r in given if r["agent"] == "all"] == ["1", "1"]
    assert "WARNING: rt-r1: --gpu 1, but the fingerprint gives 8" in agents_run[2].stdout


def test_report_agents_throughput_matches_the_logged_tokens(agents_run):
    out, _, _, given, _ = agents_run
    level = json.loads((out / "rt-r1" / "n1" / "level.json").read_text())
    t0, t1 = level["t_start"] + level["warmup"], level["t_end"]
    calls = [json.loads(x) for x in (out / "rt-r1" / "n1" / "calls.jsonl").read_text().splitlines()]
    tokens = sum(c["output_tokens"] for c in calls if t0 <= c["t_start"] + c["duration_s"] <= t1)
    row = [r for r in given if r["concurrent_tasks"] == "1" and r["agent"] == "all"][0]
    assert num(row["output_throughput_per_gpu"]) == pytest.approx(tokens / (t1 - t0), rel=1e-6)


def test_launcher_refuses_an_existing_label_without_overwrite(agents_run, fake_url):
    out, *_ = agents_run
    r = helpers.run(PY + ["-m", "agents_test.launcher", "--url", fake_url, "--label", "rt-r1", "--output", str(out)])
    assert r.returncode != 0 and "already exists" in r.stderr


@pytest.mark.parametrize("args,message", [(["--label", "ok", "--warmup", "20", "--duration", "10"], "--warmup"),
                                          (["--label", "ok", "--agents", "nope"], "Unknown agents"),
                                          (["--label", "../x"], "--label")])
def test_launcher_validates_its_arguments(tmp_path, args, message):
    r = helpers.run(PY + ["-m", "agents_test.launcher", "--url", "http://127.0.0.1:1", "--output", str(tmp_path), *args])
    assert r.returncode != 0 and message in r.stderr
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- server_config.py -> report aiperf
@pytest.mark.parametrize("fake_args,gpus,text", [([], 8, "TP 8 × PP 1 × DP 1"), (["--dev-mode"], 8, "TP 4 × PP 1 × DP 2")])
def test_server_config_snapshot_gives_the_gpu_count_to_report_aiperf(tmp_path, fake_args, gpus, text):
    root = fixtures.aiperf_tree(tmp_path / "results")
    with helpers.fake_server(*fake_args) as url:
        snapshot = tmp_path / "server_config.json"
        r = helpers.run(PY + ["tools/server_config.py", "--url", url, "--model", "bench", "--metrics", url + "/metrics",
                              "--output", str(snapshot)])
    assert r.returncode == 0, r.stderr
    assert f"[config] GPUs deduced: {gpus} ({text} × 1 vLLM server(s))" in r.stderr and "WARNING" not in r.stderr
    for run in sorted(p.name for p in root.iterdir()):
        shutil.copy(snapshot, root / run / "server_config.json")
    report = helpers.run(PY + ["-m", "report", "aiperf", str(root), "--target", "30"])
    assert report.returncode == 0, report.stderr
    assert report.stdout.count("deduced from the fingerprint.") == 4 and f"GPUs: conf-a-r1: {gpus} ({text}" in report.stdout
    rows = read_csv(root / "table.csv")
    assert {r["gpu"] for r in rows} == {str(gpus)}
    for r in rows:
        assert num(r["output_throughput_per_gpu"]) == pytest.approx(num(r["total_output_throughput"]) / gpus)


def test_report_aiperf_gpu_given_matches_the_snapshot_without_a_warning(tmp_path):
    root = fixtures.aiperf_tree(tmp_path / "results")
    with helpers.fake_server() as url:
        snapshot = tmp_path / "server_config.json"
        helpers.run(PY + ["tools/server_config.py", "--url", url, "--model", "bench", "--metrics", url + "/metrics",
                          "--output", str(snapshot)])
    for run in sorted(p.name for p in root.iterdir()):
        shutil.copy(snapshot, root / run / "server_config.json")
    ok = helpers.run(PY + ["-m", "report", "aiperf", str(root), "--target", "30", "--gpu", "8"])
    assert ok.returncode == 0 and "WARNING: conf-a-r1: --gpu" not in ok.stdout
    wrong = helpers.run(PY + ["-m", "report", "aiperf", str(root), "--target", "30", "--gpu", "2"])
    assert wrong.returncode == 0 and "WARNING: conf-a-r1: --gpu 2, but the fingerprint gives 8" in wrong.stdout


def test_server_config_warns_when_the_metrics_address_is_not_vllms(tmp_path):
    """The litellm path of the fake server answers 200 with LiteLLM metrics only: no vllm: line."""
    with helpers.fake_server() as url:
        r = helpers.run(PY + ["tools/server_config.py", "--url", url, "--model", "bench", "--metrics", url + "/litellm/metrics/",
                              "--output", str(tmp_path / "c.json")])
    assert r.returncode == 0
    assert "WARNING" in r.stderr and "is not vLLM's /metrics (no vllm: metric, LiteLLM address?)" in r.stderr
    assert "GPUs deduced" not in r.stderr
