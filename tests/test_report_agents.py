import pytest

from report.agents import percentile, summary


def call(t, duration=1.0, output=100, input_tokens=1000, cached=500, speed=50.0, ttft=0.2, error=None):
    return {"t_start": t, "duration_s": duration, "output_tokens": output, "input_tokens": input_tokens,
            "cached_tokens": cached, "speed": speed, "ttft_s": ttft, "error": error}


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.9) == 5
    assert percentile([], 0.5) is None
    assert percentile([None, 3, 1], 0.0) == 1


def test_summary_filters_on_the_end_of_the_call():
    calls = [call(0.0), call(5.0), call(9.5, duration=1.0)]  # the last one ends at 10.5 > t1
    s = summary(calls, 1.5, 10.0, n_gpu=1)
    assert s["calls"] == 1


def test_summary_throughput_per_gpu_and_cache():
    calls = [call(2.0, output=100), call(3.0, output=300)]
    s = summary(calls, 0.0, 10.0, n_gpu=2)
    assert s["output_throughput_per_gpu"] == pytest.approx(400 / 10 / 2)
    assert s["cache_rate"] == pytest.approx(0.5)
    assert s["max_context"] == 1300


def test_summary_excludes_errors_from_measurements():
    calls = [call(2.0), call(3.0, error="timeout", output=None)]
    s = summary(calls, 0.0, 10.0, n_gpu=1)
    assert s["calls"] == 2 and s["call_errors"] == 1
    assert s["output_throughput_per_gpu"] == pytest.approx(10.0)


def test_summary_empty_window():
    s = summary([], 0.0, 10.0, n_gpu=1)
    assert s["calls"] == 0 and s["ttft_p50_s"] is None and s["max_context"] is None
    assert s["cache_rate"] is None


def test_summary_speed_ignores_short_responses():
    calls = [call(1.0, output=8, speed=999.0), call(2.0, output=64, speed=40.0)]
    assert summary(calls, 0.0, 10.0, n_gpu=1)["interactivity_p50"] == 40.0


# ---------------------------------------------------------------- autonomous agents that stop early
def level(tmp_path, run, mode, tasks):
    """A level folder with `tasks` = [(agent, number of calls, error)]; returns its level.json."""
    import json
    d = tmp_path / run / "n1"
    d.mkdir(parents=True)
    (d / "level.json").write_text(json.dumps({"mode": mode, "concurrent_tasks": 1}))
    task_rows, call_rows = [], []
    for i, (agent, n, error) in enumerate(tasks):
        task_rows.append({"agent": agent, "task": f"t{i}", "error": error})
        call_rows += [{"task": f"t{i}", "agent": agent} for _ in range(n)]
    (d / "tasks.jsonl").write_text("\n".join(json.dumps(x) for x in task_rows))
    (d / "calls.jsonl").write_text("\n".join(json.dumps(x) for x in call_rows))
    return d / "level.json"


def test_early_stop_warns_when_autonomous_agents_stop_after_one_step(tmp_path):
    from report.agents import early_stop_warnings
    f = level(tmp_path, "r2", "autonomous", [("code_review", 5, None), ("pentest", 2, None), ("report", 2, None)])
    messages = early_stop_warnings([f])
    assert len(messages) == 2 and "code_review made 5 calls per task instead of 20" in messages[0]
    assert "pentest made 2 calls per task instead of 16" in messages[1] and "scripted mode" in messages[1]
    assert not any("report" in m.split(":")[2] for m in messages)      # the report profile makes no tool call


def test_early_stop_is_silent_for_scripted_runs_and_for_complete_autonomous_runs(tmp_path):
    from report.agents import early_stop_warnings
    assert early_stop_warnings([level(tmp_path, "s", "scripted", [("pentest", 2, None)])]) == []
    assert early_stop_warnings([level(tmp_path, "a", "autonomous", [("pentest", 16, None), ("code_review", 20, None)])]) == []


def test_early_stop_ignores_failed_tasks_and_old_level_files(tmp_path):
    from report.agents import early_stop_warnings
    assert early_stop_warnings([level(tmp_path, "e", "autonomous", [("pentest", 1, "boom")])]) == []
    d = tmp_path / "old" / "n1"
    d.mkdir(parents=True)
    (d / "level.json").write_text("{}")      # no "mode": a run from before the mode was recorded
    assert early_stop_warnings([d / "level.json"]) == []
