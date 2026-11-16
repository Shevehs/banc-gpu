import json

import pytest

from tools.make_traces import BLOCK, PRESETS, calls_of, main, make_set, summarize

SESSIONS = 8


@pytest.fixture(scope="module", params=sorted(PRESETS))
def preset_set(request):
    preset = PRESETS[request.param]
    return preset, make_set(preset, SESSIONS, seed=3)


def test_same_seed_gives_the_same_set():
    preset = PRESETS["tiny"]
    assert make_set(preset, 3, seed=5) == make_set(preset, 3, seed=5)


def test_another_seed_gives_another_set():
    preset = PRESETS["tiny"]
    assert make_set(preset, 3, seed=5) != make_set(preset, 3, seed=6)


def test_input_is_64_tokens_per_hash_id(preset_set):
    _, traces = preset_set
    for trace in traces:
        main_calls, sub_calls = calls_of(trace)
        for call in main_calls + sub_calls:
            assert call["in"] == BLOCK * len(call["hash_ids"])


def test_no_call_goes_above_the_limit(preset_set):
    preset, traces = preset_set
    for trace in traces:
        main_calls, sub_calls = calls_of(trace)
        assert all(c["in"] + c["out"] <= preset.max_context for c in main_calls + sub_calls)


def test_the_first_session_reaches_the_limit(preset_set):
    preset, traces = preset_set
    assert summarize(traces)["longest_call"] >= 0.6 * preset.max_context


def test_most_calls_extend_the_previous_context(preset_set):
    _, traces = preset_set
    pairs = extended = 0
    for trace in traces:
        main_calls, _ = calls_of(trace)
        for a, b in zip(main_calls[1:], main_calls[2:], strict=False):      # the first call is a small one of its own
            pairs += 1
            extended += b["hash_ids"][:len(a["hash_ids"])] == a["hash_ids"]
    assert extended / pairs > 0.8


def test_first_call_is_small_and_the_second_jumps(preset_set):
    _, traces = preset_set
    for trace in traces:
        first, second = calls_of(trace)[0][:2]
        assert first["in"] <= 12 * BLOCK
        assert second["in"] > first["in"] * 1.5


def test_main_calls_are_in_time_order(preset_set):
    _, traces = preset_set
    for trace in traces:
        times = [c["t"] for c in calls_of(trace)[0]]
        assert times == sorted(times)


def test_calls_are_mostly_streaming(preset_set):
    _, traces = preset_set
    kinds = [c["type"] for t in traces for c in calls_of(t)[0]]
    assert set(kinds) <= {"n", "s"} and kinds.count("s") / len(kinds) > 0.9


def test_streaming_calls_carry_a_ttft_and_normal_ones_do_not(preset_set):
    _, traces = preset_set
    for trace in traces:
        for call in calls_of(trace)[0] + calls_of(trace)[1]:
            assert ("ttft" in call) == (call["type"] == "s")


def test_subagents_start_from_a_shared_prefix():
    traces = make_set(PRESETS["medium"], 20, seed=2)
    groups = [r for t in traces for r in t["requests"] if r["type"] == "subagent"]
    assert groups
    for trace in traces:
        subs = [r for r in trace["requests"] if r["type"] == "subagent"]
        if len(subs) >= 2:
            firsts = [s["requests"][0]["hash_ids"] for s in subs]
            common = min(len(f) for f in firsts) - 8
            assert all(f[:common] == firsts[0][:common] for f in firsts)


def test_subagent_entries_are_complete():
    traces = make_set(PRESETS["small"], 20, seed=2)
    entries = [r for t in traces for r in t["requests"] if r["type"] == "subagent"]
    for e in entries:
        assert {"t", "agent_id", "subagent_type", "duration_ms", "total_tokens", "status", "requests",
                "models"} <= set(e)
        assert e["total_tokens"] == sum(c["in"] + c["out"] for c in e["requests"])
        assert e["duration_ms"] > 0


def test_no_subagents_option():
    traces = make_set(PRESETS["small"], 10, seed=2, subagents=False)
    assert all(r["type"] != "subagent" for t in traces for r in t["requests"])


def test_a_custom_limit_replaces_the_preset():
    traces = make_set(PRESETS["small"], 5, seed=2, max_context=10_000)
    assert summarize(traces)["longest_call"] <= 10_000


def test_the_traces_follow_aiperfs_model(preset_set):
    models = pytest.importorskip("aiperf.dataset.loader.weka_trace_models")
    _, traces = preset_set
    for trace in traces:
        models.WekaTrace.model_validate(trace)


def test_command_writes_one_file_per_session(tmp_path, capsys):
    out = tmp_path / "traces"
    assert main(["--preset", "tiny", "--output", str(out), "--sessions", "4"]) == 0
    files = sorted(out.glob("*.json"))
    assert [f.name for f in files] == [f"trace-000{i}.json" for i in range(1, 5)]
    assert json.loads(files[0].read_text())["block_size"] == BLOCK
    said = capsys.readouterr().out
    assert "4 sessions" in said and "--max-model-len" in said and "BENCH_AGENTX_DIR" in said


def test_command_refuses_a_folder_that_holds_traces(tmp_path):
    out = tmp_path / "traces"
    main(["--preset", "tiny", "--output", str(out), "--sessions", "1"])
    with pytest.raises(SystemExit, match="already holds traces"):
        main(["--preset", "tiny", "--output", str(out), "--sessions", "1"])
    assert main(["--preset", "tiny", "--output", str(out), "--sessions", "1", "--overwrite"]) == 0


@pytest.mark.parametrize("argv, message", [
    (["--sessions", "0"], "--sessions"),
    (["--max-context", "1000"], "--max-context"),
])
def test_command_rejects_bad_values(tmp_path, argv, message):
    with pytest.raises(SystemExit, match=message):
        main(["--output", str(tmp_path / "t")] + argv)
