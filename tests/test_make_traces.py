import json
import sys

import pytest

from tools.make_traces import BLOCK, PRESETS, calls_of, main, make_set, summarize, validate

SESSIONS = 8
TOLERANCE = 0.01          # seconds: the traces round times to the millisecond


@pytest.fixture(scope="module", params=sorted(PRESETS))
def preset_set(request):
    preset = PRESETS[request.param]
    return preset, make_set(preset, SESSIONS, seed=3)


def all_calls(trace):
    main_calls, sub_calls = calls_of(trace)
    return main_calls + sub_calls


def test_same_seed_gives_the_same_set():
    preset = PRESETS["tiny"]
    assert make_set(preset, 3, seed=5) == make_set(preset, 3, seed=5)


def test_another_seed_gives_another_set():
    preset = PRESETS["tiny"]
    assert make_set(preset, 3, seed=5) != make_set(preset, 3, seed=6)


def test_input_is_64_tokens_per_hash_id(preset_set):
    _, traces = preset_set
    for trace in traces:
        for call in all_calls(trace):
            assert call["in"] == BLOCK * len(call["hash_ids"])


def test_no_call_goes_above_the_limit(preset_set):
    preset, traces = preset_set
    for trace in traces:
        assert all(c["in"] + c["out"] <= preset.max_context for c in all_calls(trace))


@pytest.mark.parametrize("name", sorted(PRESETS))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_first_session_gets_close_to_the_limit(name, seed):
    preset = PRESETS[name]
    longest = summarize(make_set(preset, 1, seed))["longest_call"]
    assert 0.7 * preset.max_context <= longest <= preset.max_context


def test_most_calls_extend_the_previous_context(preset_set):
    _, traces = preset_set
    pairs = extended = 0
    for trace in traces:
        main_calls, _ = calls_of(trace)
        for a, b in zip(main_calls[1:], main_calls[2:], strict=False):   # the first call is a small one of its own
            pairs += 1
            extended += b["hash_ids"][:len(a["hash_ids"])] == a["hash_ids"]
    assert extended / pairs > 0.8


def test_a_hash_id_always_stands_for_the_same_block(preset_set):
    """An ID has one position and one prefix, in the main agent and in its sub-agents alike."""
    _, traces = preset_set
    for trace in traces:
        seen = {}
        for call in all_calls(trace):
            assert len(set(call["hash_ids"])) == len(call["hash_ids"])
            prefix = 0
            for position, block in enumerate(call["hash_ids"]):
                prefix = hash((prefix, block))
                assert seen.setdefault(block, (position, prefix)) == (position, prefix)


def test_first_call_is_small_and_the_second_jumps(preset_set):
    _, traces = preset_set
    for trace in traces:
        first, second = calls_of(trace)[0][:2]
        assert first["in"] <= 12 * BLOCK
        assert second["in"] > first["in"] * 1.5


def test_every_entry_is_in_time_order(preset_set):
    _, traces = preset_set
    for trace in traces:
        times = [r["t"] for r in trace["requests"]]
        assert times == sorted(times)


def test_think_time_is_the_real_gap_between_main_calls(preset_set):
    _, traces = preset_set
    for trace in traces:
        previous, ends = None, []
        for r in trace["requests"]:
            if r["type"] == "subagent":
                ends.append(r["t"] + r["duration_ms"] / 1000)
                continue
            if previous is not None:
                gap = r["t"] - max([previous["t"] + previous["api_time"]] + ends)
                assert r["think_time"] == pytest.approx(gap, abs=TOLERANCE)
            previous, ends = r, []


def test_think_time_is_the_real_gap_inside_a_subagent():
    for trace in make_set(PRESETS["small"], 20, seed=2):
        for entry in (r for r in trace["requests"] if r["type"] == "subagent"):
            for a, b in zip(entry["requests"], entry["requests"][1:], strict=False):
                assert b["think_time"] == pytest.approx(b["t"] - (a["t"] + a["api_time"]), abs=TOLERANCE)


def test_calls_are_mostly_streaming(preset_set):
    _, traces = preset_set
    kinds = [c["type"] for t in traces for c in calls_of(t)[0]]
    assert set(kinds) <= {"n", "s"} and kinds.count("s") / len(kinds) > 0.9


def test_streaming_calls_carry_a_ttft_and_normal_ones_do_not(preset_set):
    _, traces = preset_set
    for trace in traces:
        for call in all_calls(trace):
            assert ("ttft" in call) == (call["type"] == "s")


def test_subagents_start_from_a_shared_prefix():
    preset = PRESETS["medium"]
    shared = max(8, int(0.04 * preset.max_context / BLOCK))
    compared = 0
    for trace in make_set(preset, 30, seed=2):
        subs = [r for r in trace["requests"] if r["type"] == "subagent"]
        firsts = [s["requests"][0]["hash_ids"] for s in subs]
        for other in firsts[1:]:
            assert other[:shared] == firsts[0][:shared]
            compared += 1
    assert compared > 0


def test_subagent_entries_are_complete():
    traces = make_set(PRESETS["small"], 20, seed=2)
    entries = [r for t in traces for r in t["requests"] if r["type"] == "subagent"]
    assert entries
    for e in entries:
        assert {"t", "agent_id", "subagent_type", "duration_ms", "total_tokens", "status", "requests",
                "models"} <= set(e)
        assert e["total_tokens"] == sum(c["in"] + c["out"] for c in e["requests"])
        finish = max(c["t"] + c["api_time"] for c in e["requests"])
        assert e["duration_ms"] / 1000 == pytest.approx(finish - e["t"], abs=TOLERANCE)


def test_no_subagents_option():
    traces = make_set(PRESETS["small"], 10, seed=2, subagents=False)
    assert all(r["type"] != "subagent" for t in traces for r in t["requests"])


@pytest.mark.parametrize("limit", [4_096, 5_000, 10_000])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_custom_limit_replaces_the_preset(limit, seed):
    traces = make_set(PRESETS["small"], 3, seed, max_context=limit)
    for trace in traces:
        for call in all_calls(trace):
            assert call["in"] + call["out"] <= limit
            assert call["in"] == BLOCK * len(call["hash_ids"])


def test_a_single_session_works():
    traces = make_set(PRESETS["tiny"], 1, seed=1)
    assert len(traces) == 1 and len(traces[0]["requests"]) >= 12


def test_the_traces_follow_aiperfs_model(preset_set):
    models = pytest.importorskip("aiperf.dataset.loader.weka_trace_models")
    _, traces = preset_set
    for trace in traces:
        models.WekaTrace.model_validate(trace)


def test_validate_stops_when_aiperf_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiperf.dataset.loader.weka_trace_models", None)
    with pytest.raises(SystemExit, match="needs AIPerf"):
        validate(make_set(PRESETS["tiny"], 1, seed=1))


def test_validate_writes_nothing_when_aiperf_is_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "aiperf.dataset.loader.weka_trace_models", None)
    with pytest.raises(SystemExit, match="Nothing was written"):
        main(["--preset", "tiny", "--output", str(tmp_path / "t"), "--sessions", "1", "--validate"])
    assert not (tmp_path / "t").exists()


def test_command_writes_one_file_per_session(tmp_path, capsys):
    out = tmp_path / "traces"
    assert main(["--preset", "tiny", "--output", str(out), "--sessions", "4"]) == 0
    files = sorted(out.glob("*.json"))
    assert [f.name for f in files] == [f"trace-000{i}.json" for i in range(1, 5)]
    assert json.loads(files[0].read_text())["block_size"] == BLOCK
    said = capsys.readouterr().out
    assert "4 sessions" in said and "--max-model-len" in said and "BENCH_AGENTX_DIR" in said


def test_command_gives_the_same_files_for_the_same_seed(tmp_path):
    for name in ("a", "b"):
        main(["--preset", "tiny", "--output", str(tmp_path / name), "--sessions", "3", "--seed", "9"])
    for first in sorted((tmp_path / "a").glob("*.json")):
        assert first.read_bytes() == (tmp_path / "b" / first.name).read_bytes()


def test_command_no_subagents(tmp_path):
    main(["--preset", "tiny", "--output", str(tmp_path), "--sessions", "5", "--no-subagents"])
    for file in tmp_path.glob("*.json"):
        assert all(r["type"] != "subagent" for r in json.loads(file.read_text())["requests"])


def test_command_refuses_a_folder_that_holds_traces(tmp_path):
    out = tmp_path / "traces"
    main(["--preset", "tiny", "--output", str(out), "--sessions", "1"])
    with pytest.raises(SystemExit, match="already holds traces"):
        main(["--preset", "tiny", "--output", str(out), "--sessions", "1"])


def test_overwrite_replaces_the_old_traces(tmp_path):
    out = tmp_path / "traces"
    main(["--preset", "tiny", "--output", str(out), "--sessions", "3"])
    assert main(["--preset", "tiny", "--output", str(out), "--sessions", "1", "--overwrite"]) == 0
    assert [f.name for f in out.glob("*.json")] == ["trace-0001.json"]


def test_overwrite_warns_about_other_json_files(tmp_path, capsys):
    (tmp_path / "notes.json").write_text("{}")
    main(["--preset", "tiny", "--output", str(tmp_path), "--sessions", "1", "--overwrite"])
    assert "other .json file" in capsys.readouterr().err
    assert (tmp_path / "notes.json").exists()


def test_command_refuses_a_path_that_is_a_file(tmp_path):
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(SystemExit, match="not a folder"):
        main(["--output", str(target)])


@pytest.mark.parametrize("argv, message", [
    (["--sessions", "0"], "--sessions"),
    (["--max-context", "1000"], "--max-context"),
    (["--max-context", "10000000"], "--max-context"),
])
def test_command_rejects_bad_values(tmp_path, argv, message):
    with pytest.raises(SystemExit, match=message):
        main(["--output", str(tmp_path / "t")] + argv)
