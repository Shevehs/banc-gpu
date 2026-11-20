"""agents_test/common.py: synthetic text, tool definitions, tokenizer fallback and the measured model call."""
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("openai")

from agents_test import common  # noqa: E402
from tests import helpers  # noqa: E402


@pytest.fixture(autouse=True)
def no_tokenizer(monkeypatch):
    """Start every test from 'tokenizer not loaded yet'."""
    monkeypatch.setitem(common._TOKENIZER, "loaded", False)
    monkeypatch.setitem(common._TOKENIZER, "tok", None)


def use_estimate_only(monkeypatch):
    monkeypatch.setitem(common._TOKENIZER, "loaded", True)


@pytest.mark.parametrize("kind", ["code", "scan", "log", "notes"])
def test_make_text_size_follows_the_chars_per_token_ratio(monkeypatch, kind):
    use_estimate_only(monkeypatch)
    for n in (32, 500, 5000):
        target = n * common.CHARS_PER_TOKEN[kind]
        text = common.make_text(kind, n, 1)
        assert target <= len(text) < target + 200       # stops at the first line past the target


def test_make_text_is_deterministic_per_seed(monkeypatch):
    use_estimate_only(monkeypatch)
    assert common.make_text("code", 300, 7) == common.make_text("code", 300, 7)
    assert common.make_text("code", 300, 7) != common.make_text("code", 300, 8)
    assert common.make_text("log", 300, 7) != common.make_text("scan", 300, 7)


def test_make_text_kinds_have_their_shape(monkeypatch):
    use_estimate_only(monkeypatch)
    assert "memcpy" in common.make_text("code", 100, 1)
    assert "/tcp open" in common.make_text("scan", 100, 1)
    assert "status=" in common.make_text("log", 100, 1)
    assert common.make_text("notes", 100, 1).startswith("- Finding 1:")
    assert common.make_text("unknown", 100, 1).startswith("- Finding 1:")        # falls back to notes


def test_make_text_grows_with_the_target(monkeypatch):
    use_estimate_only(monkeypatch)
    assert len(common.make_text("notes", 2000, 1)) > 5 * len(common.make_text("notes", 200, 1))


def test_tool_definitions(monkeypatch):
    use_estimate_only(monkeypatch)
    tools = common.tool_definitions(["a", "b"], 100, 5)
    assert [t["function"]["name"] for t in tools] == ["a", "b"] and all(t["type"] == "function" for t in tools)
    assert tools[0]["function"]["description"] != tools[1]["function"]["description"]      # seed + k
    assert tools[0]["function"]["description"] == common.tool_definitions(["a"], 100, 5)[0]["function"]["description"]
    assert tools[0]["function"]["parameters"]["required"] == ["target"]
    json.dumps(tools)
    assert common.tool_definitions([], 100, 5) == []


def test_tokenizer_none_without_file(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCH_TOKENIZER", str(tmp_path))                  # a folder without tokenizer.json
    assert common.tokenizer() is None
    assert common._TOKENIZER["loaded"] is True


def test_tokenizer_none_for_a_missing_path(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCH_TOKENIZER", str(tmp_path / "nope.json"))
    assert common.tokenizer() is None


def test_tokenizer_none_for_a_corrupt_file(monkeypatch, tmp_path):
    pytest.importorskip("tokenizers")
    (tmp_path / "tokenizer.json").write_text("{not json")
    monkeypatch.setenv("BENCH_TOKENIZER", str(tmp_path))
    assert common.tokenizer() is None


def test_generic_tokenizer_variable_is_not_read(monkeypatch, tmp_path):
    pytest.importorskip("tokenizers")
    monkeypatch.setenv("TOKENIZER", str(tmp_path / "other"))
    monkeypatch.setattr(common.pathlib.Path, "is_file", lambda self: False)
    assert common.tokenizer() is None


def make_word_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))
    return tok


def test_make_text_adjusts_to_the_real_token_count(monkeypatch, tmp_path):
    pytest.importorskip("tokenizers")
    make_word_tokenizer(tmp_path / "tokenizer.json")
    monkeypatch.setenv("BENCH_TOKENIZER", str(tmp_path))
    tok = common.tokenizer()
    assert tok is not None and common.tokenizer() is tok                  # loaded once
    for n in (200, 1000):
        actual = len(tok.encode(common.make_text("notes", n, 3)).ids)
        assert abs(actual - n) / n <= 0.10
    short = common.make_text("notes", 20, 3)                              # under 50 tokens: estimate only
    assert len(short) >= 20 * common.CHARS_PER_TOKEN["notes"]


def test_tokenizer_file_path_is_accepted(monkeypatch, tmp_path):
    pytest.importorskip("tokenizers")
    make_word_tokenizer(tmp_path / "t.json")
    monkeypatch.setenv("BENCH_TOKENIZER", str(tmp_path / "t.json"))
    assert common.tokenizer() is not None


def test_call_log_writes_json_lines(tmp_path):
    log = common.CallLog(str(tmp_path))
    log.log_call(agent="a", output_tokens=3)
    log.log_task(agent="a", task="t", error=None)
    log.close()
    assert json.loads((tmp_path / "calls.jsonl").read_text()) == {"agent": "a", "output_tokens": 3}
    assert json.loads((tmp_path / "tasks.jsonl").read_text())["task"] == "t"


def read_calls(tmp_path):
    return [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_model_call_against_the_fake_server(fake_url, tmp_path):
    log = common.CallLog(str(tmp_path))
    model = common.Model(fake_url, "bench", log, "scripted", api_key="alias-1", label=True)
    text, tools = model.call([{"role": "user", "content": "hello " * 50}], 12, {"agent": "pentest", "task": "t1"})
    log.close()
    assert text.count(" w") == 12 and tools == []
    (row,) = read_calls(tmp_path)
    assert row["agent"] == "pentest" and row["task"] == "t1" and row["error"] is None
    assert row["output_tokens"] == 12 and row["input_tokens"] > 0 and row["cached_tokens"] is not None
    assert row["ttft_s"] is not None and 0 < row["ttft_s"] <= row["duration_s"] and row["speed"] > 0
    assert row["tool_call_count"] == 0 and row["t_start"] > 0 and row["chunks"] == 12


def test_model_call_with_label_reaches_the_litellm_metrics(fake_url, tmp_path):
    log = common.CallLog(str(tmp_path))
    common.Model(fake_url, "bench", log, "scripted", api_key="alias-xyz", label=True).call(
        [{"role": "user", "content": "x"}], 5, {"agent": "report"})
    log.close()
    text = helpers.urllib.request.build_opener(helpers.urllib.request.ProxyHandler({})).open(
        fake_url + "/litellm/metrics/").read().decode()
    assert 'api_key_alias="alias-xyz",metadata_agent="report"' in text


def test_model_call_error_is_logged_then_raised(tmp_path):
    log = common.CallLog(str(tmp_path))
    model = common.Model(f"http://127.0.0.1:{helpers.free_port()}", "bench", log, "scripted", max_delay_s=2)
    with pytest.raises(RuntimeError):
        model.call([{"role": "user", "content": "x"}], 5, {"agent": "a"})
    log.close()
    (row,) = read_calls(tmp_path)
    assert row["error"] and row["output_tokens"] is None and row["ttft_s"] is None and row["speed"] is None


def test_tokenizer_root_uses_the_local_copy_for_an_empty_value_or_the_name_it_was_downloaded_for(tmp_path):
    (tmp_path / ".repo").write_text("org/model\n")
    (tmp_path / "tokenizer.json").write_text("{}")
    assert common.tokenizer_root("", tmp_path) == str(tmp_path)
    assert common.tokenizer_root(None, tmp_path) == str(tmp_path)
    assert common.tokenizer_root("org/model", tmp_path) == str(tmp_path)
    assert common.tokenizer_root("org/other", tmp_path) == "org/other"           # another model: not the local copy
    assert common.tokenizer_root("/some/folder", tmp_path) == "/some/folder"


def test_tokenizer_root_without_a_marker_keeps_a_name_as_it_is(tmp_path):
    assert common.tokenizer_root("org/model", tmp_path) == "org/model"
    assert common.tokenizer_root("", tmp_path) == str(tmp_path)


def test_tokenizer_root_ignores_a_marker_without_files(tmp_path):
    (tmp_path / ".repo").write_text("org/model\n")
    assert common.tokenizer_root("org/model", tmp_path) == "org/model"
