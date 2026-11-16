import json
import pathlib
import re

import pytest

from tools.make_traces import PRESETS

DOC = pathlib.Path(__file__).resolve().parent.parent / "tools" / "traces.md"


def blocks(language):
    """The fenced code blocks of tools/traces.md written in `language`."""
    return re.findall(rf"```{language}\n(.*?)\n```", DOC.read_text(), re.S)


def test_the_example_trace_follows_aiperfs_model():
    models = pytest.importorskip("aiperf.dataset.loader.weka_trace_models")
    example = json.loads(next(b for b in blocks("json") if '"block_size"' in b))
    models.WekaTrace.model_validate(example)
    for call in example["requests"]:
        assert len(call["hash_ids"]) == -(-call["in"] // example["block_size"])


def test_the_conversion_code_builds_a_valid_trace():
    scope = {}
    exec(next(b for b in blocks("python") if "def build_trace" in b), scope)     # the code of "Route 1"
    prompt = list(range(1000, 1700))
    calls = []
    for k in range(4):
        calls.append({"t": 8.5 * k, "tokens": list(prompt), "out": 150, "api_time": 6.0})
        prompt = prompt + [5000 + k * 300 + i for i in range(300)]
    trace = scope["build_trace"]("s", "m", calls)
    ids = [r["hash_ids"] for r in trace["requests"]]
    assert [r["in"] for r in trace["requests"]] == [700, 1000, 1300, 1600]
    assert all(b[:len(a) - 1] == a[:-1] for a, b in zip(ids, ids[1:], strict=False))   # same prefix, same IDs
    assert [r.get("think_time") for r in trace["requests"]] == [None, 2.5, 2.5, 2.5]
    models = pytest.importorskip("aiperf.dataset.loader.weka_trace_models")
    models.WekaTrace.model_validate(trace)


def test_the_preset_table_matches_the_presets():
    rows = re.findall(r"\| `(\w+)` \| ([\d,]+) tokens \| ([\d,]+) \|", DOC.read_text())
    assert {name for name, _, _ in rows} == set(PRESETS)
    for name, longest, needs in rows:
        assert int(longest.replace(",", "")) == PRESETS[name].max_context
        assert int(needs.replace(",", "")) == PRESETS[name].max_context + 2_048
