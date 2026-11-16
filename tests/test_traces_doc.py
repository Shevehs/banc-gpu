import json
import pathlib
import re

import pytest

DOC = pathlib.Path(__file__).resolve().parent.parent / "tools" / "traces.md"


def blocks(language):
    """The fenced code blocks of tools/traces.md written in `language`."""
    return re.findall(rf"```{language}\n(.*?)\n```", DOC.read_text(), re.S)


def test_the_example_trace_follows_aiperfs_model():
    models = pytest.importorskip("aiperf.dataset.loader.weka_trace_models")
    example = json.loads(blocks("json")[0])
    models.WekaTrace.model_validate(example)
    for call in example["requests"]:
        assert len(call["hash_ids"]) == -(-call["in"] // example["block_size"])


def test_the_conversion_code_builds_a_valid_trace():
    scope = {}
    exec(blocks("python")[0], scope)                     # the code of "Route 1"
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
