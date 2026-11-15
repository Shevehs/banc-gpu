import pytest

from report import common
from report.common import (Choice, choose, configurations, format_number, format_pct, group_rows, median_of,
                           strip_repetition, warn_missing_metrics, warn_rejected_lower_levels, warn_saturation,
                           write_csv)


def row(throughput, ok=True, **kw):
    return {"output_throughput_per_gpu": throughput, "ok": ok, "max_waiting_requests": 0, "kv_cache_max_fraction": 0.1, **kw}


def evaluate(rows):
    return all(r["ok"] for r in rows), any(r.get("other") for r in rows)


def groups_of(spec, configuration="conf"):
    """spec: {level: [rows]} -> groups for one configuration."""
    return {(configuration, level): rows for level, rows in spec.items()}


def test_format_helpers():
    assert format_number(None) == "—" and format_number("") == "—" and format_number(0) == "0.0"
    assert format_number(1.26) == "1.3" and format_number(1.26, 0) == "1" and format_number("2.5", 2) == "2.50"
    assert format_pct(None) == "—" and format_pct("") == "—" and format_pct(0.456) == "46%" and format_pct(0) == "0%"


def test_format_builtin_is_not_shadowed():
    assert not hasattr(common, "format")
    assert format(1.5, ".1f") == "1.5"


@pytest.mark.parametrize("label,expected", [("prod-r2", "prod"), ("prod", "prod"), ("a-r1-r3", "a-r1"), ("a-r", "a-r"),
                                             ("r2", "r2"), ("conf-n4-r10", "conf-n4")])
def test_strip_repetition(label, expected):
    assert strip_repetition(label) == expected


def test_median_of_treats_none_as_zero():
    assert median_of([{"x": 4}, {"x": None}, {"x": 2}], "x") == 2
    assert median_of([{"x": 1}, {"x": 3}], "x") == 2


def test_group_rows_and_configurations():
    rows = [{"c": "b", "l": 2}, {"c": "a", "l": 1}, {"c": "b", "l": 2}, {"c": "a", "l": 4}]
    groups = group_rows(rows, lambda r: r["c"], lambda r: r["l"])
    assert sorted(groups) == [("a", 1), ("a", 4), ("b", 2)] and len(groups[("b", 2)]) == 2
    assert configurations(groups) == ["a", "b"]


def test_write_csv(tmp_path):
    write_csv(tmp_path / "t.csv", [{"a": 1, "b": "x,y"}, {"a": 2, "b": ""}])
    assert (tmp_path / "t.csv").read_text().splitlines() == ["a,b", '1,"x,y"', "2,"]


def test_choose_prefer_level_takes_the_highest_accepted_level():
    groups = groups_of({1: [row(100)], 2: [row(300)], 4: [row(200)], 8: [row(500, ok=False)]})
    c = choose(groups, "conf", evaluate, prefer="level")
    assert (c.selected, c.throughput, c.levels, c.rejected) == (4, 200, [1, 2, 4, 8], [8])
    assert c.rows == groups[("conf", 4)] and c.repetitions == 1 and c.low == c.high == 200


def test_choose_prefer_throughput_takes_the_best_throughput():
    groups = groups_of({1: [row(100)], 2: [row(300)], 4: [row(200)]})
    c = choose(groups, "conf", evaluate, prefer="throughput")
    assert (c.selected, c.throughput) == (2, 300)


def test_choose_prefer_throughput_ties_go_to_the_higher_level():
    groups = groups_of({1: [row(300)], 2: [row(300)], 4: [row(100)]})
    assert choose(groups, "conf", evaluate, prefer="throughput").selected == 2


def test_choose_default_preference_is_level():
    groups = groups_of({1: [row(300)], 2: [row(100)]})
    assert choose(groups, "conf", evaluate).selected == 2


def test_choose_median_and_range_of_repetitions():
    groups = groups_of({2: [row(100), row(300), row(200)]})
    c = choose(groups, "conf", evaluate)
    assert (c.throughput, c.repetitions, c.low, c.high) == (200, 3, 100, 300)


def test_choose_one_failing_repetition_rejects_the_level():
    groups = groups_of({1: [row(100)], 2: [row(300), row(310, ok=False)]})
    c = choose(groups, "conf", evaluate)
    assert c.selected == 1 and c.rejected == [2]


def test_choose_nothing_accepted():
    c = choose(groups_of({1: [row(100, ok=False)], 2: [row(200, ok=False)]}), "conf", evaluate)
    assert c.selected is None and c.rejected == [1, 2] and c.rows == [] and c.throughput == 0.0


def test_choose_only_looks_at_the_requested_configuration():
    groups = {**groups_of({1: [row(100)]}, "a"), **groups_of({1: [row(5)], 8: [row(9)]}, "b")}
    c = choose(groups, "a", evaluate)
    assert c.levels == [1] and c.selected == 1


def test_choose_none_throughput_counts_as_zero():
    c = choose(groups_of({1: [row(None)]}), "conf", evaluate)
    assert c.selected == 1 and c.throughput == 0


def test_choose_other_reason_and_missing_metrics():
    groups = groups_of({1: [row(1, other=True)], 2: [row(2, missing=True)], 3: [row(3)]})
    c = choose(groups, "conf", evaluate, missing=lambda rows: any(r.get("missing") for r in rows))
    assert c.other_reason == {1} and c.without_metrics is True


def test_choose_without_missing_callback_never_flags_metrics():
    assert choose(groups_of({1: [row(1)]}), "conf", evaluate).without_metrics is False


def test_choice_defaults_are_not_shared():
    a, b = Choice("a", []), Choice("b", [])
    a.rejected.append(1)
    assert b.rejected == [] and b.other_reason == set() and b.rows == []


def test_warn_rejected_lower_levels(capsys):
    warn_rejected_lower_levels(Choice("c", [1, 2, 4], selected=4, rejected=[1, 2]))
    assert "lower levels (1, 2) do not meet the target" in capsys.readouterr().out
    warn_rejected_lower_levels(Choice("c", [1, 2, 4], selected=2, rejected=[4]), unit="task")
    assert capsys.readouterr().out == ""
    warn_rejected_lower_levels(Choice("c", [1, 2], selected=2, rejected=[1]), unit="task")
    assert "lower tasks (1)" in capsys.readouterr().out


def test_warn_missing_metrics(capsys):
    warn_missing_metrics(Choice("c", [], without_metrics=False))
    assert capsys.readouterr().out == ""
    warn_missing_metrics(Choice("c", [], without_metrics=True))
    assert "server metrics missing" in capsys.readouterr().out


def test_warn_saturation(capsys):
    warn_saturation([row(1)])
    assert capsys.readouterr().out == ""
    warn_saturation([row(1, max_waiting_requests=2)], "task")
    out = capsys.readouterr().out
    assert "waited in the server queue at this task" in out and "KV cache" not in out
    warn_saturation([row(1, kv_cache_max_fraction=0.95)])
    assert "KV cache is 95 % full" in capsys.readouterr().out
    warn_saturation([row(1, kv_cache_max_fraction=0.949, max_waiting_requests=None)])
    assert capsys.readouterr().out == ""
    warn_saturation([row(1, kv_cache_max_fraction=None, max_waiting_requests=None)])
    assert capsys.readouterr().out == ""
