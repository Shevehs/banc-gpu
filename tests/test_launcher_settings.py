"""agents_test/launcher.py: --url, --model and --metrics default to BENCH_URL, BENCH_MODEL and BENCH_METRICS, like the
shell scripts. (tests/conftest.py removes every BENCH_* variable before each test.)"""
import pytest

from agents_test.launcher import build_parser

DEFAULTS = ("http://127.0.0.1:8000", "bench", None)


def parse(*args):
    p = build_parser().parse_args(["--label", "x", *args])
    return p.url, p.model, p.metrics


def test_without_variables_the_defaults_are_unchanged():
    assert parse() == DEFAULTS


def test_the_variables_are_the_defaults(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    monkeypatch.setenv("BENCH_MODEL", "alias-1")
    monkeypatch.setenv("BENCH_METRICS", "http://h1:8000/metrics http://h2:8000/metrics")
    assert parse() == ("http://gateway:4000", "alias-1", "http://h1:8000/metrics http://h2:8000/metrics")


def test_an_option_on_the_command_line_wins(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    monkeypatch.setenv("BENCH_MODEL", "alias-1")
    monkeypatch.setenv("BENCH_METRICS", "http://h1:8000/metrics")
    assert parse("--url", "http://other:1", "--model", "m2", "--metrics", "http://o:2/metrics") == (
        "http://other:1", "m2", "http://o:2/metrics")


@pytest.mark.parametrize("name", ["BENCH_URL", "BENCH_MODEL", "BENCH_METRICS"])
def test_an_empty_variable_counts_as_not_set(monkeypatch, name):
    monkeypatch.setenv(name, "")
    assert parse() == DEFAULTS


def test_a_variable_with_a_generic_name_is_not_read(monkeypatch):
    for name in ("URL", "MODEL", "METRICS"):
        monkeypatch.setenv(name, "http://must-not-be-used")
    assert parse() == DEFAULTS


def test_only_one_variable_set_leaves_the_others_at_their_defaults(monkeypatch):
    monkeypatch.setenv("BENCH_MODEL", "alias-1")
    assert parse() == ("http://127.0.0.1:8000", "alias-1", None)


def test_the_other_options_do_not_depend_on_the_variables(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    p = build_parser().parse_args(["--label", "x"])
    assert (p.levels, p.duration, p.warmup, p.scale, p.mode, p.output) == ("1 2 4 8", 900, 300, 1.0, "scripted", "results-agents")
    assert p.label == "x" and p.no_metrics is False and p.overwrite is False and p.label_agents is False


def test_the_label_is_still_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
