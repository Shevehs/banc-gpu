"""agents_test/launcher.py: --url, --model and --metrics default to BENCH_URL, BENCH_MODEL and BENCH_METRICS, like the
shell scripts. (tests/conftest.py removes every BENCH_* variable before each test.)"""
import pytest

from agents_test.launcher import build_parser, metrics_urls, parse_args

DEFAULTS = ("http://127.0.0.1:8000", "bench", None)


def parse(*args):
    p = parse_args(["--label", "x", *args])
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


def test_the_metrics_variable_goes_with_the_url_variable(monkeypatch):
    """A run aimed at another URL (the fake server, say) must not read the metrics of the server of .env."""
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    monkeypatch.setenv("BENCH_METRICS", "http://real-vllm:8000/metrics")
    p = parse_args(["--label", "x", "--url", "http://127.0.0.1:8001"])
    assert p.url == "http://127.0.0.1:8001" and p.metrics is None
    assert metrics_urls(p) == ["http://127.0.0.1:8001/metrics"]


def test_the_model_variable_still_applies_when_only_the_url_is_given(monkeypatch):
    monkeypatch.setenv("BENCH_MODEL", "alias-1")
    assert parse("--url", "http://127.0.0.1:8001") == ("http://127.0.0.1:8001", "alias-1", None)


@pytest.mark.parametrize("name", ["BENCH_URL", "BENCH_MODEL", "BENCH_METRICS"])
@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_variable_counts_as_not_set(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert parse() == DEFAULTS


def test_white_space_around_a_variable_is_ignored(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "  http://gateway:4000\n")
    monkeypatch.setenv("BENCH_MODEL", " alias-1 ")
    monkeypatch.setenv("BENCH_METRICS", " http://h1:8000/metrics ")
    assert parse() == ("http://gateway:4000", "alias-1", "http://h1:8000/metrics")


def test_a_variable_with_a_generic_name_is_not_read(monkeypatch):
    for name in ("URL", "MODEL", "METRICS"):
        monkeypatch.setenv(name, "http://must-not-be-used")
    assert parse() == DEFAULTS


def test_only_one_variable_set_leaves_the_others_at_their_defaults(monkeypatch):
    monkeypatch.setenv("BENCH_MODEL", "alias-1")
    assert parse() == ("http://127.0.0.1:8000", "alias-1", None)


def test_the_other_options_do_not_depend_on_the_variables(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    p = parse_args(["--label", "x"])
    assert (p.levels, p.duration, p.warmup, p.scale, p.mode, p.output) == ("1 2 4 8", 900, 300, 1.0, "scripted", "results-agents")
    assert p.label == "x" and p.no_metrics is False and p.overwrite is False and p.label_agents is False


def test_the_label_is_still_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_the_raw_parser_leaves_the_target_options_empty(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    p = build_parser().parse_args(["--label", "x"])
    assert (p.url, p.model, p.metrics) == (None, None, None)


# ---------------------------------------------------------------- the /metrics URLs the run reads
@pytest.mark.parametrize("args,expected", [
    ([], ["http://127.0.0.1:8000/metrics"]),                                           # default: the target's own /metrics
    (["--url", "http://h:8000/"], ["http://h:8000/metrics"]),                          # a trailing slash is dropped
    (["--metrics", "http://a:1/metrics http://b:2/metrics"], ["http://a:1/metrics", "http://b:2/metrics"]),   # one per replica
    (["--metrics", "   "], ["http://127.0.0.1:8000/metrics"]),                         # blank: the default
    (["--no-metrics"], []),
    (["--no-metrics", "--metrics", "http://a:1/metrics"], []),                         # --no-metrics wins
])
def test_metrics_urls(args, expected):
    assert metrics_urls(parse_args(["--label", "x", *args])) == expected


def test_no_metrics_wins_over_the_variable(monkeypatch):
    monkeypatch.setenv("BENCH_METRICS", "http://real-vllm:8000/metrics")
    assert metrics_urls(parse_args(["--label", "x", "--no-metrics"])) == []


def test_the_metrics_variable_feeds_metrics_urls(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://gateway:4000")
    monkeypatch.setenv("BENCH_METRICS", "http://h1:8000/metrics http://h2:8000/metrics")
    assert metrics_urls(parse_args(["--label", "x"])) == ["http://h1:8000/metrics", "http://h2:8000/metrics"]
