"""Characterization tests: the standard output and CSV of the reports (aiperf, agents) and of the three views of
observation.query are compared with recorded outputs.
To regenerate them after an intended change (then review the diff):  UPDATE_GOLDEN=1 python -m pytest tests/test_report_golden.py"""
import json
import os
import pathlib

import pytest

from tests import fixtures

GOLDEN = pathlib.Path(__file__).parent / "golden"


pytestmark = pytest.mark.usefixtures("utc_timezone")


def run_report(kind, argv, capsys):
    from report.cli import main
    main([kind] + argv)
    return capsys.readouterr().out


def check(name, text):
    f = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        f.write_text(text, encoding="utf-8")
    assert f.exists(), f"reference file {name} missing; run with UPDATE_GOLDEN=1"
    assert text == f.read_text(encoding="utf-8")


def normalize(output, root):
    return output.replace(str(root), "<root>")


def check_aiperf(capsys, tmp_path, extra=(), name="aiperf"):
    root = fixtures.aiperf_tree(tmp_path)
    output = run_report("aiperf", [str(root), "--target", "30", "--gpu", "2", *extra], capsys)
    check(f"{name}.txt", normalize(output, root))
    check(f"{name}.csv", (root / "table.csv").read_text())


def test_aiperf_default(capsys, tmp_path):
    check_aiperf(capsys, tmp_path)


def test_aiperf_ttft_max_and_required_metrics(capsys, tmp_path):
    check_aiperf(capsys, tmp_path, ["--ttft-max", "410", "--require-metrics"], "aiperf_strict")


def test_agents_default(capsys, tmp_path):
    root = fixtures.agents_tree(tmp_path)
    output = run_report("agents", [str(root), "--target", "30", "--gpu", "1"], capsys)
    check("agents.txt", normalize(output, root))
    check("agents.csv", (root / "table_agents.csv").read_text())


def test_agents_required_metrics_and_missing(capsys, tmp_path):
    root = fixtures.agents_tree(tmp_path)
    p = root / "conf-r1" / "n1" / "level.json"
    meta = json.loads(p.read_text()); meta["metrics"] = {}; p.write_text(json.dumps(meta))
    output = run_report("agents", [str(root), "--target", "30", "--gpu", "1", "--require-metrics"], capsys)
    check("agents_strict.txt", normalize(output, root))


def run_query(prom, argv, capsys, monkeypatch):
    """Run observation.query.main() against the Prometheus stub; the stub URL is replaced by <prometheus>."""
    from observation import query
    monkeypatch.setattr("sys.argv", ["observation.query", "--prometheus", prom.url, "--target", "30", "--gpu", "8",
                                     "--start", "2024-05-14 00:00", "--end", "2024-05-15 00:00", "--window", "3h",
                                     *argv])
    query.main()
    return capsys.readouterr().out.replace(prom.url, "<prometheus>")


@pytest.fixture
def day(prom):
    """A Prometheus stub with the eight canned windows of fixtures.observation_day()."""
    return prom(fixtures.prometheus_responder(fixtures.DAY + fixtures.WINDOW_3H, fixtures.WINDOW_3H,
                                               fixtures.observation_day()))


def test_query_windows(day, capsys, monkeypatch, tmp_path):
    out = run_query(day, ["--litellm", 'job="litellm"', "--output", str(tmp_path / "t.csv")], capsys, monkeypatch)
    check("query_windows.txt", out)
    check("query_windows.csv", (tmp_path / "t.csv").read_text())


@pytest.mark.parametrize("name,extra", [("query_period", ["--view", "period"]),
                                        ("query_period_hour", ["--view", "period", "--period", "hour"])])
def test_query_period(day, capsys, monkeypatch, name, extra):
    check(name + ".txt", run_query(day, ["--litellm", 'job="litellm"', *extra], capsys, monkeypatch))


def test_query_agents(day, capsys, monkeypatch):
    check("query_agents.txt", run_query(day, ["--litellm", 'job="litellm"', "--view", "agents"], capsys, monkeypatch))
