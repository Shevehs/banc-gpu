"""Characterization tests: the standard output and CSV of the three reports are compared with recorded outputs.
To regenerate them after an intended change:  UPDATE_GOLDEN=1 python -m pytest tests/test_report_golden.py"""
import json
import os
import pathlib
import time

import pytest

from tests import fixtures

GOLDEN = pathlib.Path(__file__).parent / "golden"


@pytest.fixture(autouse=True)
def utc_timezone(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


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


@pytest.mark.parametrize("view,extra", [("windows", []), ("period", ["--period", "hour"]),
                                         ("period", []), ("agents", [])])
def test_observation_views(capsys, tmp_path, view, extra):
    root = fixtures.observation_tree(tmp_path)
    output = run_report("observation", [str(root), "--target", "30", "--gpu", "2", "--view", view, *extra], capsys)
    name = "observation_" + view + ("_hour" if "hour" in extra else "")
    check(name + ".txt", normalize(output, root))
    check(name + ".csv", (root / "table_observation.csv").read_text())
