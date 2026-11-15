"""sweep.sh and agentx.sh: argument and environment validation, without AIPerf and without the network.

Every case fails validation before aiperf would be called; a fake `aiperf` first on PATH records any call."""
import os
import shutil

import pytest

from tests import helpers

SWEEP, AGENTX = "sweep.sh", "agentx.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not found")


@pytest.fixture
def sandbox(tmp_path):
    """(run, tmp_path): run(script, *args, **env) executes the script from the repository in a tmp cwd and results dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "aiperf-was-called"
    fake = bin_dir / "aiperf"
    fake.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    fake.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()

    def run(script, *args, **env):
        environment = helpers.clean_environment(PATH=f"{bin_dir}:{os.environ['PATH']}", BENCH_ROOT_DIR=str(work / "results"),
                                                BENCH_URL="http://127.0.0.1:1", **env)
        result = helpers.run([BASH, str(helpers.ROOT / script), *args], cwd=work, env=environment, timeout=60)
        assert not marker.exists(), "aiperf was called"
        return result
    run.work = work
    return run


def nothing_created(sandbox):
    return list(sandbox.work.iterdir()) == []


# ---------------------------------------------------------------- shared checks
@pytest.mark.parametrize("script", [SWEEP, AGENTX])
@pytest.mark.parametrize("label", ["-x", "a b", "a..b", "a/b", ".x", "_x", "a;ls", "a$(id)", "x" * 81, "../x"])
def test_invalid_label(sandbox, script, label):
    r = sandbox(script, label)
    assert r.returncode == 1 and "Invalid label" in r.stdout
    assert nothing_created(sandbox)


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_missing_label(sandbox, script):
    r = sandbox(script)
    assert r.returncode != 0 and "give a label" in r.stderr


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_existing_results_directory_needs_overwrite(sandbox, script):
    existing = sandbox.work / "results" / "lbl"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("precious")
    r = sandbox(script, "lbl", "1")
    assert r.returncode == 1
    assert "already exists" in r.stdout and "BENCH_OVERWRITE=1" in r.stdout and str(existing) in r.stdout
    assert (existing / "keep.txt").read_text() == "precious"


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_overwrite_variable_is_the_prefixed_one(sandbox, script):
    """The old names (OVERWRITE, ECRASER) must not authorize deleting results."""
    existing = sandbox.work / "results" / "lbl"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("precious")
    r = sandbox(script, "lbl", "1", OVERWRITE="1", ECRASER="1")
    assert r.returncode == 1 and "BENCH_OVERWRITE=1" in r.stdout and (existing / "keep.txt").exists()


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_legacy_variables_warn_on_stderr_without_values(sandbox, script):
    existing = sandbox.work / "results" / "lbl"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("x")
    r = sandbox(script, "lbl", MODELE="old-model", CLE_API="sk-never-printed", API_KEY="sk-generic", URL="http://old")
    assert r.returncode == 1
    for old, new in (("MODELE", "BENCH_MODEL"), ("CLE_API", "BENCH_API_KEY"), ("API_KEY", "BENCH_API_KEY")):
        assert f"WARNING: {old} is set but is no longer read; use {new} instead." in r.stderr
    assert "URL is set" not in r.stderr                  # BENCH_URL is set by the sandbox: nothing to warn about
    assert "sk-never-printed" not in r.stdout + r.stderr and "sk-generic" not in r.stdout + r.stderr
    assert "WARNING" not in r.stdout


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_no_warning_when_only_bench_variables_are_set(sandbox, script):
    existing = sandbox.work / "results" / "lbl"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("x")
    r = sandbox(script, "lbl", BENCH_MODEL="m", BENCH_API_KEY="k")
    assert r.returncode == 1 and "WARNING" not in r.stderr


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_scripts_read_the_documented_bench_variables(script):
    text = (helpers.ROOT / script).read_text()
    names = ["BENCH_URL", "BENCH_MODEL", "BENCH_METRICS", "BENCH_API_KEY", "BENCH_TOKENIZER", "BENCH_ROOT_DIR", "BENCH_OVERWRITE", "BENCH_SEED"]
    names += ["BENCH_ISL", "BENCH_OSL", "BENCH_DURATION", "BENCH_REQUESTS_PER_CONCURRENCY", "BENCH_TEMPERATURE", "BENCH_EXTRA"] if script == SWEEP \
        else ["BENCH_AGENTX_DIR", "BENCH_AGENTX_VARIANT", "BENCH_SHORT_RUN", "BENCH_MAX_CONTEXT", "BENCH_DURATION"]
    assert [n for n in names if f"${{{n}:-" not in text] == []
    assert '${BENCH_MODEL:-bench}' in text


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_generic_names_are_not_read_by_the_scripts(script):
    text = (helpers.ROOT / script).read_text()
    for generic in ("URL", "MODEL", "API_KEY", "METRICS", "TOKENIZER", "DURATION", "ROOT_DIR", "OVERWRITE", "SEED", "EXTRA"):
        assert f"${{{generic}:-" not in text and f"${{{generic}}}" not in text.replace("${BENCH_", "")


# ---------------------------------------------------------------- agentx.sh only
@pytest.mark.parametrize("duration", ["abc", "1.5", "-5", "12s", "1 2"])
def test_agentx_non_numeric_duration(sandbox, duration):
    r = sandbox(AGENTX, "lbl", "1", BENCH_DURATION=duration)
    assert r.returncode == 1 and "BENCH_DURATION" in r.stdout and "integer number of seconds" in r.stdout
    assert nothing_created(sandbox)


@pytest.mark.parametrize("value", ["abc", "0", "012", "-1", "1.5", "32k"])
def test_agentx_bad_max_context(sandbox, value):
    r = sandbox(AGENTX, "lbl", "1", BENCH_MAX_CONTEXT=value)
    assert r.returncode == 1 and "BENCH_MAX_CONTEXT" in r.stdout
    assert nothing_created(sandbox)


@pytest.mark.parametrize("value", ["512k", "256K", "full", "1"])
def test_agentx_bad_variant(sandbox, value):
    r = sandbox(AGENTX, "lbl", "1", BENCH_AGENTX_VARIANT=value)
    assert r.returncode == 1 and "BENCH_AGENTX_VARIANT: empty or 256k" in r.stdout
    assert nothing_created(sandbox)


def test_agentx_missing_traces_folder(sandbox):
    r = sandbox(AGENTX, "lbl", "1", BENCH_AGENTX_DIR=str(sandbox.work / "no-such-folder"))
    assert r.returncode == 1 and "BENCH_AGENTX_DIR" in r.stdout and "folder not found" in r.stdout
    assert nothing_created(sandbox)


@pytest.mark.parametrize("levels", ["1 a", "1,2", "-1", "1.5"])
def test_agentx_invalid_concurrency(sandbox, levels):
    r = sandbox(AGENTX, "lbl", levels)
    assert r.returncode == 1 and "Invalid concurrency" in r.stdout
    assert nothing_created(sandbox)


def test_agentx_default_duration_is_long_enough_but_short_values_need_short_run(sandbox):
    r = sandbox(AGENTX, "lbl", "1", BENCH_DURATION="899")
    assert r.returncode == 1 and "at least 900 s" in r.stdout and "BENCH_SHORT_RUN=1" in r.stdout
    assert nothing_created(sandbox)


def test_agentx_generic_duration_is_ignored_and_warned(sandbox):
    """DURATION=5 (old name) must not make the run short: the default 1800 s applies, so only the warning shows."""
    existing = sandbox.work / "results" / "lbl"
    existing.mkdir(parents=True)
    (existing / "k").write_text("x")
    r = sandbox(AGENTX, "lbl", "1", DURATION="5")
    assert "at least 900" not in r.stdout and "WARNING: DURATION is set but is no longer read; use BENCH_DURATION" in r.stderr
    assert "already exists" in r.stdout


def test_agentx_validation_order_reports_the_first_problem(sandbox):
    r = sandbox(AGENTX, "lbl", "1", BENCH_DURATION="abc", BENCH_MAX_CONTEXT="abc")
    assert "BENCH_DURATION" in r.stdout and "BENCH_MAX_CONTEXT" not in r.stdout


# ---------------------------------------------------------------- sweep.sh only
def test_sweep_bad_label_is_checked_before_anything_else(sandbox):
    r = sandbox(SWEEP, "bad label", "1 2", CLE_API="sk-x")
    assert r.returncode == 1 and r.stderr == ""          # no settings warning yet: the label comes first
