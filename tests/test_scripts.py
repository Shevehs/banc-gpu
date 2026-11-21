"""sweep.sh and agentx.sh: argument and environment validation, without AIPerf and without the network.

Every validation case fails before aiperf would be called; a fake `aiperf` first on PATH records any call. The
tests that let agentx.sh reach aiperf (environment passed to it, final message) use their own fake that only dumps
the AIPERF_* variables it receives and exits."""
import json
import os
import shutil
import sys

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
    fake.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then printf "%s\\n" "${FAKE_AIPERF_VERSION-0.13.0}"; exit 0; fi\n'
                    f"touch '{marker}'\nexit 1\n")
    fake.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()

    def run(script, *args, **env):
        environment = helpers.clean_environment(**{"PATH": f"{bin_dir}:{os.environ['PATH']}", "BENCH_ROOT_DIR": str(work / "results"),
                                                    "BENCH_URL": "http://127.0.0.1:1", **env})
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


# ---------------------------------------------------------------- agentx.sh reaching aiperf (a fake that records its environment)
TIMEOUTS = ("AIPERF_DATASET_CONFIGURATION_TIMEOUT", "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT")


@pytest.fixture
def agentx_run(tmp_path):
    """run(**env) -> (result, {AIPERF_* variable: value seen by aiperf}, results dir); the fake aiperf exits with 0."""
    bin_dir, work = tmp_path / "bin", tmp_path / "work"
    bin_dir.mkdir()
    work.mkdir()
    dump = tmp_path / "aiperf-env"
    fake = bin_dir / "aiperf"
    fake.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo 0.13.0; exit 0; fi\n'
                    f"env | grep '^AIPERF_' | sort > '{dump}'\nexit 0\n")
    fake.chmod(0o755)
    system = bin_dir / "uname"                      # the operating system is chosen by the test: FAKE_UNAME (default Darwin)
    system.write_text('#!/bin/sh\necho "${FAKE_UNAME:-Darwin}"\n')
    system.chmod(0o755)

    def run(*, script=None, **env):
        environment = helpers.clean_environment(**{"PATH": f"{bin_dir}:{os.environ['PATH']}", "BENCH_ROOT_DIR": str(work / "results"),
                                                    "BENCH_URL": "http://127.0.0.1:1", "BENCH_SHORT_RUN": "1",
                                                    "BENCH_DURATION": "5", **env})
        result = helpers.run([BASH, str(script or helpers.ROOT / AGENTX), "lbl", "1"], cwd=work, env=environment, timeout=120)
        seen = dict(x.split("=", 1) for x in dump.read_text().splitlines()) if dump.exists() else None
        return result, seen, work / "results" / "lbl"
    return run


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk not found")
@pytest.mark.parametrize("env,expected", [
    ({}, ("1800", "1800")),                                                                   # defaults: 30 minutes
    ({TIMEOUTS[0]: "600"}, ("600", "600")),                                                   # profile timeout follows
    ({TIMEOUTS[0]: "600", TIMEOUTS[1]: "2400"}, ("600", "2400")),                             # larger profile value kept
    ({TIMEOUTS[1]: "3000"}, ("1800", "3000")),
    ({TIMEOUTS[1]: "100"}, ("1800", "1800")),                                                 # raised to the dataset value
    ({TIMEOUTS[0]: "3000"}, ("3000", "3000")),
    ({TIMEOUTS[0]: "900", TIMEOUTS[1]: "900"}, ("900", "900")),
])
def test_agentx_aiperf_timeouts(agentx_run, env, expected):
    result, seen, _ = agentx_run(**env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (seen[TIMEOUTS[0]], seen[TIMEOUTS[1]]) == expected
    assert int(seen[TIMEOUTS[1]]) >= int(seen[TIMEOUTS[0]])          # AIPerf requires it


WORKERS = "AIPERF_DATASET_WEKA_PARALLEL_WORKERS"


@pytest.mark.parametrize("system,tokenizer,terminal,expected", [
    # system (uname), BENCH_TOKENIZER ("folder" = an existing folder), value set in the terminal, value seen by aiperf
    ("Darwin", "folder", None, "1"),                        # a local folder: the traces are loaded in a single process
    ("Darwin", "folder", "", "1"),                          # an empty value counts as not set
    ("Darwin", "folder", "4", "4"),                         # a value set in the terminal is kept
    ("Darwin", "folder", "0", "0"),                         # 0 (automatic, so parallel) is kept too: the user asked for it
    ("Darwin", "some-org/some-model", None, None),          # a Hugging Face name: the parallel loading is left as it is
    ("Linux", "folder", None, None),                        # AIPerf preloads the tokenizer there: nothing is changed
])
def test_agentx_workers_for_the_trace_loading(agentx_run, tmp_path, monkeypatch, system, tokenizer, terminal, expected):
    monkeypatch.delenv(WORKERS, raising=False)
    if tokenizer == "folder":
        (tmp_path / "tokenizer").mkdir()
        tokenizer = str(tmp_path / "tokenizer")
    env = {"BENCH_TOKENIZER": tokenizer, "FAKE_UNAME": system}
    if terminal is not None:
        env[WORKERS] = terminal
    result, seen, _ = agentx_run(**env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen.get(WORKERS) == expected
    assert ("single process" in result.stdout) == (expected == "1")          # said only when the script chose it


def default_tokenizer_tree(tmp_path, with_tokenizer, marker=None):
    """A copy of agentx.sh next to a tools/ folder, with or without the default tokenizer, to try the default path."""
    root = tmp_path / "tree"
    (root / "tools").mkdir(parents=True)
    shutil.copy(helpers.ROOT / AGENTX, root / AGENTX)
    for item in (helpers.ROOT / "tools").iterdir():
        if item.name != "tokenizer":
            (root / "tools" / item.name).symlink_to(item)
    if with_tokenizer:
        (root / "tools" / "tokenizer").mkdir()
        (root / "tools" / "tokenizer" / "tokenizer.json").write_text("{}")
        if marker:
            (root / "tools" / "tokenizer" / ".repo").write_text(marker + "\n")      # the model the download was for
    return root / AGENTX


@pytest.mark.parametrize("with_tokenizer,expected", [(True, "1"), (False, None)])
def test_agentx_default_tokenizer_folder(agentx_run, tmp_path, monkeypatch, with_tokenizer, expected):
    """Without BENCH_TOKENIZER, tools/tokenizer/ is used when it holds tokenizer.json: the same single-process loading."""
    monkeypatch.delenv(WORKERS, raising=False)
    result, seen, _ = agentx_run(script=default_tokenizer_tree(tmp_path, with_tokenizer), FAKE_UNAME="Darwin")
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen.get(WORKERS) == expected


@pytest.mark.parametrize("marker,expected", [("org/model", "1"), ("org/other", None), (None, None)])
def test_agentx_a_name_uses_the_local_copy_downloaded_for_it(agentx_run, tmp_path, monkeypatch, marker, expected):
    """BENCH_TOKENIZER=<name> uses tools/tokenizer/ only if its .repo file holds that name; otherwise AIPerf gets the name."""
    monkeypatch.delenv(WORKERS, raising=False)
    result, seen, _ = agentx_run(script=default_tokenizer_tree(tmp_path, True, marker), FAKE_UNAME="Darwin",
                                 BENCH_TOKENIZER="org/model")
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen.get(WORKERS) == expected


def test_agentx_final_message_says_the_gpus_are_deduced(agentx_run):
    result, _, results = agentx_run()
    assert result.returncode == 0
    assert "(GPUs deduced from the fingerprint; --gpu N to force)" in result.stdout
    assert "--gpu <GPU>" not in result.stdout and "python -m report aiperf" in result.stdout
    config = json.loads((results / "server_config.json").read_text())      # collected even though the target is down
    assert config["fingerprint"]["response"] is False and config["servers"][0]["vllm_lines"] is None


# ---------------------------------------------------------------- a model name the server does not know
VLLM_UNKNOWN_MODEL = (404, {"error": {"message": "The model `bench` does not exist.", "type": "NotFoundError",
                                      "param": "model", "code": 404}}, {})
LITELLM_UNKNOWN_MODEL = (400, {"error": {"message": "/chat/completions: Invalid model name passed in model=bench.",
                                         "type": "invalid_request_error", "param": None, "code": "400"}}, {})
SERVED = (200, {"data": [{"id": "served-one"}, {"id": "served-two"}]}, {})


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
@pytest.mark.parametrize("answer", [VLLM_UNKNOWN_MODEL, LITELLM_UNKNOWN_MODEL])
def test_unknown_model_stops_before_anything_is_created(sandbox, stub, script, answer):
    s = stub({"/v1/chat/completions": answer, "/v1/models": SERVED})
    r = sandbox(script, "lbl", BENCH_URL=s.url)
    assert r.returncode == 1
    assert 'does not know the model "bench"' in r.stderr and "served-one, served-two" in r.stderr
    assert "set BENCH_MODEL" in r.stdout
    assert nothing_created(sandbox)


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_unknown_model_keeps_existing_results_even_with_overwrite(sandbox, stub, script):
    old = sandbox.work / "results" / "lbl"
    old.mkdir(parents=True)
    (old / "old.txt").write_text("kept")
    s = stub({"/v1/chat/completions": VLLM_UNKNOWN_MODEL, "/v1/models": SERVED})
    r = sandbox(script, "lbl", BENCH_URL=s.url, BENCH_OVERWRITE="1")
    assert r.returncode == 1 and (old / "old.txt").read_text() == "kept"


@pytest.fixture
def aiperf_reached(tmp_path):
    """run(script, **env) -> (result, whether aiperf was called). The checks before aiperf run for real; the fake aiperf
    fails at once. A fake python3 makes the model check (--check-model) exit with the code FAKE_CHECK_EXIT, when set."""
    bin_dir, work = tmp_path / "bin", tmp_path / "work"
    bin_dir.mkdir()
    work.mkdir()
    marker = tmp_path / "aiperf-was-called"
    fake = bin_dir / "aiperf"
    fake.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then printf "%s\\n" "${FAKE_AIPERF_VERSION-0.13.0}"; exit 0; fi\n'
                    f"touch '{marker}'\nexit 1\n")
    fake.chmod(0o755)
    shim = bin_dir / "python3"
    shim.write_text('#!/bin/sh\ncase " $* " in *" --check-model "*) [ -n "$FAKE_CHECK_EXIT" ] && exit "$FAKE_CHECK_EXIT" ;; '
                    f'esac\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)

    def run(script, **env):
        environment = helpers.clean_environment(**{"PATH": f"{bin_dir}:{os.environ['PATH']}", "BENCH_ROOT_DIR": str(work / "results"),
                                                    "BENCH_URL": "http://127.0.0.1:1", **env})
        result = helpers.run([BASH, str(helpers.ROOT / script), "lbl"], cwd=work, env=environment, timeout=120)
        return result, marker.exists()
    return run


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_a_model_the_server_accepts_goes_on(aiperf_reached, stub, script):
    s = stub({"/v1/chat/completions": (200, {"system_fingerprint": None}, {}), "/v1/models": SERVED})
    result, reached = aiperf_reached(script, BENCH_URL=s.url, BENCH_MODEL="served-one")
    assert reached and "does not know" not in result.stderr


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
@pytest.mark.parametrize("code,stops", [("0", False), ("1", False), ("2", False), ("127", False), ("3", True)])
def test_only_exit_code_3_of_the_model_check_stops_the_script(aiperf_reached, script, code, stops):
    result, reached = aiperf_reached(script, FAKE_CHECK_EXIT=code)
    assert reached is not stops
    assert ("Stopped before any measurement" in result.stdout) is stops


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_the_key_of_the_shell_reaches_the_model_check(sandbox, stub, script):
    s = stub({"/v1/chat/completions": VLLM_UNKNOWN_MODEL, "/v1/models": SERVED})
    sandbox(script, "lbl", BENCH_URL=s.url, BENCH_API_KEY="sk-x")
    assert {r["headers"].get("Authorization") for r in s.seen} == {"Bearer sk-x"}


# ---------------------------------------------------------------- the bench needs AIPerf 0.13 or later
@pytest.mark.parametrize("script", [SWEEP, AGENTX])
@pytest.mark.parametrize("version", ["0.12.0", "0.12", "0.9.1", "0.10.0", "0.12.3", "aiperf 0.12.0", "v0.12.0", "0.12.0.dev20261001",
                                     "Loading the plugins...\n0.12.0"])
def test_an_aiperf_older_than_0_13_stops_the_script_before_anything_is_created(sandbox, script, version):
    r = sandbox(script, "lbl", FAKE_AIPERF_VERSION=version)
    assert r.returncode == 1
    assert "needs AIPerf 0.13.0 or later (pip install aiperf==0.13.0)" in r.stdout
    assert nothing_created(sandbox)


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
def test_an_aiperf_older_than_0_13_keeps_existing_results_even_with_overwrite(sandbox, script):
    old = sandbox.work / "results" / "lbl"
    old.mkdir(parents=True)
    (old / "old.txt").write_text("kept")
    r = sandbox(script, "lbl", FAKE_AIPERF_VERSION="0.12.0", BENCH_OVERWRITE="1")
    assert r.returncode == 1 and (old / "old.txt").read_text() == "kept"


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
@pytest.mark.parametrize("version", ["0.13.0", "0.13", "0.13.1", "0.14.0", "1.0.0", "0.13.0.dev20261113", "0.100.0", "aiperf 0.13.0"])
def test_an_aiperf_0_13_or_later_goes_on(aiperf_reached, script, version):
    result, reached = aiperf_reached(script, FAKE_AIPERF_VERSION=version)
    assert reached and "needs AIPerf" not in result.stdout


@pytest.mark.parametrize("script", [SWEEP, AGENTX])
@pytest.mark.parametrize("version", ["", "not a version"])
def test_an_unreadable_aiperf_version_does_not_stop_the_script(aiperf_reached, script, version):
    """The later steps report a missing or broken aiperf themselves."""
    result, reached = aiperf_reached(script, FAKE_AIPERF_VERSION=version)
    assert reached and "needs AIPerf" not in result.stdout
