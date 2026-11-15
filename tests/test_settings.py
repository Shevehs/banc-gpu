import io

import pytest

from tests import helpers
from tools.settings import IGNORED, api_key, read_setting, warn_ignored_settings


def test_prefixed_name_is_read(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://new")
    assert read_setting("URL") == "http://new"


def test_unprefixed_name_is_not_read(monkeypatch):
    monkeypatch.setenv("URL", "http://old")
    assert read_setting("URL") is None


def test_prefix_wins_over_generic_name(monkeypatch):
    monkeypatch.setenv("BENCH_URL", "http://new")
    monkeypatch.setenv("URL", "http://old")
    assert read_setting("URL") == "http://new"


def test_empty_value_counts_as_absent(monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", "")
    monkeypatch.setenv("API_KEY", "k")
    assert api_key() is None


def test_generic_api_key_is_never_read(monkeypatch):
    monkeypatch.setenv("API_KEY", "other-tool-secret")
    monkeypatch.setenv("CLE_API", "legacy")
    assert api_key() is None


def test_api_key_from_prefixed_name(monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", "k")
    assert api_key() == "k"


def test_default(monkeypatch):
    assert read_setting("MODEL", "bench") == "bench"
    monkeypatch.setenv("MODEL", "other")  # generic name: ignored
    assert read_setting("MODEL", "bench") == "bench"


def test_default_is_none_without_value():
    assert read_setting("MODEL") is None


def test_ignored_maps_to_prefixed_names():
    assert IGNORED and all(new.startswith("BENCH_") for new in IGNORED.values())
    assert not any(old.startswith("BENCH_") and old == new for old, new in IGNORED.items())


def warnings_for(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = io.StringIO()
    return warn_ignored_settings(out), out.getvalue()


def test_no_warning_without_ignored_variable():
    count, text = warnings_for(pytest.MonkeyPatch())
    assert count == 0 and text == ""


def test_warning_names_old_and_new_variable(monkeypatch):
    count, text = warnings_for(monkeypatch, MODELE="m")
    assert count == 1
    assert text == "WARNING: MODELE is set but is no longer read; use BENCH_MODEL instead.\n"


def test_warning_never_prints_values(monkeypatch):
    count, text = warnings_for(monkeypatch, API_KEY="sk-very-secret", CLE_API="sk-other-secret", BANC_JETON="tok-123")
    assert count == 3
    assert "secret" not in text and "tok-123" not in text
    assert "API_KEY" in text and "CLE_API" in text and "BANC_JETON" in text


def test_no_warning_when_prefixed_counterpart_is_set(monkeypatch):
    count, text = warnings_for(monkeypatch, URL="http://old", BENCH_URL="http://new")
    assert (count, text) == (0, "")


def test_empty_prefixed_counterpart_does_not_silence(monkeypatch):
    count, _ = warnings_for(monkeypatch, URL="http://old", BENCH_URL="")
    assert count == 1


def test_empty_ignored_variable_does_not_warn(monkeypatch):
    count, _ = warnings_for(monkeypatch, URL="")
    assert count == 0


def test_shared_target_warns_once_per_old_name(monkeypatch):
    count, text = warnings_for(monkeypatch, DURATION="1", DUREE="2")
    assert count == 2 and text.count("BENCH_DURATION") == 2


def test_default_stream_is_stderr(monkeypatch, capsys):
    monkeypatch.setenv("MODELE", "m")
    assert warn_ignored_settings() == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "MODELE" in captured.err


def test_script_prints_warnings_on_stderr_only():
    env = helpers.clean_environment(CLE_API="sk-secret", BENCH_MODEL="x")
    r = helpers.run([helpers.sys.executable, "-B", "tools/settings.py"], env=env)
    assert r.returncode == 0 and r.stdout == ""
    assert "CLE_API" in r.stderr and "BENCH_API_KEY" in r.stderr and "sk-secret" not in r.stderr
