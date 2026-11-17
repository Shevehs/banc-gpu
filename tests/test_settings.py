import io
import os

import pytest

from tests import helpers
from tools.settings import IGNORED, api_key, load_target_env, read_setting, warn_ignored_settings


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


@pytest.mark.parametrize("raw,expected", [("  http://x \n", "http://x"), ("\t", "fallback"), ("   ", "fallback")])
def test_white_space_around_a_value_is_removed_and_a_blank_value_counts_as_absent(monkeypatch, raw, expected):
    monkeypatch.setenv("BENCH_URL", raw)
    assert read_setting("URL", "fallback") == expected


def test_the_api_key_is_stripped_too(monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", " sk-1\n")
    assert api_key() == "sk-1"


def test_load_target_env_reads_the_exports_without_running_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for n in ("URL", "MODEL", "METRICS", "TOKENIZER", "API_KEY"):
        monkeypatch.delenv("BENCH_" + n, raising=False)
    f = tmp_path / "target.env"
    f.write_text('# comment\nexport BENCH_URL=http://h:8000\nexport BENCH_MODEL="org/model-4bit"  # served name\n'
                 "BENCH_METRICS='http://h:8001/metrics http://h:8002/metrics'\nexport BENCH_TOKENIZER=\"\"\n"
                 "export BENCH_API_KEY=$(cat secret)\nexport DO_NOT_TRACK=1\nexport BENCH_BAD=\"a;b$(x)\"\n")
    taken = load_target_env(f)
    assert taken == ["BENCH_URL", "BENCH_MODEL", "BENCH_METRICS"]
    assert read_setting("MODEL") == "org/model-4bit"
    assert read_setting("METRICS") == "http://h:8001/metrics http://h:8002/metrics"
    assert read_setting("TOKENIZER") is None and api_key() is None
    assert "DO_NOT_TRACK" not in taken


def test_load_target_env_keeps_what_the_terminal_already_set(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("BENCH_MODEL", "from-terminal")
    f = tmp_path / "target.env"
    f.write_text("export BENCH_MODEL=from-file\n")
    assert load_target_env(f) == [] and read_setting("MODEL") == "from-terminal"


def test_load_target_env_without_the_file(tmp_path):
    assert load_target_env(tmp_path / "missing.env") == []


def test_load_target_env_keeps_a_hash_inside_a_value_and_ignores_a_comment_after_a_space(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for n in ("URL", "MODEL"):
        os.environ.pop("BENCH_" + n, None)
    f = tmp_path / "target.env"
    f.write_text("export BENCH_URL=http://h:8000/#x\r\nexport BENCH_MODEL=m   # the model\r\n")
    assert load_target_env(f) == ["BENCH_URL", "BENCH_MODEL"]
    assert read_setting("URL") == "http://h:8000/#x" and read_setting("MODEL") == "m"


def test_load_target_env_takes_only_the_target_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("BENCH_ALLOWED_DIRS", None)
    f = tmp_path / "target.env"
    f.write_text("export BENCH_ALLOWED_DIRS=/\nexport BENCH_ALLOW_PUBLIC_URLS=1\nexport BENCH_PROMETHEUS_KEY=x\n")
    assert load_target_env(f) == [] and "BENCH_ALLOWED_DIRS" not in os.environ


def test_load_target_env_survives_a_bom_and_a_file_that_is_not_text(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("BENCH_MODEL", None)
    f = tmp_path / "target.env"
    f.write_bytes("\ufeffexport BENCH_MODEL=m\n".encode("utf-8"))
    assert load_target_env(f) == ["BENCH_MODEL"]
    f.write_bytes(b"\xff\xfe\x00bad")
    assert load_target_env(f) == []
