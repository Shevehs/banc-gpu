"""Modules import from the repository root, without any sys.path manipulation."""
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", ["tools.settings", "tools.server_config", "observation.analysis",
                                    "observation.query", "agents_test.metrics", "report.cli", "report.agents",
                                    "report.aiperf", "report.common"])
def test_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", ["observation.observer", "report.observation"])
def test_removed_modules_are_gone(module):
    """Observation now goes through Prometheus (observation.query); the old scraper and its report are removed."""
    with pytest.raises(ImportError):
        importlib.import_module(module)
    assert not (ROOT / (module.replace(".", "/") + ".py")).exists()


def test_report_has_only_the_aiperf_and_agents_kinds(capsys):
    from report import cli
    assert sorted(cli.KINDS) == ["agents", "aiperf"]
    with pytest.raises(SystemExit) as e:
        cli.main(["observation", "dir", "--target", "30"])
    assert e.value.code == 2 and "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["aiperf", "agents"])
def test_gpu_is_optional_for_the_reports(kind):
    from report import cli
    args = cli.build_parser().parse_args([kind, "dir", "--target", "30"])
    assert args.gpu is None and cli.build_parser().parse_args([kind, "dir", "--target", "30", "--gpu", "8"]).gpu == 8


# Modules that need third-party packages: (module, packages to find first).
@pytest.mark.parametrize("module,needs", [("interface.server", ["aiohttp"]),
                                          ("agents_test.common", ["httpx", "openai"]),
                                          ("agents_test.agents", ["httpx", "openai"]),
                                          ("agents_test.launcher", ["httpx", "openai"])])
def test_import_with_third_party_dependencies(module, needs):
    for package in needs:
        pytest.importorskip(package)
    importlib.import_module(module)


def test_no_sys_path_insert():
    offenders = [str(f.relative_to(ROOT)) for d in ("tools", "observation", "agents_test", "interface", "report")
                 for f in (ROOT / d).glob("*.py") if "sys.path.insert" in f.read_text(encoding="utf-8")]
    assert offenders == []


def test_target_env_ignored():
    gitignore = (ROOT / ".gitignore").read_text()
    assert "*.env" in gitignore and "!*.env.example" in gitignore
