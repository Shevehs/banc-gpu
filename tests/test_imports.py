"""Modules import from the repository root, without any sys.path manipulation."""
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", ["tools.settings", "tools.server_config", "observation.analysis",
                                    "observation.observer", "agents_test.metrics", "report.cli", "report.agents",
                                    "report.aiperf", "report.observation", "report.common"])
def test_import(module):
    importlib.import_module(module)


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
