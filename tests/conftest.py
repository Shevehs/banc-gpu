"""Shared fixtures: a hermetic environment, a stub HTTP server and the fake server."""
import os
import time

import pytest

from tests import helpers
from tools.settings import IGNORED


@pytest.fixture(autouse=True)
def clean_bench_environment(monkeypatch):
    """No BENCH_* variable nor old/generic name leaks from the developer's shell into a test."""
    for name in list(os.environ):
        if name.startswith("BENCH_") or name in IGNORED:
            monkeypatch.delenv(name)


@pytest.fixture
def stub():
    """Factory of local HTTP stubs: stub({path: (status, body, headers)}) -> Stub (closed at teardown)."""
    made = []

    def factory(routes):
        s = helpers.Stub(routes)
        made.append(s)
        return s
    yield factory
    for s in made:
        s.close()


@pytest.fixture(scope="module")
def fake_url():
    """Base URL of a fake server shared by the tests of a module (skipped without aiohttp)."""
    pytest.importorskip("aiohttp")
    with helpers.fake_server() as url:
        yield url


@pytest.fixture
def prom():
    """Factory of Prometheus stubs: prom(responder) -> PromStub (closed at teardown)."""
    made = []

    def factory(responder):
        s = helpers.PromStub(responder)
        made.append(s)
        return s
    yield factory
    for s in made:
        s.close()


@pytest.fixture
def utc_timezone(monkeypatch):
    """Local time is UTC during the test (period labels and window dates are in local time)."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()
