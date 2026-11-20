"""Input validators and command builders of interface/server.py: the security boundary of the interface."""
import asyncio
import json
import pathlib
import sys

import pytest

pytest.importorskip("aiohttp")

from interface import server  # noqa: E402
from interface.server import Refusal  # noqa: E402

LONG = "a" * 400
PRIVATE_ADDRESS = "10.0.0.9"
PUBLIC_ADDRESS = "93.184.216.34"


@pytest.fixture(autouse=True)
def fake_dns_and_dirs(monkeypatch, tmp_path):
    """Host names resolve to a private address (names starting with 'public' or ending in .example.com resolve
    to a public one, '*.invalid' do not resolve); tmp_path is an allowed directory."""
    import ipaddress

    def addresses(host):
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            pass
        if host.endswith(".invalid"):
            return []
        if host.startswith("public") or host.endswith(".example.com"):
            return [ipaddress.ip_address(PUBLIC_ADDRESS)]
        return [ipaddress.ip_address(PRIVATE_ADDRESS)]

    monkeypatch.setattr(server, "_addresses", addresses)
    monkeypatch.setenv("BENCH_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.delenv("BENCH_ALLOW_PUBLIC_URLS", raising=False)


def accepts(fn, cases):
    return [pytest.param(v, expected, id=repr(v)[:40]) for v, expected in cases]


# ---------------------------------------------------------------- results_dir
@pytest.mark.parametrize("value", ["results", "agents_test/results-agents", "a/b/c", "results.v2", "r_1-x", " results "])
def test_results_dir_accepts(value):
    assert server.results_dir(value) == value.strip()


@pytest.mark.parametrize("value", ["", None, "   ", "..", "../x", "a/../b", "a/..", "/etc", "/abs/path", "a b", "a;b",
                                    "a$(id)", "a`id`", "a|b", "a\nb", "a\\b", "é", "a" * 121])
def test_results_dir_rejects(value):
    with pytest.raises(Refusal):
        server.results_dir(value)


# ---------------------------------------------------------------- valid_extra
@pytest.mark.parametrize("value,expected", [
    ("", ""), (None, ""), ("--use-server-token-count", "--use-server-token-count"),
    ("--benchmark-grace-period 30", "--benchmark-grace-period 30"),
    ("  --num-dataset-entries   100  --use-server-token-count ", "--num-dataset-entries 100 --use-server-token-count"),
    ("--prompt-corpus corpus.txt", "--prompt-corpus corpus.txt"),
    ("--request-timeout-seconds 1.5", "--request-timeout-seconds 1.5"),
])
def test_valid_extra_accepts(value, expected):
    assert server.valid_extra(value) == expected


@pytest.mark.parametrize("value", [
    "--api-key x", "--use-server-token-count --api-key k", "--url http://evil", "-m other", "--help",
    "use-server-token-count", "--benchmark-grace-period",                   # missing value
    "--use-server-token-count extra",                                       # value for a flag without value
    "--benchmark-grace-period 1;ls", "--benchmark-grace-period $(id)", "--benchmark-grace-period `id`",
    "--prompt-corpus ../secret", "--prompt-corpus /etc/passwd", "--prompt-corpus " + "a" * 41,
    "--benchmark-grace-period -5 --api-key k", "; rm -rf /", "--use-server-token-count && ls",
    "--use-server-token-count --use-server-token-count --api-key=x",
])
def test_valid_extra_rejects(value):
    with pytest.raises(Refusal):
        server.valid_extra(value)


def test_valid_extra_unbalanced_quote_is_a_refusal():
    with pytest.raises(Refusal):
        server.valid_extra("--prompt-corpus 'a")


# ---------------------------------------------------------------- valid_local_url
@pytest.mark.parametrize("value,expected", [
    ("http://127.0.0.1:8000", "http://127.0.0.1:8000"), ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
    (" https://gpu-node.lan:443 ", "https://gpu-node.lan:443"), ("http://localhost", "http://localhost"),
    ("http://10.1.2.3:8000", "http://10.1.2.3:8000"), ("http://192.168.1.5", "http://192.168.1.5"),
    ("http://169.254.1.1:80", "http://169.254.1.1:80"),
])
def test_valid_local_url_accepts(value, expected):
    assert server.valid_local_url(value) == expected


@pytest.mark.parametrize("value", [
    "", None, "ftp://h:21", "file:///etc/passwd", "javascript:alert(1)", "http://", "http://u:p@host:80",
    "http://host/path", "http://host:8000/v1", "http://host:8000?x=1", "http://host:123456", "http://host:80x",
    "http://[::1]:8000", "http://ho st", "http://host;ls", "http://host:8000/ extra", "http://host$(id)", "host:8000",
    "http://host\nHost: evil", "http://" + "a" * 5 + "_b",
])
def test_valid_local_url_rejects(value):
    with pytest.raises(Refusal):
        server.valid_local_url(value)


# ---------------------------------------------------------------- valid_name
@pytest.mark.parametrize("value", ["trial-r1", "a", "A.b_c-1", "9lives", "x" * 80])
def test_valid_name_accepts(value):
    assert server.valid_name(value, "label") == value


@pytest.mark.parametrize("value", ["", None, "-x", ".x", "_x", "a..b", "a/b", "../x", "x" * 81, "a b", "a;b", "a$(id)",
                                    "a\nb", "--api-key", "é"])
def test_valid_name_rejects(value):
    with pytest.raises(Refusal, match="label"):
        server.valid_name(value, "label")


def test_valid_name_strips_whitespace():
    assert server.valid_name("  ok  ", "label") == "ok"


@pytest.mark.parametrize("value", ["http://8.8.8.8:8000", "http://93.184.216.34", "http://gpu-node.example.com:8000",
                                    "http://public-host:8000", "http://unresolvable.invalid:8000"])
def test_valid_local_url_rejects_public_or_unresolvable_hosts(value):
    with pytest.raises(Refusal, match="target URL"):
        server.valid_local_url(value)


def test_valid_local_url_public_host_allowed_by_environment(monkeypatch):
    monkeypatch.setenv("BENCH_ALLOW_PUBLIC_URLS", "1")
    assert server.valid_local_url("http://gpu-node.example.com:8000") == "http://gpu-node.example.com:8000"


def test_valid_local_url_refuses_a_name_with_any_public_address(monkeypatch):
    import ipaddress
    monkeypatch.setattr(server, "_addresses", lambda host: [ipaddress.ip_address("10.0.0.1"),
                                                           ipaddress.ip_address(PUBLIC_ADDRESS)])
    with pytest.raises(Refusal, match="not on a local or private network"):
        server.valid_local_url("http://mixed:8000")


def test_valid_metrics_urls_rejects_public_hosts():
    with pytest.raises(Refusal, match="metrics URL"):
        server.valid_metrics_urls("http://h:8000/metrics http://8.8.8.8:8000/metrics")


# ---------------------------------------------------------------- valid_metrics_urls
def test_valid_metrics_urls_accepts():
    assert server.valid_metrics_urls("") == [] and server.valid_metrics_urls(None) == []
    assert server.valid_metrics_urls(" http://h1:8000/metrics   http://h2:8001/metrics ") == \
        ["http://h1:8000/metrics", "http://h2:8001/metrics"]
    assert server.valid_metrics_urls("http://h:4000/metrics/") == ["http://h:4000/metrics/"]
    assert server.valid_metrics_urls("http://h:8000") == ["http://h:8000"]


@pytest.mark.parametrize("value", ["ftp://h/metrics", "http://h/metrics?x=1", "http://h/m;ls", "http://h/m$(id)",
                                    "http://u:p@h/metrics", "h:8000/metrics", "http://h:8000/metrics http://bad host",
                                    "http://h/metrics#frag", "http://h/m|cat", "http://h:123456/metrics"])
def test_valid_metrics_urls_rejects(value):
    with pytest.raises(Refusal):
        server.valid_metrics_urls(value)


# ---------------------------------------------------------------- local_path
def test_local_path_accepts_absolute_directory(tmp_path):
    assert server.local_path(str(tmp_path), "dir") == str(tmp_path)


def test_local_path_relative_to_repository():
    assert server.local_path("tools", "dir") == str(server.ROOT / "tools")


def test_local_path_expands_home(tmp_path, monkeypatch):
    (tmp_path / "traces").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert server.local_path("~/traces", "dir") == str(tmp_path / "traces")


def test_local_path_refuses_directories_outside_the_allowed_locations():
    for outside in ("/etc", "/", "/usr"):
        with pytest.raises(Refusal, match="outside the allowed"):
            server.local_path(outside, "dir")


def test_local_path_refuses_a_symlink_that_leaves_the_allowed_directory(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Refusal, match="outside the allowed"):
        server.local_path(str(link), "dir")


def test_local_path_refuses_parent_traversal_out_of_the_allowed_directory(tmp_path):
    (tmp_path / "a").mkdir()
    with pytest.raises(Refusal, match="outside the allowed"):
        server.local_path(str(tmp_path / "a" / ".." / ".." / ".."), "dir")


def test_local_path_allows_a_directory_listed_in_bench_allowed_dirs(monkeypatch, tmp_path_factory):
    extra = tmp_path_factory.mktemp("extra")
    monkeypatch.setenv("BENCH_ALLOWED_DIRS", str(extra))
    assert server.local_path(str(extra), "dir") == str(extra)


@pytest.mark.parametrize("value", ["", None, "-x", "--help", "-", "a b", "a;b", "$(id)", "`id`", "a|b", "a\nb",
                                    "does-not-exist-xyz", "pyproject.toml", "/nonexistent/dir", "x" * 301])
def test_local_path_rejects(value):
    with pytest.raises(Refusal, match="dir"):
        server.local_path(value, "dir")


# ---------------------------------------------------------------- valid_model
@pytest.mark.parametrize("value", ["bench", "Qwen/Qwen3-0.6B", "org/model:tag@v1", "a", "a" * 200])
def test_valid_model_accepts(value):
    assert server.valid_model(value) == value


@pytest.mark.parametrize("value", ["", None, "-x", "--model", "../x", "a..b", "/abs", "a b", "a;b", "a$(id)", "a" * 201,
                                    "a\nb", "é"])
def test_valid_model_rejects(value):
    with pytest.raises(Refusal):
        server.valid_model(value)


# ---------------------------------------------------------------- valid_number
@pytest.mark.parametrize("value,integer,expected", [
    ("5", True, 5), (5, True, 5), (1, True, 1), (10, True, 10), ("2.5", False, 2.5), (0, False, 0.0), (10, False, 10.0),
    (" 7 ", True, 7),
])
def test_valid_number_accepts(value, integer, expected):
    x = server.valid_number(value, "n", 0 if not integer else 1, 10, integer)
    assert x == expected and isinstance(x, int if integer else float)


@pytest.mark.parametrize("value,integer", [
    (None, True), ("", True), ("abc", True), ("1.5", True), ("1e1", True), ("nan", False), ("inf", False), ("-inf", False),
    ("0", True), ("11", True), (11, True), (-1, False), ("1;ls", True), ([], True), ({}, False),
])
def test_valid_number_rejects(value, integer):
    with pytest.raises(Refusal, match="n"):
        server.valid_number(value, "n", 1 if integer else 0, 10, integer)


# ---------------------------------------------------------------- valid_key
@pytest.mark.parametrize("value,expected", [("", ""), (None, ""), ("  ", ""), ("sk-abc_123.XYZ", "sk-abc_123.XYZ"),
                                             (" k ", "k"), ("a" * 500, "a" * 500), ("!~", "!~")])
def test_valid_key_accepts(value, expected):
    assert server.valid_key(value) == expected


@pytest.mark.parametrize("value", ["--api-key x", "a b", "a\tb", "a\nb", "a" * 501, "clé", "a\x00b", "a\x7fb"])
def test_valid_key_rejects(value):
    with pytest.raises(Refusal):
        server.valid_key(value)


# ---------------------------------------------------------------- command builders
SYNTH = {"url": "http://127.0.0.1:8000", "model": "bench", "isl": 2048, "osl": 256, "label": "t1", "levels": "1  2 4"}


def test_synthetic_command_environment_and_command():
    cmd, env, cwd = server.synthetic_command(SYNTH)
    assert cmd == ["bash", str(server.ROOT / "sweep.sh"), "t1", "1 2 4"] and cwd == server.ROOT
    assert env["BENCH_URL"] == "http://127.0.0.1:8000" and env["BENCH_MODEL"] == "bench"
    assert (env["BENCH_ISL"], env["BENCH_OSL"], env["BENCH_TEMPERATURE"]) == ("2048", "256", "0.7")
    assert env["BENCH_ROOT_DIR"] == "results" and env["BENCH_DURATION"] == "" and env["BENCH_REQUESTS_PER_CONCURRENCY"] == "8"
    assert env["BENCH_API_KEY"] == "" and env["BENCH_METRICS"] == "" and env["BENCH_EXTRA"] == "" and env["BENCH_TOKENIZER"] == ""


def test_synthetic_command_duration_replaces_request_count():
    _, env, _ = server.synthetic_command({**SYNTH, "duration": 120, "requests_per_concurrency": 99})
    assert env["BENCH_DURATION"] == "120" and env["BENCH_REQUESTS_PER_CONCURRENCY"] == "8"


def test_synthetic_command_optional_fields():
    _, env, _ = server.synthetic_command({**SYNTH, "api_key": "sk-1", "metrics": "http://h:8000/metrics http://h2:8000/metrics",
                                          "extra": "--use-server-token-count", "root": "out/x", "requests_per_concurrency": 3,
                                          "tokenizer": "Qwen/Qwen3-0.6B", "temperature": 0})
    assert env["BENCH_API_KEY"] == "sk-1" and env["BENCH_METRICS"] == "http://h:8000/metrics http://h2:8000/metrics"
    assert env["BENCH_EXTRA"] == "--use-server-token-count" and env["BENCH_ROOT_DIR"] == "out/x"
    assert env["BENCH_REQUESTS_PER_CONCURRENCY"] == "3" and env["BENCH_TOKENIZER"] == "Qwen/Qwen3-0.6B"
    assert env["BENCH_TEMPERATURE"] == "0.0"


def test_synthetic_command_tokenizer_home_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _, env, _ = server.synthetic_command({**SYNTH, "tokenizer": "~/tok"})
    assert env["BENCH_TOKENIZER"] == str(tmp_path / "tok")


def test_synthetic_command_sets_only_bench_variables():
    _, env, _ = server.synthetic_command(SYNTH)
    assert [k for k in env if not k.startswith("BENCH_")] == []


@pytest.mark.parametrize("change", [
    {"url": "ftp://x"}, {"model": "-x"}, {"model": ""}, {"isl": 0}, {"isl": "abc"}, {"osl": 100_001}, {"temperature": 3},
    {"root": "../out"}, {"root": "/tmp/out"}, {"tokenizer": "--evil"}, {"tokenizer": "a b"}, {"tokenizer": "a;b"},
    {"metrics": "ftp://x"}, {"api_key": "a b"}, {"duration": 5}, {"duration": 9999}, {"extra": "--api-key x"},
    {"levels": ""}, {"levels": "1;ls"}, {"levels": "1 a"}, {"levels": "12345"}, {"levels": " ".join(["1"] * 32)},
    {"label": "../x"}, {"label": "-x"}, {"label": ""}, {"requests_per_concurrency": 0},
])
def test_synthetic_command_rejects(change):
    with pytest.raises(Refusal):
        server.synthetic_command({**SYNTH, **change})


AGENTX = {"url": "http://127.0.0.1:8000", "model": "bench", "label": "ax1", "levels": "1 2"}


def test_agentx_command_defaults():
    cmd, env, cwd = server.agentx_command(AGENTX)
    assert cmd == ["bash", str(server.ROOT / "agentx.sh"), "ax1", "1 2"] and cwd == server.ROOT
    assert env == {"BENCH_URL": "http://127.0.0.1:8000", "BENCH_MODEL": "bench", "BENCH_METRICS": "", "BENCH_API_KEY": "",
                   "BENCH_DURATION": "1800", "BENCH_ROOT_DIR": "results-agentx", "BENCH_AGENTX_DIR": "",
                   "BENCH_SHORT_RUN": "", "BENCH_OVERWRITE": "", "BENCH_SEED": "", "BENCH_TOKENIZER": "",
                   "BENCH_MAX_CONTEXT": "", "BENCH_AGENTX_VARIANT": ""}


def test_agentx_command_options(tmp_path):
    _, env, _ = server.agentx_command({**AGENTX, "short_run": True, "duration": 30, "variant": "256k", "max_context": 32768,
                                      "traces_dir": str(tmp_path), "tokenizer": "tok", "api_key": "k", "root": "r/x"})
    assert env["BENCH_SHORT_RUN"] == "1" and env["BENCH_DURATION"] == "30" and env["BENCH_AGENTX_VARIANT"] == "256k"
    assert env["BENCH_MAX_CONTEXT"] == "32768" and env["BENCH_AGENTX_DIR"] == str(tmp_path)
    assert env["BENCH_TOKENIZER"] == "tok" and env["BENCH_API_KEY"] == "k" and env["BENCH_ROOT_DIR"] == "r/x"


def test_agentx_command_unknown_variant_means_default():
    assert server.agentx_command({**AGENTX, "variant": "other"})[1]["BENCH_AGENTX_VARIANT"] == ""


@pytest.mark.parametrize("change", [
    {"duration": 100},                      # below 900 s without short_run
    {"duration": 100_000}, {"duration": "x"}, {"short_run": True, "duration": 10},
    {"max_context": 10}, {"max_context": "abc"}, {"traces_dir": "/nonexistent-xyz"}, {"traces_dir": "-x"},
    {"tokenizer": "--x"}, {"levels": "a"}, {"label": "a/b"}, {"url": "x"}, {"root": "../x"}, {"metrics": "x"},
])
def test_agentx_command_rejects(change):
    with pytest.raises(Refusal):
        server.agentx_command({**AGENTX, **change})


OBS = {"prometheus": "http://10.0.0.5:9090", "target": 30, "gpu": 4}


def test_observation_command_minimal():
    cmd, env, cwd = server.observation_command(OBS)
    assert cmd == [sys.executable, "-m", "observation.query", "--prometheus", "http://10.0.0.5:9090", "--target", "30.0",
                   "--gpu", "4", "--vllm", 'job="vllm"', "--view", "windows", "--period", "range"]
    assert env == {} and cwd == server.ROOT


def test_observation_command_all_options():
    cmd, env, _ = server.observation_command({
        **OBS, "prometheus": " http://prom.lan:9090/ ", "prometheus_key": "tok-1", "vllm": 'job="vllm", instance=~"10.0.0.*"',
        "litellm": 'job="litellm"', "by": "api_key_alias", "since": "6h", "window": "15m", "scrape": "30s",
        "ttft_max": 2000, "view": "period", "period": "hour"}, excluded=[(1000.4, 2060.6)])
    joined = " ".join(cmd)
    assert "--prometheus http://prom.lan:9090 " in joined and "--ttft-max 2000.0" in joined
    assert '--litellm job="litellm" --by api_key_alias' in joined and "--since 6h" in joined
    assert "--window 15m" in joined and "--scrape 30s" in joined and "--view period --period hour" in joined
    assert "--exclude 1000/2061" in joined and "--start" not in joined
    assert cmd[cmd.index("--vllm") + 1] == 'job="vllm", instance=~"10.0.0.*"'
    assert env == {"BENCH_PROMETHEUS_KEY": "tok-1"} and "tok-1" not in joined  # the key never goes in the command


def test_observation_command_dates_and_default_litellm_label():
    cmd, _, _ = server.observation_command({**OBS, "litellm": 'job="l"', "start": "2026-05-01T08:00", "end": "2026-05-02 09:30"})
    assert cmd[cmd.index("--start") + 1] == "2026-05-01 08:00" and cmd[cmd.index("--end") + 1] == "2026-05-02 09:30"
    assert cmd[cmd.index("--by") + 1] == "metadata_agent"


def test_observation_command_unknown_view_means_windows():
    cmd, _, _ = server.observation_command({**OBS, "view": "x; ls", "period": "x"})
    assert cmd[-4:] == ["--view", "windows", "--period", "range"]


@pytest.mark.parametrize("change", [
    {"prometheus": ""}, {"prometheus": "ftp://x"}, {"prometheus": "http://h:9090/a b"}, {"prometheus": "http://h;ls:9090"},
    {"prometheus": "http://" + PUBLIC_ADDRESS + ":9090"}, {"prometheus": "http://public-prom.lan:9090"},
    {"prometheus": "http://nope.invalid:9090"}, {"target": 0}, {"target": "x"}, {"target": None}, {"gpu": 0}, {"gpu": 5000},
    {"gpu": ""}, {"gpu": None}, {"vllm": 'job="x"} or vector(1) #'}, {"vllm": "job=vllm"}, {"vllm": 'job="a\\"'},
    {"vllm": 'job="a" and up'}, {"vllm": 'rate(x[5m])'}, {"vllm": '{job="a"}'}, {"vllm": 'job="a";rm -rf /'},
    {"litellm": 'job="x"} or up{'}, {"litellm": "x"}, {"litellm": 'job="l"', "by": "bad label"},
    {"litellm": 'job="l"', "by": "metadata_"}, {"litellm": 'job="l"', "by": "x; ls"}, {"view": "agents"},
    {"since": "2 h"}, {"since": "401d"}, {"since": "-5m"}, {"since": "5x"}, {"window": "1e3"}, {"scrape": "5m;ls"},
    {"start": "yesterday"}, {"start": "2026-05-01"}, {"end": "2026-05-01 08:00; ls"}, {"ttft_max": 0}, {"ttft_max": "x"},
    {"prometheus_key": "a b"},
])
def test_observation_command_rejects(change):
    with pytest.raises(Refusal):
        server.observation_command({**OBS, **change})


@pytest.mark.parametrize("value", ['job="vllm"', 'job="vllm",instance=~"10.0.0.*"', ' job != "a" , env !~ "x|y" ',
                                   'metadata_agent="a b"', 'job=""'])
def test_valid_selector_accepts(value):
    assert server.valid_selector(value, "s") == value.strip()


@pytest.mark.parametrize("value", ["", None, "job", 'job=~', 'job="a"}', 'job="a" or x', "job='a'", 'job="a\\"b"',
                                   'job="a"\n,x="b"; up', "1job=\"a\"", 'sum(job="a")', 'job="a",'])
def test_valid_selector_rejects(value):
    with pytest.raises(Refusal):
        server.valid_selector(value, "s")


def test_valid_prometheus_url_accepts_private_hosts_with_a_path():
    assert server.valid_prometheus_url("http://localhost:9090/") == "http://localhost:9090"
    assert server.valid_prometheus_url("https://prom.lan/prometheus/") == "https://prom.lan/prometheus"


def test_valid_prometheus_url_public_host_allowed_by_environment(monkeypatch):
    monkeypatch.setenv("BENCH_ALLOW_PUBLIC_URLS", "1")
    assert server.valid_prometheus_url("http://prom.example.com:9090") == "http://prom.example.com:9090"


class FakeQuery:
    """Stand-in for the observation subprocess: records the call, writes the CSV asked by --view-output."""

    def __init__(self, cmd, kw, output="Highest mean concurrency that holds: 3\nWARNING: x\nplain\n", code=0, hang=False):
        self.cmd, self.kw, self.returncode, self.text, self.hang, self.killed = cmd, kw, code, output, hang, False
        if code == 0 and "--view-output" in cmd:
            with open(cmd[cmd.index("--view-output") + 1], "w", encoding="utf-8") as f:
                f.write("start,verdict\n2026-05-01 08:00,holds\n")

    async def communicate(self):
        if self.hang:
            await asyncio.sleep(5)
        return self.text.encode(), b""

    def kill(self):
        self.killed = True

    async def wait(self):
        return -9


@pytest.fixture
def fake_subprocess(monkeypatch, tmp_path):
    """Replaces create_subprocess_exec for observation() and read_results(): returns (spawned list, options dict)."""
    spawned, options = [], {}

    async def fake_exec(*cmd, **kw):
        spawned.append((cmd, kw))
        return FakeQuery(cmd, kw, **options)
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(server, "RUNS", {})
    return spawned, options


def call_observation(body):
    resp = asyncio.run(server.observation(FakeRequest(body)))
    return resp.status, json.loads(resp.text)


def test_observation_endpoint_reads_the_csv_and_filters_the_messages(fake_subprocess):
    spawned, _ = fake_subprocess
    status, body = call_observation({**OBS, "prometheus_key": "sk-PROM"})
    assert status == 200 and body["code"] == 0
    assert body["rows"] == [{"start": "2026-05-01 08:00", "verdict": "holds"}]
    assert body["messages"] == ["Highest mean concurrency that holds: 3", "WARNING: x"]
    (cmd, kw), = spawned
    assert kw["env"]["BENCH_PROMETHEUS_KEY"] == "sk-PROM" and "sk-PROM" not in body["command"]
    assert "BENCH_API_KEY" not in kw["env"] and kw["cwd"] == str(server.ROOT)
    assert not list((server.LOGS_DIR).glob("observation-*.csv"))  # the temporary CSV is removed


def test_observation_endpoint_excludes_the_runs_started_here(fake_subprocess):
    spawned, _ = fake_subprocess
    server.RUNS.update(a={"start": 1000.0, "end": 2000.0, "code": 0}, b={"start": 3000.0, "end": None, "code": None})
    call_observation(OBS)
    cmd = spawned[0][0]
    exclusions = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--exclude"]
    assert exclusions[0] == "1000/2060" and exclusions[1].startswith("3000/")


def test_observation_endpoint_refuses_with_400_and_runs_nothing(fake_subprocess):
    spawned, _ = fake_subprocess
    for body in ({**OBS, "vllm": 'job="x"} or up'}, {**OBS, "prometheus": "http://8.8.8.8:9090"}, ["list"]):
        status, payload = call_observation(body)
        assert status == 400 and payload["error"]
    assert spawned == []


def test_observation_endpoint_works_while_a_run_is_in_progress(fake_subprocess):
    server.RUNS["x"] = {"start": 1.0, "end": None, "code": None}
    status, _ = call_observation(OBS)
    assert status == 200


def test_observation_endpoint_times_out_with_504(fake_subprocess, monkeypatch):
    _, options = fake_subprocess
    options["hang"] = True
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(server.asyncio, "wait_for", lambda aw, timeout: real_wait_for(aw, 0.05))
    status, payload = call_observation(OBS)
    assert status == 504 and "120 s" in payload["error"]


def test_observation_endpoint_keeps_the_error_output(fake_subprocess):
    _, options = fake_subprocess
    options.update(output="Prometheus unreachable (http://10.0.0.5:9090): refused\n", code=1)
    status, body = call_observation(OBS)
    assert status == 200 and body["code"] == 1 and body["rows"] == [] and body["messages"] == [body["output"].strip()]


class FakeGet:
    """Request with only a query string."""

    def __init__(self, **query):
        self.query = query


def call_results(**query):
    resp = asyncio.run(server.read_results(FakeGet(**query)))
    return resp.status, json.loads(resp.text)


@pytest.mark.parametrize("kind,module", [("aiperf", "aiperf"), ("agents", "agents"), (None, "aiperf")])
def test_results_without_gpu_does_not_pass_it(fake_subprocess, kind, module):
    spawned, _ = fake_subprocess
    status, body = call_results(root="results", target="30", gpu="", type=kind)
    cmd = spawned[0][0]
    assert status == 200 and cmd[1:4] == ("-m", "report", module) and "--gpu" not in cmd and "--gpu" not in body["command"]


def test_results_with_gpu_passes_it(fake_subprocess):
    spawned, _ = fake_subprocess
    call_results(root="results", target="30", gpu="8", require="1")
    cmd = spawned[0][0]
    assert cmd[cmd.index("--gpu") + 1] == "8" and "--require-metrics" in cmd


def test_results_missing_gpu_parameter_is_also_deduced(fake_subprocess):
    spawned, _ = fake_subprocess
    call_results(root="results", target="30")
    assert "--gpu" not in spawned[0][0]


def test_results_keeps_the_gpu_deduction_notes_and_errors(fake_subprocess):
    _, options = fake_subprocess
    options.update(output="GPUs: r1: 8 (TP 8 x PP 1 x DP 1), deduced from the fingerprint.\nERROR: number of GPUs unknown\nnoise\n", code=1)
    _, body = call_results(root="results", target="30", gpu="")
    assert body["messages"] == ["GPUs: r1: 8 (TP 8 x PP 1 x DP 1), deduced from the fingerprint.",
                                "ERROR: number of GPUs unknown"]


@pytest.mark.parametrize("query", [dict(root="../x", target="30"), dict(root="r", target="0"), dict(root="r", target="30", gpu="0"),
                                   dict(root="r", target="30", gpu="4097"), dict(root="r", target="30", gpu="x")])
def test_results_refuses_bad_parameters(fake_subprocess, query):
    status, body = call_results(**query)
    assert status == 400 and body["error"] and fake_subprocess[0] == []


def test_roots_have_no_observation_kind(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    (tmp_path / "results" / "r1" / "c1").mkdir(parents=True)
    (tmp_path / "results" / "r1" / "c1" / "profile_export_aiperf.json").write_text("{}")
    (tmp_path / "results-obs" / "r1").mkdir(parents=True)
    (tmp_path / "results-obs" / "r1" / "observation.json").write_text("{}")
    body = json.loads(asyncio.run(server.list_roots(None)).text)
    assert body == {"synthetic": ["results"], "agents": [], "agentx": []}


AGENTS = {"agents": ["code_review", "pentest"], "levels": "1 2", "mode": "scripted", "url": "http://127.0.0.1:8000",
          "model": "bench", "duration": 60, "label": "a1"}


def test_agents_command_minimal():
    cmd, env, cwd = server.agents_command(AGENTS)
    assert cmd == [sys.executable, "-m", "agents_test.launcher", "--url", "http://127.0.0.1:8000", "--model", "bench",
                   "--agents", "code_review,pentest", "--levels", "1 2", "--duration", "60.0", "--warmup", "60.0",
                   "--scale", "1.0", "--latency-factor", "1.0", "--mode", "scripted", "--label", "a1",
                   "--output", str(server.ROOT / "agents_test/results-agents")]
    assert env == {} and cwd == server.ROOT


def test_agents_command_options_and_keys():
    cmd, env, _ = server.agents_command({**AGENTS, "api_key": "shared", "agent_keys": "code_review=k1, pentest=k2",
                                         "metrics": "http://h:8000/metrics", "label_agents": True, "scale": 0.2,
                                         "warmup": 5, "latency_factor": 0, "mode": "autonomous"})
    assert env == {"BENCH_API_KEY": "shared", "BENCH_AGENT_KEYS": "code_review=k1,pentest=k2"}
    assert "--label-agents" in cmd and cmd[cmd.index("--metrics") + 1] == "http://h:8000/metrics"
    assert cmd[cmd.index("--scale") + 1] == "0.2" and cmd[cmd.index("--mode") + 1] == "autonomous"


def test_agents_command_unknown_agent_names_are_dropped():
    cmd, _, _ = server.agents_command({**AGENTS, "agents": ["report", "rm -rf /", "../x"]})
    assert cmd[cmd.index("--agents") + 1] == "report"


@pytest.mark.parametrize("change", [
    {"agents": []}, {"agents": ["nope"]}, {"agents": None}, {"mode": "other"}, {"mode": None}, {"levels": "x"},
    {"duration": 5}, {"duration": None}, {"warmup": -1}, {"warmup": 4000}, {"scale": 0}, {"scale": 3},
    {"latency_factor": 11}, {"label": "../a"}, {"url": "x"}, {"model": ""}, {"root": "../a"}, {"metrics": "x"},
    {"agent_keys": "evil=k"}, {"agent_keys": "pentest="}, {"agent_keys": "pentest=a b"}, {"api_key": "a b"},
])
def test_agents_command_rejects(change):
    with pytest.raises(Refusal):
        server.agents_command({**AGENTS, **change})


# ---------------------------------------------------------------- environment and launch
def test_base_environment_passes_only_the_allowlist(monkeypatch):
    for k, v in {"PATH": "/bin", "BENCH_API_KEY": "secret", "API_KEY": "secret2", "LC_ALL": "C", "AWS_SECRET": "x",
                 "HF_HOME": "/hf"}.items():
        monkeypatch.setenv(k, v)
    env = server.base_environment()
    assert env["PATH"] == "/bin" and env["LC_ALL"] == "C" and env["HF_HOME"] == "/hf"
    assert not {"BENCH_API_KEY", "API_KEY", "AWS_SECRET"} & set(env)


class FakeRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeProcess:
    pid = 0

    async def wait(self):
        return 0


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """launch() with no real subprocess: returns call(body) -> (response status, json, spawned (cmd, kwargs) list)."""
    spawned = []

    async def fake_exec(*cmd, **kw):
        spawned.append((cmd, kw))
        return FakeProcess()
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(server, "RUNS", {})

    def call(body):
        async def go():
            resp = await server.launch(FakeRequest(body))
            await asyncio.sleep(0.01)  # lets follow() record the end of the fake process
            return resp
        resp = asyncio.run(go())
        return resp.status, json.loads(resp.text), spawned
    return call


def test_launch_masks_secrets_in_log_and_command(launcher):
    status, body, spawned = launcher({"type": "synthetic", **SYNTH, "api_key": "sk-SECRET-123"})
    assert status == 200
    run = server.RUNS[body["id"]]
    log = open(run["log"], encoding="utf-8").read()
    assert "sk-SECRET-123" not in run["command"] and "sk-SECRET-123" not in log
    assert "BENCH_API_KEY=***" in run["command"] and "BENCH_URL=http://127.0.0.1:8000" in run["command"]
    (cmd, kwargs), = spawned
    assert kwargs["env"]["BENCH_API_KEY"] == "sk-SECRET-123" and kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert kwargs["start_new_session"] is True and cmd[0] == "bash"
    assert run["code"] == 0 and "== end, exit code 0 ==" in open(run["log"], encoding="utf-8").read()


def test_launch_masks_every_secret_name(launcher):
    status, body, _ = launcher({"type": "agents", **AGENTS, "api_key": "sk-A", "agent_keys": "pentest=sk-B"})
    assert status == 200
    command = server.RUNS[body["id"]]["command"]
    assert "sk-A" not in command and "sk-B" not in command and "BENCH_AGENT_KEYS=***" in command


def test_secret_names_are_the_bench_names():
    assert set(server.SECRETS) == {"BENCH_API_KEY", "BENCH_AGENT_KEYS", "BENCH_PROMETHEUS_KEY"}


def test_prometheus_key_is_masked_in_the_displayed_command(launcher, monkeypatch):
    """Whatever command carries BENCH_PROMETHEUS_KEY in its environment, the log and the run record show ***."""
    monkeypatch.setattr(server, "synthetic_command",
                        lambda p: (["true"], {"BENCH_PROMETHEUS_KEY": "sk-PROM", "BENCH_URL": "http://h:1"}, server.ROOT))
    status, body, _ = launcher({"type": "synthetic"})
    run = server.RUNS[body["id"]]
    assert status == 200 and "sk-PROM" not in run["command"] and "BENCH_PROMETHEUS_KEY=***" in run["command"]
    assert "sk-PROM" not in open(run["log"], encoding="utf-8").read()


def test_observation_is_not_a_run_type(launcher):
    status, payload, spawned = launcher({"type": "observation", **OBS})
    assert status == 400 and "unknown run type" in payload["error"] and spawned == []


def test_launch_does_not_inherit_the_terminal_secrets(launcher, monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", "from-terminal")
    _, _, spawned = launcher({"type": "agentx", **AGENTX})
    assert spawned[0][1]["env"]["BENCH_API_KEY"] == ""


def test_agents_launch_does_not_inherit_the_terminal_target(launcher, monkeypatch):
    """The launcher now reads BENCH_URL, BENCH_MODEL and BENCH_METRICS: none of them may come from the terminal that
    started the interface, so that an empty form field keeps its meaning (the target's own /metrics)."""
    for name, value in (("BENCH_URL", "http://gateway:4000"), ("BENCH_MODEL", "terminal-model"),
                        ("BENCH_METRICS", "http://real-vllm:8000/metrics")):
        monkeypatch.setenv(name, value)
    status, _, spawned = launcher({"type": "agents", **AGENTS})
    cmd, kwargs = spawned[0]
    assert status == 200
    assert not [k for k in kwargs["env"] if k in ("BENCH_URL", "BENCH_MODEL", "BENCH_METRICS")]
    assert "--metrics" not in cmd                                      # an empty form field passes no --metrics
    assert cmd[cmd.index("--url") + 1] == "http://127.0.0.1:8000" and cmd[cmd.index("--model") + 1] == "bench"


@pytest.mark.parametrize("body,message", [
    ({"type": "nope"}, "unknown run type"), (ValueError("x"), "JSON body expected"), (["list"], "JSON body expected"),
    ({"type": "synthetic", **SYNTH, "url": "ftp://x"}, "target URL"),
])
def test_launch_refuses_with_400(launcher, body, message):
    status, payload, spawned = launcher(body)
    assert status == 400 and message in payload["error"] and spawned == []


def test_launch_refuses_while_a_run_is_in_progress(launcher):
    server.RUNS["x"] = {"code": None}
    status, payload, spawned = launcher({"type": "synthetic", **SYNTH})
    assert status == 409 and "already in progress" in payload["error"] and spawned == []


def test_public_view_hides_the_process():
    assert server.public_view({"id": "a", "proc": object()}) == {"id": "a"}


# ---------------------------------------------------------------- form defaults, and the key of the terminal
@pytest.mark.parametrize("a,b,expected", [
    ("http://127.0.0.1:8000", "http://127.0.0.1:8000", True),
    ("http://127.0.0.1:8000/", "http://127.0.0.1:8000", True),             # a trailing slash does not matter
    (" HTTP://LOCALHOST:8000 ", "http://localhost:8000", True),            # nor case, nor spaces around
    ("http://host", "http://host:80", True),                               # nor a default port
    ("https://host", "https://host:443/", True),
    ("http://127.0.0.1:8000", "http://localhost:8000", False),             # two names for the same machine: not the same address
    ("http://127.0.0.1:8000", "http://127.0.0.1:8001", False),
    ("http://127.0.0.1:8000", "https://127.0.0.1:8000", False),
    ("http://h:8000/v1", "http://h:8000", False),
    ("", "", False), (None, "http://h:1", False), ("not a url", "not a url", False), ("http://h:1", None, False),
])
def test_same_url(a, b, expected):
    assert server.same_url(a, b) is expected


ENV_URL, ENV_KEY = "http://127.0.0.1:8000", "sk-from-the-terminal"


@pytest.fixture
def terminal(monkeypatch):
    """The terminal that started the interface had loaded target.env (tests/conftest.py clears BENCH_* otherwise)."""
    monkeypatch.setenv("BENCH_URL", ENV_URL)
    monkeypatch.setenv("BENCH_API_KEY", ENV_KEY)


def test_the_key_of_the_form_wins(terminal):
    assert server.target_key({"url": ENV_URL, "api_key": "sk-typed"}) == "sk-typed"


@pytest.mark.parametrize("url", [ENV_URL, ENV_URL + "/", " http://127.0.0.1:8000"])
def test_the_key_of_the_terminal_is_used_for_the_target_of_the_terminal(terminal, url):
    assert server.target_key({"url": url}) == ENV_KEY
    assert server.target_key({"url": url, "api_key": ""}) == ENV_KEY


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001", "http://localhost:8000", "https://127.0.0.1:8000", "http://10.0.0.5:8000", "", None])
def test_the_key_of_the_terminal_never_goes_to_another_target(terminal, url):
    assert server.target_key({"url": url}) == ""


def test_no_key_without_a_url_in_the_terminal(monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", ENV_KEY)
    assert server.target_key({"url": ENV_URL}) == ""


def test_no_key_without_a_key_in_the_terminal(monkeypatch):
    monkeypatch.setenv("BENCH_URL", ENV_URL)
    assert server.target_key({"url": ENV_URL}) == ""


def test_the_key_of_the_terminal_is_stripped_and_a_bad_one_is_ignored(monkeypatch):
    monkeypatch.setenv("BENCH_URL", ENV_URL)
    monkeypatch.setenv("BENCH_API_KEY", "  sk-1\n")
    assert server.target_key({"url": ENV_URL}) == "sk-1"
    monkeypatch.setenv("BENCH_API_KEY", "bad key with spaces")
    assert server.target_key({"url": ENV_URL}) == ""


def test_a_generic_variable_is_not_a_key(monkeypatch):
    monkeypatch.setenv("BENCH_URL", ENV_URL)
    monkeypatch.setenv("API_KEY", "must-not-be-used")
    assert server.target_key({"url": ENV_URL}) == ""


def test_the_commands_use_the_key_of_the_terminal_only_for_its_target(terminal):
    same = {"url": ENV_URL}
    other = {"url": "http://127.0.0.1:8001"}
    assert server.synthetic_command({**SYNTH, **same})[1]["BENCH_API_KEY"] == ENV_KEY
    assert server.synthetic_command({**SYNTH, **other})[1]["BENCH_API_KEY"] == ""
    assert server.agentx_command({**AGENTX, **same})[1]["BENCH_API_KEY"] == ENV_KEY
    assert server.agentx_command({**AGENTX, **other})[1]["BENCH_API_KEY"] == ""
    assert server.agents_command({**AGENTS, **same})[1]["BENCH_API_KEY"] == ENV_KEY
    assert "BENCH_API_KEY" not in server.agents_command({**AGENTS, **other})[1]


def test_a_launch_gets_the_key_of_the_terminal_for_its_target(launcher, terminal):
    status, _, spawned = launcher({"type": "agentx", **AGENTX, "url": ENV_URL})
    assert status == 200 and spawned[0][1]["env"]["BENCH_API_KEY"] == ENV_KEY


def test_the_key_of_the_terminal_is_hidden_in_the_displayed_command(launcher, terminal):
    status, payload, _ = launcher({"type": "agentx", **AGENTX, "url": ENV_URL})
    assert status == 200 and ENV_KEY not in json.dumps(payload)


def defaults_payload():
    return json.loads(asyncio.run(server.defaults(None)).text)


def test_defaults_come_from_the_terminal_without_any_key(monkeypatch):
    monkeypatch.setenv("BENCH_URL", ENV_URL)
    monkeypatch.setenv("BENCH_MODEL", "my-model")
    monkeypatch.setenv("BENCH_METRICS", "http://h1:8000/metrics http://h2:8000/metrics")
    monkeypatch.setenv("BENCH_TOKENIZER", "tools/tokenizer")
    monkeypatch.setenv("BENCH_API_KEY", ENV_KEY)
    monkeypatch.setenv("BENCH_PROMETHEUS_KEY", "prom-secret")
    monkeypatch.setenv("BENCH_AGENT_KEYS", "pentest=agent-secret")
    payload = defaults_payload()
    assert payload == {"url": ENV_URL, "model": "my-model", "metrics": "http://h1:8000/metrics http://h2:8000/metrics",
                       "tokenizer": "tools/tokenizer", "prometheus": "", "litellm": "",
                       "api_key_set": True, "api_key_url": ENV_URL}
    text = json.dumps(payload)
    assert ENV_KEY not in text and "prom-secret" not in text and "agent-secret" not in text


def test_defaults_are_empty_without_variables():
    assert defaults_payload() == {"url": "", "model": "", "metrics": "", "tokenizer": "", "prometheus": "", "litellm": "",
                                "api_key_set": False, "api_key_url": ""}


def test_defaults_strip_white_space_and_ignore_generic_names(monkeypatch):
    monkeypatch.setenv("BENCH_MODEL", "  my-model \n")
    monkeypatch.setenv("URL", "http://must-not-be-used")
    monkeypatch.setenv("MODEL", "must-not-be-used")
    payload = defaults_payload()
    assert payload["model"] == "my-model" and payload["url"] == ""


def test_a_key_in_the_terminal_without_a_url_has_no_target(monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", ENV_KEY)
    payload = defaults_payload()
    assert payload["api_key_set"] is True and payload["api_key_url"] == ""


# ------------------------------------------- the observation endpoint returns the rows of the chosen view
def run_observation(monkeypatch, tmp_path, body, csv_text):
    """POST /api/observation with a fake `observation.query` that writes csv_text where --view-output says."""
    seen = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b"Highest concurrency meeting the target\n", None

    async def fake_exec(*cmd, **kw):
        seen.append(cmd)
        pathlib.Path(cmd[cmd.index("--view-output") + 1]).write_text(csv_text)
        return Process()
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(server, "RUNS", {})
    resp = asyncio.run(server.observation(FakeRequest(body)))
    return resp.status, json.loads(resp.text), seen


@pytest.mark.parametrize("view,csv_text,first_row", [
    ("windows", "start,verdict\n2026-11-16 18:00,holds\n", {"start": "2026-11-16 18:00", "verdict": "holds"}),
    ("period", "period,start_hour,windows\nnight (0-6 h),0,3\n", {"period": "night (0-6 h)", "start_hour": "0", "windows": "3"}),
    ("agents", "agent,requests\ncoder,240\n", {"agent": "coder", "requests": "240"}),
])
def test_observation_returns_the_rows_of_the_view(monkeypatch, tmp_path, view, csv_text, first_row):
    body = {**OBS, "view": view, **({"litellm": 'job="litellm"'} if view == "agents" else {})}
    status, payload, seen = run_observation(monkeypatch, tmp_path, body, csv_text)
    assert status == 200 and payload["view"] == view and payload["rows"] == [first_row]
    assert "--view-output" in seen[0] and "--output" not in seen[0]
    assert seen[0][seen[0].index("--view") + 1] == view
    assert payload["messages"] == ["Highest concurrency meeting the target"] and payload["code"] == 0


def test_observation_with_an_unknown_view_returns_the_windows(monkeypatch, tmp_path):
    status, payload, _ = run_observation(monkeypatch, tmp_path, {**OBS, "view": "x; ls"}, "start\n2026-11-16 18:00\n")
    assert status == 200 and payload["view"] == "windows"


def test_observation_refuses_the_agents_view_without_litellm(monkeypatch, tmp_path):
    status, payload, seen = run_observation(monkeypatch, tmp_path, {**OBS, "view": "agents"}, "")
    assert status == 400 and "LiteLLM" in payload["error"] and seen == []


def test_observation_cancelled_kills_the_query_and_leaves_no_file(monkeypatch, tmp_path):
    """The page asks again while a query runs: the handler is cancelled, the process killed, no CSV left behind."""
    procs = []

    class Hanging(FakeQuery):
        def __init__(self, cmd, kw):
            super().__init__(cmd, kw, hang=True)
            self.returncode = None      # still running

    async def fake_exec(*cmd, **kw):
        procs.append(Hanging(cmd, kw))
        return procs[-1]

    async def scenario():
        task = asyncio.ensure_future(server.observation(FakeRequest(OBS)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(server, "RUNS", {})
    asyncio.run(scenario())
    assert procs[0].killed and not list((tmp_path / "logs").glob("observation-*"))


def test_results_cancelled_kills_the_report(monkeypatch):
    """The page asks again while a report runs: the handler is cancelled and the process killed."""
    procs = []

    class Hanging(FakeQuery):
        def __init__(self, cmd, kw):
            super().__init__(cmd, kw, hang=True)
            self.returncode = None      # still running

    async def fake_exec(*cmd, **kw):
        procs.append(Hanging(cmd, kw))
        return procs[-1]

    async def scenario():
        task = asyncio.ensure_future(server.read_results(FakeGet(type="synthetic", root="results", target="30")))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(scenario())
    assert procs[0].killed


def test_results_of_the_same_folder_run_one_at_a_time(monkeypatch):
    """Two reports on one results folder must not overlap: `python -m report` rewrites table.csv there."""
    running, peak = [0], [0]

    class Slow(FakeQuery):
        async def communicate(self):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            await asyncio.sleep(0.05)
            running[0] -= 1
            return await super().communicate()

    async def fake_exec(*cmd, **kw):
        return Slow(cmd, kw)

    async def scenario():
        await asyncio.gather(*(server.read_results(FakeGet(type="synthetic", root="results", target=str(t)))
                               for t in (30, 31, 32)))
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "REPORT_LOCKS", {})
    asyncio.run(scenario())
    assert peak[0] == 1


def test_shielded_handler_finishes_even_if_the_client_goes_away():
    done = []

    async def handler(req):
        await asyncio.sleep(0.05)
        done.append(True)
        return "ok"

    async def scenario():
        task = asyncio.ensure_future(server.shielded(handler)(None))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
    asyncio.run(scenario())
    assert done == [True]


# ---------------------------------------------------------------- AgentX: traces generated by the interface
def test_agentx_generated_traces_command():
    cmd, env, cwd = server.agentx_command({**AGENTX, "generate": "small", "variant": "256k", "traces_dir": "/nonexistent-xyz"})
    assert cmd[:3] == ["bash", "-c", server.GENERATE_THEN_AGENTX] and cwd == server.ROOT
    assert cmd[3:] == ["generate-traces", "traces-small", sys.executable, "small", "bash", str(server.ROOT / "agentx.sh"), "ax1", "1 2"]
    assert env["BENCH_AGENTX_DIR"] == "traces-small" and env["BENCH_AGENTX_VARIANT"] == ""


@pytest.mark.parametrize("preset", ["huge", "../x", "small; ls", "-h"])
def test_agentx_generated_traces_refuses_an_unknown_preset(preset):
    with pytest.raises(Refusal):
        server.agentx_command({**AGENTX, "generate": preset})


def test_agentx_without_generation_keeps_the_plain_command():
    cmd, _, _ = server.agentx_command({**AGENTX, "generate": ""})
    assert cmd == ["bash", str(server.ROOT / "agentx.sh"), "ax1", "1 2"]


def test_generate_then_agentx_script_writes_the_traces_once_then_replays(tmp_path):
    """The real shell script, with a stand-in for agentx.sh: first run generates, second run reuses."""
    import subprocess
    fake = tmp_path / "agentx.sh"
    fake.write_text('echo "AGENTX $@"\n')
    folder = tmp_path / "traces-tiny"

    def run():
        return subprocess.run(["bash", "-c", server.GENERATE_THEN_AGENTX, "generate-traces", str(folder), sys.executable,
                               "tiny", "bash", str(fake), "lbl", "1 2"], cwd=server.ROOT, capture_output=True, text=True, timeout=120)
    first, second = run(), run()
    assert first.returncode == 0 and "generating" in first.stdout and "AGENTX lbl 1 2" in first.stdout
    assert list(folder.glob("trace-*.json"))
    assert second.returncode == 0 and "reusing" in second.stdout and "generating" not in second.stdout


def test_generate_then_agentx_script_stops_when_the_generation_fails(tmp_path):
    import subprocess
    fake = tmp_path / "agentx.sh"
    fake.write_text('echo "AGENTX"\n')
    out = subprocess.run(["bash", "-c", server.GENERATE_THEN_AGENTX, "generate-traces", str(tmp_path / "t"), sys.executable,
                          "nonexistent-preset", "bash", str(fake), "l", "1"], cwd=server.ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode != 0 and "AGENTX" not in out.stdout


def test_generate_then_agentx_script_ignores_a_stale_half_written_folder(tmp_path):
    """A generation that was stopped leaves at most a `.tmp` folder: the next run starts again and the final folder is whole."""
    import subprocess
    fake = tmp_path / "agentx.sh"
    fake.write_text('echo "AGENTX"\n')
    folder = tmp_path / "traces-tiny"
    (tmp_path / "traces-tiny.tmp").mkdir()
    (tmp_path / "traces-tiny.tmp" / "trace-0001.json").write_text("{}")
    out = subprocess.run(["bash", "-c", server.GENERATE_THEN_AGENTX, "generate-traces", str(folder), sys.executable, "tiny",
                          "bash", str(fake), "l", "1"], cwd=server.ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "AGENTX" in out.stdout
    assert len(list(folder.glob("trace-*.json"))) == 30 and not (tmp_path / "traces-tiny.tmp").exists()


def test_display_command_says_what_a_generated_run_does():
    cmd, _, _ = server.agentx_command({**AGENTX, "generate": "small"})
    shown = server.display_command(cmd)
    assert shown[0].startswith("[writes traces-small with the small preset") and shown[1:] == cmd[7:]
    plain, _, _ = server.agentx_command(AGENTX)
    assert server.display_command(plain) == plain


def test_defaults_give_the_prometheus_address_and_the_litellm_selector(monkeypatch):
    monkeypatch.setenv("BENCH_PROMETHEUS", "http://prometheus:9090")
    monkeypatch.setenv("BENCH_LITELLM_SELECTOR", 'job="litellm"')
    payload = defaults_payload()
    assert payload["prometheus"] == "http://prometheus:9090" and payload["litellm"] == 'job="litellm"'


# ---------------------------------------------------------------- the interface in a container
class GuardRequest:
    def __init__(self, host, token="t"):
        self.path, self.host, self.method, self.content_type = "/api/runs", host, "GET", "application/json"
        self.headers = {"X-Bench-Token": token}


def run_guard(host, monkeypatch, public_port):
    monkeypatch.setattr(server, "PUBLIC_PORT", [public_port])
    monkeypatch.setattr(server, "TOKEN", "t")

    async def handler(req):
        return server.web.json_response({"ok": True})
    return asyncio.run(server.guard(GuardRequest(host), handler)).status


def test_guard_checks_the_host_against_the_port_the_browser_uses(monkeypatch):
    assert run_guard("127.0.0.1:9000", monkeypatch, 9000) == 200
    assert run_guard("localhost:9000", monkeypatch, 9000) == 200
    assert run_guard("127.0.0.1:8765", monkeypatch, 9000) == 403      # the port inside the container is not the browser's
    assert run_guard("evil.example:9000", monkeypatch, 9000) == 403


@pytest.mark.parametrize("argv,host,port,public", [
    ([], "127.0.0.1", 8765, 8765),
    (["--port", "9000"], "127.0.0.1", 9000, 9000),
    (["--host", "0.0.0.0", "--port", "8765", "--public-port", "9100"], "0.0.0.0", 8765, 9100),
])
def test_main_listens_where_it_is_told(monkeypatch, argv, host, port, public):
    seen = {}
    monkeypatch.setattr(server.web, "run_app", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr(server.settings, "load_target_env", lambda path: [])
    monkeypatch.setattr(sys, "argv", ["server"] + argv)
    monkeypatch.setattr(server, "PORT", [8765])
    monkeypatch.setattr(server, "PUBLIC_PORT", [8765])
    server.main()
    assert seen["host"] == host and seen["port"] == port and server.PUBLIC_PORT[0] == public
