"""Input validators and command builders of interface/server.py: the security boundary of the interface."""
import asyncio
import json
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


OBS = {"metrics": "http://h:8000/metrics", "label": "prod", "duration": 900}


def test_observation_command_minimal():
    cmd, env, cwd = server.observation_command(OBS)
    assert cmd == [sys.executable, "-m", "observation.observer", "--metrics", "http://h:8000/metrics", "--duration", "900",
                   "--label", "prod", "--output", str(server.ROOT / "results-observation")]
    assert env == {} and cwd == server.ROOT


def test_observation_command_all_options():
    cmd, env, _ = server.observation_command({**OBS, "interval": 60, "level": 4, "litellm": "http://l:4000/metrics/",
                                              "per": "metadata_agent", "fingerprint": True, "url": "http://h:8000",
                                              "model": "bench", "api_key": "k1", "litellm_key": "k2"})
    joined = " ".join(cmd)
    assert "--interval 60" in joined and "--level 4" in joined and "--by metadata_agent" in joined
    assert "--url http://h:8000 --model bench" in joined
    assert env == {"BENCH_API_KEY": "k1", "BENCH_LITELLM_KEY": "k2"}


def test_observation_command_default_litellm_label_and_zero_level():
    cmd, _, _ = server.observation_command({**OBS, "litellm": "http://l:4000/metrics", "level": 0})
    assert cmd[cmd.index("--by") + 1] == "api_key_alias" and cmd[cmd.index("--level") + 1] == "0"


@pytest.mark.parametrize("change", [
    {"metrics": ""}, {"metrics": "ftp://x"}, {"duration": 59}, {"duration": 7 * 86400 + 1}, {"duration": "x"},
    {"interval": 59}, {"interval": 901}, {"level": -1}, {"level": "x"}, {"label": "-x"}, {"root": "/abs"},
    {"litellm": "ftp://x"}, {"litellm": "http://l/m;ls"}, {"litellm": "http://l:4000/metrics", "per": "bad label"},
    {"litellm": "http://l:4000/metrics", "per": "metadata_"}, {"litellm": "http://l:4000/metrics", "per": "x; ls"},
    {"fingerprint": True}, {"fingerprint": True, "url": "http://h:8000", "model": "-x"}, {"api_key": "a b"},
])
def test_observation_command_rejects(change):
    with pytest.raises(Refusal):
        server.observation_command({**OBS, **change})


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
    status, body, _ = launcher({"type": "observation", **OBS, "label": "o2", "litellm_key": "sk-C"})
    assert status == 200 and "sk-C" not in server.RUNS[body["id"]]["command"]


def test_secret_names_are_the_bench_names():
    assert set(server.SECRETS) == {"BENCH_API_KEY", "BENCH_AGENT_KEYS", "BENCH_LITELLM_KEY"}


def test_launch_does_not_inherit_the_terminal_secrets(launcher, monkeypatch):
    monkeypatch.setenv("BENCH_API_KEY", "from-terminal")
    _, _, spawned = launcher({"type": "agentx", **AGENTX})
    assert spawned[0][1]["env"]["BENCH_API_KEY"] == ""


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
