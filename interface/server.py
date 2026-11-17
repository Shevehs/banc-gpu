"""Bench interface: a small local server that launches the bench scripts and reads their results.

Usage (from the repository root, with the .venv-aiperf environment activated):
    python -m interface.server            # then open http://127.0.0.1:8765
    python -m interface.server --port 9000
On a remote machine: SSH tunnel from your workstation, e.g. ssh -L 8765:127.0.0.1:8765 <machine>.

The server only listens on 127.0.0.1 and requires a random token, shown at startup in the address to
open: another web page or another user of the machine cannot start a run without it.
It also checks the Host header and the origin of requests. It only runs sweep.sh, agentx.sh,
agents_test/launcher.py and the report scripts (including observation/query.py, which queries Prometheus), with
validated parameters (AIPerf options on an allowlist); a run started here is identical to the same run started by
hand (the exact command is written in its log, without the API key). One run at a time; the observation is not a
run: it reads Prometheus and can be displayed while a run is in progress.
Dependency: aiohttp (already installed with AIPerf).
"""
import argparse
import asyncio
import csv
import ipaddress
import os
import re
import secrets
import shlex
import signal
import socket
import sys
import time
import urllib.parse
from pathlib import Path

from aiohttp import web

from tools import settings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = HERE / "dist"
LOGS_DIR = HERE / "logs"
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
TOKEN = os.environ.get("BENCH_TOKEN") or secrets.token_urlsafe(24)
PORT = [8765]
# AIPerf options allowed in "extra options": name -> number of expected values
EXTRA_OK = {"--use-server-token-count": 0, "--benchmark-grace-period": 1, "--prompt-corpus": 1,
            "--num-dataset-entries": 1, "--request-timeout-seconds": 1}
LEVELS_OK = re.compile(r"^\s*\d{1,4}(\s+\d{1,4}){0,30}\s*$")
RUNS = {}  # id -> dict(type, command, start, end, code, log, proc)
COUNTER = [0]


class Refusal(Exception):
    """Invalid request parameter: its message is returned to the client with a 400 status."""


MODEL_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,199}$")


def valid_model(v):
    """Served model name or gateway alias (may contain / and :, e.g. Qwen/Qwen3-0.6B)."""
    v = str(v or "").strip()
    if not MODEL_OK.match(v) or ".." in v:
        raise Refusal("model: letters, digits and . _ - / : @, starting with a letter or a digit")
    return v


def valid_name(v, field_name):
    """Label or name checked against NAME_OK; returns the stripped value.

    Raises:
        Refusal: if it is not valid (`field_name` names the field in the message)."""
    v = str(v or "").strip()
    if not NAME_OK.match(v) or ".." in v:
        raise Refusal(f"{field_name}: letters, digits, dot, hyphen or underscore, starting with a letter or a digit")
    return v


def valid_extra(v):
    """Extra AIPerf options, restricted to an allowlist."""
    try:
        words = shlex.split(str(v or ""))
    except ValueError as e:
        raise Refusal(f"extra options: {e}") from e
    i = 0
    while i < len(words):
        m = words[i]
        if m not in EXTRA_OK:
            raise Refusal(f"AIPerf option not allowed: {m} (allowed: {', '.join(EXTRA_OK)})")
        n = EXTRA_OK[m]
        vals = words[i + 1:i + 1 + n]
        if len(vals) < n or any(not re.match(r"^[A-Za-z0-9._-]{1,40}$", x) for x in vals):
            raise Refusal(f"invalid value for {m}")
        i += 1 + n
    return " ".join(words)


def valid_number(v, field_name, mini, maxi, integer=False):
    """Number between `mini` and `maxi` (an int if `integer`).

    Raises:
        Refusal: if `v` is not a number or is out of range."""
    try:
        x = int(v) if integer else float(v)
    except (TypeError, ValueError):
        raise Refusal(f"{field_name}: number expected") from None
    if not mini <= x <= maxi:
        raise Refusal(f"{field_name}: between {mini} and {maxi}")
    return x


def _addresses(host):
    """IP addresses of `host` (a literal address, or the result of a DNS lookup); empty if it does not resolve."""
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        return [ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, None)]
    except (socket.gaierror, UnicodeError, ValueError):
        return []


def _check_private_host(url, what):
    """Refuse a URL whose host is not on this machine or a private network.

    The target receives the API key and the metrics URLs are fetched by this server, so by default only loopback,
    private (RFC 1918 / unique local) and link-local addresses are accepted; a host name must resolve only to such
    addresses. Start the interface with BENCH_ALLOW_PUBLIC_URLS=1 to allow public hosts.

    Raises:
        Refusal: if the host does not resolve or any of its addresses is public."""
    if os.environ.get("BENCH_ALLOW_PUBLIC_URLS") == "1":
        return
    host = re.match(r"^https?://([A-Za-z0-9.\-]+)", url).group(1)
    addresses = _addresses(host)
    if not addresses:
        raise Refusal(f"{what}: host {host} does not resolve")
    if not all(a.is_loopback or a.is_private or a.is_link_local for a in addresses):
        raise Refusal(f"{what}: host {host} is not on a local or private network "
                      "(start the interface with BENCH_ALLOW_PUBLIC_URLS=1 to allow public hosts)")


def valid_local_url(v):
    """Target URL of the form http(s)://host[:port] on this machine or a private network, without a trailing slash.

    Raises:
        Refusal: if the form is not respected or the host is public (see _check_private_host)."""
    v = str(v or "").strip()
    if not re.match(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?/?$", v):
        raise Refusal("target URL: expected form http://host:port")
    _check_private_host(v, "target URL")
    return v.rstrip("/")


def valid_metrics_urls(v):
    """vLLM /metrics URLs, separated by spaces (optional), on this machine or a private network.

    Raises:
        Refusal: if a URL has the wrong form or a public host."""
    urls = str(v or "").split()
    for u in urls:
        if not re.match(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?(/[A-Za-z0-9._/-]*)?$", u):
            raise Refusal(f"metrics URL: expected form http://host:port/metrics ({u})")
        _check_private_host(u, "metrics URL")
    return urls


def valid_key(v):
    """API key (printable ASCII, 1 to 500 characters), stripped; an empty string if none.

    Raises:
        Refusal: if it contains characters that are not allowed."""
    v = str(v or "").strip()
    if v and not re.match(r"^[\x21-\x7e]{1,500}$", v):
        raise Refusal("API key: characters not allowed")
    return v


def _address(url):
    """(scheme, host, port, path) of a URL typed by a person, compared without case and without a trailing slash."""
    parts = urllib.parse.urlsplit(str(url or "").strip())
    scheme = parts.scheme.lower()
    try:
        port = parts.port or {"http": 80, "https": 443}.get(scheme)
    except ValueError:
        port = None
    return scheme, (parts.hostname or "").lower(), port, parts.path.rstrip("/")


def same_url(a, b):
    """True if two target URLs are the same address (scheme, host, port and path; case and a trailing slash ignored).

    "localhost" and "127.0.0.1" are not the same address here: when in doubt, the answer is no."""
    left = _address(a)
    return bool(left[0] and left[1]) and left == _address(b)


def target_key(p):
    """The API key to use for the target of the form `p`.

    The key typed in the form wins. Otherwise BENCH_API_KEY of the terminal that started the interface is used, but only
    when the form's target URL is the BENCH_URL of that terminal: a key meant for one server must never go to another.
    The key never goes to the browser. An unusable key in the terminal is ignored."""
    key = valid_key(p.get("api_key"))
    if key:
        return key
    env_key, env_url = settings.api_key(), settings.read_setting("URL")
    if env_key and env_url and same_url(p.get("url"), env_url):
        try:
            return valid_key(env_key)
        except Refusal:
            return ""
    return ""


SECRETS = ("BENCH_API_KEY", "BENCH_AGENT_KEYS", "BENCH_PROMETHEUS_KEY")


def results_dir(v):
    """Results directory relative to the repository root, without parent traversal."""
    v = str(v or "").strip()
    if not v or not re.match(r"^[A-Za-z0-9._/-]{1,120}$", v) or ".." in v.split("/") or v.startswith("/"):
        raise Refusal("directory: path relative to the repository expected")
    return v


def synthetic_command(p):
    """Command of a synthetic sweep (sweep.sh) from the form fields `p`.

    Returns (command, environment, working directory); every variable read by sweep.sh is set.

    Raises:
        Refusal: if a field is invalid."""
    env = {
        "BENCH_URL": valid_local_url(p.get("url")), "BENCH_MODEL": valid_model(p.get("model")),
        "BENCH_ISL": str(valid_number(p.get("isl"), "input tokens", 1, 1_000_000, True)),
        "BENCH_OSL": str(valid_number(p.get("osl"), "output tokens", 1, 100_000, True)),
        "BENCH_TEMPERATURE": str(valid_number(p.get("temperature", 0.7), "temperature", 0, 2)),
        "BENCH_ROOT_DIR": results_dir(p.get("root") or "results"),
    }
    tok = str(p.get("tokenizer") or "").strip()
    if tok:
        if tok.startswith("-") or not re.match(r"^[A-Za-z0-9._/@:~-]{1,300}$", tok):
            raise Refusal("tokenizer: Hugging Face name or local path expected")
        env["BENCH_TOKENIZER"] = os.path.expanduser(tok)
    env.setdefault("BENCH_TOKENIZER", "")  # empty = the repository's default tokenizer
    # All variables read by sweep.sh are set here: nothing is inherited from the terminal.
    env["BENCH_METRICS"] = " ".join(valid_metrics_urls(p.get("metrics")))
    env["BENCH_API_KEY"] = target_key(p)
    if p.get("duration"):
        env["BENCH_DURATION"] = str(valid_number(p.get("duration"), "duration", 10, 7200, True)); env["BENCH_REQUESTS_PER_CONCURRENCY"] = "8"
    else:
        env["BENCH_DURATION"] = ""
        env["BENCH_REQUESTS_PER_CONCURRENCY"] = str(valid_number(p.get("requests_per_concurrency", 8), "requests per concurrency", 1, 1000, True))
    env["BENCH_EXTRA"] = valid_extra(p.get("extra")) or ""
    env.update(BENCH_OVERWRITE="", BENCH_SEED="")
    levels = str(p.get("levels") or "")
    if not LEVELS_OK.match(levels):
        raise Refusal("levels: integers separated by spaces")
    cmd = ["bash", str(ROOT / "sweep.sh"), valid_name(p.get("label"), "label"), " ".join(levels.split())]
    return cmd, env, ROOT


def _allowed_dirs():
    """Directories under which a user-supplied path must lie: the repository, the home directory, and any
    directory listed in BENCH_ALLOWED_DIRS (separated like PATH)."""
    extra = [d for d in os.environ.get("BENCH_ALLOWED_DIRS", "").split(os.pathsep) if d]
    return [Path(d).expanduser().resolve() for d in [str(ROOT), "~", *extra]]


def local_path(v, field_name):
    """Path of an existing directory on the interface machine (absolute, ~ or relative to the repository),
    which must lie under the repository, the home directory or BENCH_ALLOWED_DIRS once symlinks and `..` are resolved.

    Raises:
        Refusal: if the path is malformed, missing, not a directory, or outside the allowed directories."""
    v = str(v or "").strip()
    if not re.match(r"^[A-Za-z0-9._/~-]{1,300}$", v) or v.startswith("-"):
        raise Refusal(f"{field_name}: directory path expected")
    c = Path(os.path.expanduser(v))
    c = (c if c.is_absolute() else ROOT / c).resolve()
    if not c.is_dir():
        raise Refusal(f"{field_name}: directory not found ({c})")
    if not any(c == d or d in c.parents for d in _allowed_dirs()):
        raise Refusal(f"{field_name}: directory outside the allowed locations "
                      "(the repository, your home directory, or BENCH_ALLOWED_DIRS)")
    return str(c)


def agentx_command(p):
    """AgentX stress test (agentx.sh); all its variables are set here."""
    short = bool(p.get("short_run"))
    duration = valid_number(p.get("duration", 1800), "duration per level", 30 if short else 900, 7200, True)
    env = {"BENCH_URL": valid_local_url(p.get("url")), "BENCH_MODEL": valid_model(p.get("model")),
           "BENCH_METRICS": " ".join(valid_metrics_urls(p.get("metrics"))), "BENCH_API_KEY": target_key(p),
           "BENCH_DURATION": str(duration), "BENCH_ROOT_DIR": results_dir(p.get("root") or "results-agentx"),
           "BENCH_AGENTX_DIR": local_path(p.get("traces_dir"), "traces directory") if p.get("traces_dir") else "",
           "BENCH_SHORT_RUN": "1" if short else "", "BENCH_OVERWRITE": "", "BENCH_SEED": "", "BENCH_TOKENIZER": "",
           "BENCH_MAX_CONTEXT": str(valid_number(p.get("max_context"), "max context", 1024, 10_000_000, True)) if p.get("max_context") else "",
           "BENCH_AGENTX_VARIANT": "256k" if p.get("variant") == "256k" else ""}
    tok = str(p.get("tokenizer") or "").strip()
    if tok:
        if tok.startswith("-") or not re.match(r"^[A-Za-z0-9._/@:~-]{1,300}$", tok):
            raise Refusal("tokenizer: Hugging Face name or local path expected")
        env["BENCH_TOKENIZER"] = os.path.expanduser(tok)
    levels = str(p.get("levels") or "")
    if not LEVELS_OK.match(levels):
        raise Refusal("levels: integers separated by spaces")
    cmd = ["bash", str(ROOT / "agentx.sh"), valid_name(p.get("label"), "label"), " ".join(levels.split())]
    return cmd, env, ROOT


def agents_command(p):
    """Command of an agent-launcher run (agents_test.launcher) from the form fields `p`.

    Returns (command, environment, working directory); keys go in the environment, never in the command.

    Raises:
        Refusal: if a field is invalid."""
    agents = [a for a in (p.get("agents") or []) if a in ("code_review", "pentest", "report")]
    if not agents:
        raise Refusal("agents: at least one of code_review, pentest, report")
    levels = str(p.get("levels") or "")
    if not LEVELS_OK.match(levels):
        raise Refusal("levels: integers separated by spaces")
    mode = p.get("mode") if p.get("mode") in ("scripted", "autonomous") else None
    if not mode:
        raise Refusal("mode: scripted or autonomous")
    key = target_key(p)
    env = {"BENCH_API_KEY": key} if key else {}
    pairs = []
    for pair in str(p.get("agent_keys") or "").split(","):
        if pair.strip():
            k_, _, v_ = pair.partition("=")
            if k_.strip() not in ("code_review", "pentest", "report") or not valid_key(v_):
                raise Refusal("per-agent keys: code_review=<key>,pentest=<key>,report=<key>")
            pairs.append(f"{k_.strip()}={valid_key(v_)}")
    if pairs:
        env["BENCH_AGENT_KEYS"] = ",".join(pairs)
    met = valid_metrics_urls(p.get("metrics"))
    cmd = [sys.executable, "-m", "agents_test.launcher",
           "--url", valid_local_url(p.get("url")), "--model", valid_model(p.get("model")),
           "--agents", ",".join(agents), "--levels", " ".join(levels.split()),
           "--duration", str(valid_number(p.get("duration"), "duration", 10, 7200)),
           "--warmup", str(valid_number(p.get("warmup", 60), "warmup", 0, 3600)),
           "--scale", str(valid_number(p.get("scale", 1), "scale", 0.01, 2)),
           "--latency-factor", str(valid_number(p.get("latency_factor", 1), "latency factor", 0, 10)),
           "--mode", mode, "--label", valid_name(p.get("label"), "label"),
           "--output", str(ROOT / results_dir(p.get("root") or "agents_test/results-agents"))]
    if p.get("label_agents"):
        cmd.append("--label-agents")
    if met:
        cmd += ["--metrics", " ".join(met)]
    return cmd, env, ROOT


# Only these terminal variables are passed to runs; the bench settings come from the interface fields.
INHERITED_ENV = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TERM", "VIRTUAL_ENV", "HF_HOME",
              "HF_HUB_CACHE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
              "https_proxy", "http_proxy", "no_proxy", "DO_NOT_TRACK", "TZ"}


def base_environment():
    """Terminal variables passed to the runs: those in INHERITED_ENV and LC_*."""
    return {k: v for k, v in os.environ.items() if k in INHERITED_ENV or k.startswith("LC_")}


async def follow(eid, proc, f):
    """Wait for the process of run `eid`, write its exit code at the end of its log `f` (then close it)
    and record the end time and code in RUNS."""
    code = await proc.wait()
    f.write(f"\n== end, exit code {code} ==\n"); f.close()
    RUNS[eid].update(end=time.time(), code=code)


async def read_body(req):
    """JSON object of the request body.

    Raises:
        Refusal: if the body is not a JSON object."""
    try:
        p = await req.json()
    except Exception:  # noqa: BLE001
        raise Refusal("JSON body expected") from None
    if not isinstance(p, dict):
        raise Refusal("JSON body expected")
    return p


async def launch(req):
    """POST /api/runs: validate the parameters and start the run in a subprocess of its own session.

    Returns {"id"}; 409 if a run is already in progress, 400 if the parameters are refused."""
    if any(e["code"] is None for e in RUNS.values()):
        return web.json_response({"error": "a run is already in progress: wait for it to finish or stop it"}, status=409)
    try:
        p = await read_body(req)
    except Refusal as e:
        return web.json_response({"error": str(e)}, status=400)
    try:
        factory = {"synthetic": synthetic_command, "agents": agents_command,
                   "agentx": agentx_command}.get(p.get("type"))
        if not factory:
            raise Refusal("unknown run type")
        cmd, env, cwd = factory(p)
    except Refusal as e:
        return web.json_response({"error": str(e)}, status=400)
    LOGS_DIR.mkdir(exist_ok=True)
    COUNTER[0] += 1
    eid = f"{time.strftime('%Y%m%d-%H%M%S')}-{COUNTER[0]:03d}-{p.get('label', '')}"
    log = LOGS_DIR / f"{eid}.log"
    f = open(log, "w", encoding="utf-8")
    display = " ".join(f"{k}={'***' if k in SECRETS else repr(v) if ' ' in v else v}" for k, v in env.items()) \
        + (" " if env else "") + " ".join(f"'{c}'" if " " in c else c for c in cmd)
    f.write(f"$ cd {cwd}\n$ {display}\n\n"); f.flush()
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), env={**base_environment(), **env, "PYTHONUNBUFFERED": "1"},
                                                stdout=f, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    RUNS[eid] = dict(id=eid, type=p.get("type"), label=p.get("label"), command=display,
                       start=time.time(), end=None, code=None, log=str(log), proc=proc)
    asyncio.create_task(follow(eid, proc, f))
    return web.json_response({"id": eid})


def public_view(e):
    """Run record without its process object (serializable to JSON)."""
    return {k: v for k, v in e.items() if k != "proc"}


async def list_runs(req):
    """GET /api/runs: the runs of this session, most recent first."""
    return web.json_response(sorted((public_view(e) for e in RUNS.values()), key=lambda e: -e["start"]))


async def run_log(req):
    """GET /api/runs/{eid}/log?since=<bytes>: next chunk (up to 200,000 bytes) of the log, the new position and
    whether the run has finished."""
    e = RUNS.get(req.match_info["eid"])
    if not e:
        return web.json_response({"error": "unknown run"}, status=404)
    try:
        since = max(0, int(req.query.get("since", "0")))
    except ValueError:
        since = 0
    with open(e["log"], "rb") as fh:
        fh.seek(since); chunk = fh.read(200_000)
    return web.json_response({"text": chunk.decode("utf-8", "replace"), "position": since + len(chunk),
                              "finished": e["code"] is not None})


async def stop_run(req):
    """POST /api/runs/{eid}/stop: send SIGTERM to the process group of a run in progress."""
    e = RUNS.get(req.match_info["eid"])
    if not e or e["code"] is not None:
        return web.json_response({"error": "unknown or finished run"}, status=400)
    try:
        os.killpg(e["proc"].pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return web.json_response({"ok": True})


async def list_roots(req):
    """Results directories present, by type (judged by their content)."""
    synthetic, agents, agentx = [], [], []
    for d in sorted(ROOT.glob("results*")) + sorted((ROOT / "agents_test").glob("results*")):
        if d.is_dir():
            rel = str(d.relative_to(ROOT))
            if any(d.glob("*/c*/profile_export_aiperf.json")):
                (agentx if any(d.glob("*/agentx.txt")) else synthetic).append(rel)
            if any(d.glob("*/n*/level.json")):
                agents.append(rel)
    return web.json_response({"synthetic": synthetic, "agents": agents, "agentx": agentx})


async def read_results(req):
    """GET /api/results: run `python -m report` on a results directory.

    `gpu` is optional: without it, `python -m report` deduces the GPU count of each run from its fingerprint.
    Returns the table rows (CSV), the key messages, the full output, the exit code and the command."""
    q = req.query
    try:
        root = results_dir(q.get("root"))
        target = valid_number(q.get("target"), "target", 0.1, 10_000)
        gpu = valid_number(q.get("gpu"), "GPU", 1, 4096, True) if q.get("gpu") not in (None, "") else None
    except Refusal as e:
        return web.json_response({"error": str(e)}, status=400)
    kind_name = q.get("type")
    type_, table = {"agents": ("agents", "table_agents.csv")}.get(kind_name, ("aiperf", "table.csv"))
    cmd = [sys.executable, "-m", "report", type_, str(ROOT / root), "--target", str(target)]
    if gpu:
        cmd += ["--gpu", str(gpu)]
    if q.get("require") == "1":
        cmd.append("--require-metrics")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    output = (await proc.communicate())[0].decode("utf-8", "replace")
    file = ROOT / root / table
    rows = []
    if proc.returncode == 0 and file.exists():
        with open(file, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    # Keep in sync with the wording printed by report/*: "selected ...", "that holds", "WARNING", "no ...",
    # "GPUs: <run>: ... deduced from the fingerprint", "ERROR: ..."
    messages = [l for l in output.splitlines() if "selected" in l or "WARNING" in l or "that holds" in l
                or l.startswith("GPUs:") or l.startswith("ERROR") or re.search(r"\bno\b", l.lower())]
    return web.json_response({"rows": rows, "messages": messages, "output": output, "code": proc.returncode,
                              "command": " ".join(cmd)})


SELECTOR_OK = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*"'
                         r'(\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*")*\s*$')
DURATION_OK = re.compile(r"^\d{1,6}[smhd]?$")
DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")
LABEL_OK = re.compile(r"^(api_key_alias|team_alias|user|end_user|metadata_[A-Za-z0-9_]{1,40})$")


def valid_selector(v, what):
    """Prometheus label selector (without braces), e.g. job="vllm",instance=~"10.0.0.*"; returned stripped.

    Only label matchers against double-quoted strings without a backslash are accepted: a selector is pasted into
    PromQL queries by observation/query.py, so nothing else (functions, operators, braces) may get through.

    Raises:
        Refusal: if it does not match; `what` names the field in the message."""
    v = str(v or "").strip()
    if not SELECTOR_OK.match(v):
        raise Refusal(f'{what}: label selector expected, e.g. job="vllm"')
    return v


def valid_duration(v, field_name, max_days=None):
    """Duration such as 30, 90s, 5m, 2h or 1d (stripped); `max_days` caps a duration given in days.

    Raises:
        Refusal: if the form is wrong or the number of days exceeds `max_days`."""
    v = str(v or "").strip()
    if not DURATION_OK.match(v) or (max_days and v.endswith("d") and int(v[:-1]) > max_days):
        raise Refusal(f"{field_name}: duration expected, e.g. 5m, 2h, 1d" + (f" ({max_days} days at most)" if max_days else ""))
    return v


def valid_date(v, field_name):
    """Local date and time YYYY-MM-DD HH:MM (a T separator, as sent by a datetime-local field, is accepted).

    Raises:
        Refusal: if the form is wrong."""
    v = str(v or "").strip()
    if not DATE_OK.match(v):
        raise Refusal(f"{field_name}: date expected, form YYYY-MM-DD HH:MM")
    return v.replace("T", " ")


def valid_prometheus_url(v):
    """Prometheus URL of the form http(s)://host[:port][/path] on this machine or a private network, without a
    trailing slash. This server sends the Prometheus key there, so the host gets the same check as the target.

    Raises:
        Refusal: if the form is wrong or the host is public (see _check_private_host)."""
    v = str(v or "").strip()
    if not re.match(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?(/[A-Za-z0-9._/-]*)?$", v):
        raise Refusal("Prometheus URL: expected form http://host:9090")
    _check_private_host(v, "Prometheus URL")
    return v.rstrip("/")


def observation_command(p, excluded=()):
    """Command of the observation (observation/query.py, which queries Prometheus) from the fields `p`.

    `excluded` lists (start, end) Unix times of ranges to discard (stress tests started from this interface).
    Returns (command, environment, working directory); the Prometheus key goes in the environment only. The
    command has no --output: the caller adds it.

    Raises:
        Refusal: if a field is invalid."""
    cmd = [sys.executable, "-m", "observation.query", "--prometheus", valid_prometheus_url(p.get("prometheus")),
           "--target", str(valid_number(p.get("target"), "target", 0.1, 10_000)),
           "--gpu", str(valid_number(p.get("gpu"), "GPU", 1, 4096, True))]
    cmd += ["--vllm", valid_selector(p.get("vllm") or 'job="vllm"', "vLLM selector")]
    view = p.get("view") if p.get("view") in ("windows", "period", "agents") else "windows"
    if p.get("litellm"):
        by = str(p.get("by") or "metadata_agent").strip()
        if not LABEL_OK.match(by):
            raise Refusal("agents distinguished by: api_key_alias, team_alias, user, end_user or metadata_<field>")
        cmd += ["--litellm", valid_selector(p.get("litellm"), "LiteLLM selector"), "--by", by]
    elif view == "agents":
        raise Refusal("per-agent view: give the LiteLLM selector")
    for field_name, flag in (("window", "--window"), ("scrape", "--scrape"), ("since", "--since")):
        v = valid_duration(p.get(field_name), field_name, 400 if field_name == "since" else None) if p.get(field_name) else ""
        if v:
            cmd += [flag, v]
    for field_name, flag in (("start", "--start"), ("end", "--end")):
        if str(p.get(field_name) or "").strip():
            cmd += [flag, valid_date(p.get(field_name), field_name)]
    if p.get("ttft_max") not in (None, ""):
        cmd += ["--ttft-max", str(valid_number(p.get("ttft_max"), "max TTFT", 1, 10_000_000))]
    cmd += ["--view", view, "--period", "hour" if p.get("period") == "hour" else "range"]
    for start, end in excluded:
        cmd += ["--exclude", f"{start:.0f}/{end:.0f}"]
    key = valid_key(p.get("prometheus_key"))
    return cmd, ({"BENCH_PROMETHEUS_KEY": key} if key else {}), ROOT


async def observation(req):
    """POST /api/observation: query Prometheus with `python -m observation.query`; nothing to start, no load.

    The ranges of the runs started from this interface (plus a minute) are passed as --exclude: their load must
    not count as real usage. Returns the table rows (CSV of the windows), the key messages, the full output,
    the exit code and the command; 400 if the parameters are refused, 504 if Prometheus is too slow."""
    try:
        p = await read_body(req)
        excluded = [(e["start"], (e["end"] or time.time()) + 60) for e in RUNS.values()]
        cmd, env, cwd = observation_command(p, excluded)
    except Refusal as e:
        return web.json_response({"error": str(e)}, status=400)
    LOGS_DIR.mkdir(exist_ok=True)
    csv_file = LOGS_DIR / f"observation-{secrets.token_hex(4)}.csv"
    proc = await asyncio.create_subprocess_exec(*cmd, "--output", str(csv_file), cwd=str(cwd),
                                                env={**base_environment(), **env},
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output = (await asyncio.wait_for(proc.communicate(), timeout=120))[0].decode("utf-8", "replace")
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return web.json_response({"error": "Prometheus did not answer within 120 s: shorten the range"}, status=504)
    rows = []
    if csv_file.exists():
        with open(csv_file, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        csv_file.unlink()
    # Keep in sync with the wording printed by observation/query.py
    messages = [l for l in output.splitlines() if "WARNING" in l or "Highest" in l or l.startswith("No ")
                or "unreachable" in l or "HTTP" in l]
    return web.json_response({"rows": rows, "messages": messages, "output": output, "code": proc.returncode,
                              "command": " ".join(cmd)})


async def defaults(req):
    """GET /api/defaults: the starting values of the form, from the BENCH_* variables of the terminal that started the
    interface (the same variables as target.env).

    Only what is not secret is returned: the target, the model, the metrics URLs and the tokenizer. The keys never go to
    the browser: `api_key_set` only says that BENCH_API_KEY exists, and `api_key_url` which target it applies to."""
    key_set = bool(settings.api_key())
    return web.json_response({
        "url": settings.read_setting("URL", ""), "model": settings.read_setting("MODEL", ""),
        "metrics": settings.read_setting("METRICS", ""), "tokenizer": settings.read_setting("TOKENIZER", ""),
        "api_key_set": key_set, "api_key_url": settings.read_setting("URL", "") if key_set else ""})


async def test_connection(req):
    """POST /api/test-connection: check the target (/v1/models) and the vLLM metrics (/metrics), and read the
    server configuration (tools.server_config.collect, plus the deduced GPU count under config.deduced_gpus).

    Each metrics entry has ok, status and vllm_lines; each config server also carries vllm_lines and
    metrics_status (an HTTP status, or "budget" when the time budget ran out)."""
    import aiohttp
    try:
        p = await read_body(req)
        url = valid_local_url(p.get("url")); k = target_key(p)
        met = valid_metrics_urls(p.get("metrics")) or [url + "/metrics"]
        mod = valid_model(p.get("model")) if p.get("model") else None
    except Refusal as e:
        return web.json_response({"error": str(e)}, status=400)
    # During a run, no one-token completion: it would fall inside the measured window.
    running = any(e["code"] is None for e in RUNS.values())
    if running:
        mod = None
    headers = {"Authorization": f"Bearer {k}"} if k else {}
    res = {"run_in_progress": running, "target": None, "metrics": [], "gateway_without_metrics": not valid_metrics_urls(p.get("metrics"))}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
        try:
            async with s.get(url + "/v1/models", headers=headers) as r:
                j = await r.json(content_type=None) if r.status == 200 else None
                res["target"] = {"ok": r.status == 200, "status": r.status,
                                "models": [m.get("id") for m in (j or {}).get("data", [])] if j else []}
        except Exception as e:  # noqa: BLE001
            res["target"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        for u in met:
            try:
                async with s.get(u, allow_redirects=False) as r:  # /metrics does not require a key: we do not send it
                    t = await r.text() if r.status == 200 else ""
                    n = sum(1 for l in t.splitlines() if l.startswith("vllm:"))
                    res["metrics"].append({"url": u, "ok": r.status == 200 and n > 0, "status": r.status, "vllm_lines": n})
            except Exception as e:  # noqa: BLE001
                res["metrics"].append({"url": u, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
    try:  # version, cache, parallelism (fingerprint); /server_info only if VLLM_SERVER_DEV_MODE=1
        from tools.server_config import collect, deduced_gpus
        res["config"] = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(
            None, collect, url, valid_metrics_urls(p.get("metrics")), k, mod), timeout=60)
        count, explanation, masking = deduced_gpus(res["config"])  # what `python -m report` will use without --gpu
        res["config"]["deduced_gpus"] = {"count": count, "explanation": explanation, "masking_possible": masking}
    except Exception as e:  # noqa: BLE001
        res["config"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return web.json_response(res)


async def home(req):
    """GET /: the built interface (dist/index.html); 500 if it has not been built."""
    if not (DIST / "index.html").exists():
        return web.Response(text="Interface not built: dist/ is missing (see interface/README.md).", status=500)
    return web.FileResponse(DIST / "index.html")


@web.middleware
async def guard(req, handler):
    """Token, Host and origin checked for the whole API (protects against other web pages and DNS rebinding)."""
    if req.path.startswith("/api/"):
        hosts = {f"127.0.0.1:{PORT[0]}", f"localhost:{PORT[0]}"}
        if req.host not in hosts:
            return web.json_response({"error": "host refused"}, status=403)
        origin = req.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in hosts}:
            return web.json_response({"error": "origin refused"}, status=403)
        if not secrets.compare_digest(req.headers.get("X-Bench-Token", ""), TOKEN):
            return web.json_response({"error": "token missing or invalid: open the address shown at startup"}, status=401)
        if req.method == "POST" and req.content_type != "application/json":
            return web.json_response({"error": "JSON expected"}, status=415)
    return await handler(req)


async def clean_shutdown(app):
    """When the interface stops, also stop the runs in progress (otherwise they would keep going alone)."""
    for e in RUNS.values():
        if e["code"] is None:
            try:
                os.killpg(e["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    """Parse --port and serve the interface on 127.0.0.1; the address to open (with the token) is printed."""
    a = argparse.ArgumentParser(); a.add_argument("--port", type=int, default=8765)
    p = a.parse_args()
    PORT[0] = p.port
    app = web.Application(middlewares=[guard])
    app.on_shutdown.append(clean_shutdown)
    app.add_routes([web.post("/api/runs", launch), web.get("/api/runs", list_runs),
                    web.get("/api/runs/{eid}/log", run_log), web.post("/api/runs/{eid}/stop", stop_run),
                    web.get("/api/roots", list_roots), web.get("/api/defaults", defaults), web.get("/api/results", read_results),
                    web.post("/api/test-connection", test_connection), web.post("/api/observation", observation),
                    web.get("/", home)])
    if (DIST / "assets").exists():
        app.router.add_static("/assets", DIST / "assets")
    print(f"Bench interface: http://127.0.0.1:{p.port}/?token={TOKEN}\n(open this exact address; Ctrl-C stops the interface and the runs in progress)")
    web.run_app(app, host="127.0.0.1", port=p.port, print=None)


if __name__ == "__main__":
    main()
