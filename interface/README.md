# Bench interface

A small local app to start the bench stress tests, follow their logs, display the results and query
Prometheus for the observation of real usage. It calls the same scripts as the command line (`sweep.sh`,
`agentx.sh`, `python -m agents_test.launcher`, `python -m observation.query` and the report package): a run
started here is identical to the same run started by hand, and the exact command is written at the top of its log.

## Start

From the repository root, with the `.venv-aiperf` environment activated (it already contains `aiohttp`):

```bash
python -m interface.server
```

then open **the exact address printed at startup** (it contains an access token: `http://127.0.0.1:8765/?token=…`).

On a remote machine (client VM), the server only listens on 127.0.0.1; from your workstation:

```bash
ssh -L 8765:127.0.0.1:8765 <node>
```

then open, on your workstation, the address printed by the server (with its token).

The interface never starts vLLM: it measures the one that is running. Options to check for each mode: section 4.2 of the repository README.

## Security

- Listens on 127.0.0.1 only; a random token is required for the whole API (printed at startup, or set with
  the `BENCH_TOKEN` environment variable, sent as the `X-Bench-Token` header); the Host header and the request
  origin are checked.
- Only runs the bench scripts, with validated parameters; extra AIPerf options come from an allowlist.
- Target, metrics and Prometheus URLs must point to this machine or a private network (loopback, RFC 1918, unique local, link-local;
  a host name must resolve only to such addresses). The target receives the API key, and the metrics and Prometheus URLs are fetched by
  this machine (the Prometheus key goes to Prometheus), so public hosts are refused unless the interface was started with `BENCH_ALLOW_PUBLIC_URLS=1`.
- The traces folder (AgentX) must lie, once `..` and symbolic links are resolved, under the repository, your home
  directory or a directory listed in `BENCH_ALLOWED_DIRS` (separated like `PATH`).
  One run at a time; stopping the interface stops the run in progress.
- Prometheus selectors (vLLM, LiteLLM) are validated against a strict label-matcher pattern before they reach a PromQL
  query: no function, operator, brace, backslash or unquoted value.
- The API key, the per-agent keys and the Prometheus key are passed through environment variables and are written to no log (the command at the top of a log shows them as `***`). The AIPerf-based modes give the key to AIPerf on its command line, so it is visible in `ps` on this machine while AIPerf runs.
- Runs do not inherit the terminal's settings (`BENCH_URL`, `BENCH_API_KEY`, `BENCH_EXTRA`…): only the interface fields count,
  and the interface passes them to the scripts under the same `BENCH_*` names. Two exceptions, both explained below: the
  starting values of the form, and the key;
  from the terminal, only `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `TMPDIR`, the locale (`LANG`, `LC_*`),
  the time zone (`TZ`), `VIRTUAL_ENV`, `HF_HOME`, `HF_HUB_CACHE`, the proxy settings, the certificates and
  `DO_NOT_TRACK` are passed on.
- The form starts with `BENCH_URL`, `BENCH_MODEL`, `BENCH_METRICS` and `BENCH_TOKENIZER` of the terminal that started the
  interface, or else of `target.env` at the root of the repository (`GET /api/defaults`, behind the same token as the rest of the
  API). `target.env` is read once, when the interface starts, and never run: only plain `export BENCH_NAME=value`
  lines are taken, and only the target settings (`BENCH_URL`, `BENCH_MODEL`, `BENCH_METRICS`, `BENCH_TOKENIZER`,
  `BENCH_API_KEY`); the terminal wins. No key is ever sent to the page: the
  response only says whether `BENCH_API_KEY` exists and for which target. When the API key field is empty, the server
  uses `BENCH_API_KEY` for the run, but only if the form's target URL is the `BENCH_URL` of the terminal or of `target.env` (same scheme,
  host, port and path, case and a trailing slash ignored). A key meant for one server never goes to another.
- The API key goes to the target; no key is sent to vLLM: neither to its `/metrics` nor, for the fingerprint through a gateway, to its `/v1`. In observation, only the "Prometheus key"
  (if filled in, `BENCH_PROMETHEUS_KEY`) is sent, to Prometheus; LiteLLM's key belongs in Prometheus's configuration,
  not in the interface.

## What it does

- **Configuration**: the target (the `BENCH_*` variables of `target.env`, repository README section 5.2), with a "Test
  connection" button (which also shows the vLLM server configuration: version, prefix cache, number of speculative tokens,
  TP/PP/DP/EP parallelism read from the fingerprint (the target's, or vLLM's when the target is a gateway that gives none), the number of GPUs deduced from it, a warning when a
  metrics URL is not vLLM's `/metrics` and, only if vLLM is started with `VLLM_SERVER_DEV_MODE=1`, the speculative
  decoding method and the scheduler); the number of GPUs behind the target (filled in by the connection test when
  empty); and Prometheus (URL, key, vLLM and LiteLLM selectors, the label that tells agents apart; the scrape interval is
  read from Prometheus).
- **Launch stress tests**: one of the three stress tests: synthetic (AIPerf), AgentX or test agents
  (repository README, section 6). For the test agents, "Label the agents" and "Per-agent keys" give every
  request the identity of its profile, for the per-agent view through LiteLLM (repository README, section 7.5).
- **Runs**: the runs started since the interface server was started, a live log, and stop. Logs are kept in `interface/logs/`.
- **Results**: analysis of a results directory, shown as soon as a field changes (table, selected point, warnings,
  throughput-per-GPU vs. interactivity curve with the target). The "GPUs" field is optional: empty, the count of each run is deduced
  from its fingerprint (`server_config.json`).
- **Observation**: nothing to start. Prometheus and the GPUs come from Configuration; range ("Since", or start
  and end), window length: a chart, drawn as soon as a field changes, with the choice of the variables on x (the date by default) and y,
  by window, by period (6-hour ranges or hours) and by agent (repository README, section 7). The warnings are folded under the verdict, grouped by kind. A "refresh every
  minute" box follows real usage live. The ranges of the runs
  started from the interface (since its server was started) are marked "run in progress" and discarded from the
  verdicts. It works while a run is in progress.

## Modifying the interface

The compiled files are in `interface/dist/` and are committed, so Node is not needed to use the interface.
To modify it (Node 18 or later):

```bash
cd interface/web && npm ci && npm run build
```
