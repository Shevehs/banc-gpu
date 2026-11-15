# Bench interface

A small local app to start the bench runs, follow their logs and display the results.
It calls the same scripts as the command line (`sweep.sh`, `agentx.sh`, `python -m agents_test.launcher`,
`python -m observation.observer` and the report package): a run started here is identical to the same run
started by hand, and the exact command is written at the top of its log.

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

The vLLM server is always started by hand, with the options from section 4.2 of the repository README.

## Security

- Listens on 127.0.0.1 only; a random token is required for the whole API (printed at startup, or set with
  the `BENCH_TOKEN` environment variable, sent as the `X-Bench-Token` header); the Host header and the request
  origin are checked.
- Only runs the bench scripts, with validated parameters; extra AIPerf options come from an allowlist.
- Target and metrics URLs must point to this machine or a private network (loopback, RFC 1918, unique local, link-local;
  a host name must resolve only to such addresses). The target receives the API key and the metrics URLs are fetched by
  this machine, so public hosts are refused unless the interface was started with `BENCH_ALLOW_PUBLIC_URLS=1`.
- The traces folder (AgentX) must lie, once `..` and symbolic links are resolved, under the repository, your home
  directory or a directory listed in `BENCH_ALLOWED_DIRS` (separated like `PATH`).
  One run at a time; stopping the interface stops the run in progress.
- The API key and the per-agent keys are passed through environment variables and are written to no log (the command at the top of a log shows them as `***`). The AIPerf-based modes give the key to AIPerf on its command line, so it is visible in `ps` on this machine while AIPerf runs.
- Runs do not inherit the terminal's settings (`BENCH_URL`, `BENCH_API_KEY`, `BENCH_EXTRA`…): only the interface fields count,
  and the interface passes them to the scripts under the same `BENCH_*` names;
  from the terminal, only `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `TMPDIR`, the locale (`LANG`, `LC_*`),
  the time zone (`TZ`), `VIRTUAL_ENV`, `HF_HOME`, `HF_HUB_CACHE`, the proxy settings, the certificates and
  `DO_NOT_TRACK` are passed on.
- The API key goes to the target. In observation mode with the per-agent view, LiteLLM's `/metrics/` (protected
  by default) receives the "LiteLLM key" field if it is filled in, otherwise the API key. No key is sent to vLLM's `/metrics`.

## What it does

- **Launch**: connection fields (the `BENCH_*` variables of `target.env`, repository README section 5.2), a "Test connection" button
  (which also shows the vLLM server configuration: version, prefix cache, number of speculative tokens,
  TP/PP/DP/EP parallelism read from the response fingerprint and, only if vLLM is started with
  `VLLM_SERVER_DEV_MODE=1`, the speculative decoding method and the scheduler), then one of the four modes:
  synthetic stress test (AIPerf), AgentX stress test, test-agents stress test, or observation of real usage
  (repository README, sections 7 and 8). For the test agents, "Label the agents" and "Per-agent keys" give every
  request the identity of its profile, for the per-agent view through LiteLLM (repository README, section 8.4).
- **Runs**: the runs started since the interface was opened, a live log, and stop (a stopped observation saves
  its current window, shortened). Logs are kept in `interface/logs/`.
- **Results**: analysis of a results directory (table, selected point, warnings, throughput-per-GPU vs.
  interactivity curve with the target); for observation, views by window, by period (6-hour ranges or hours)
  and by agent (LiteLLM metrics).

## Modifying the interface

The compiled files are in `interface/dist/` and are committed, so Node is not needed to use the interface.
To modify it (Node 18 or later):

```bash
cd interface/web && npm ci && npm run build
```
