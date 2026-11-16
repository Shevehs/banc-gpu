"""Run the test agents at levels of concurrent tasks and log every call.

Example (small model, 32,768-token context):
  python -m agents_test.launcher --url http://127.0.0.1:8000 --model bench --agents code_review,pentest,report \
      --levels "1 2 4" --duration 600 --warmup 120 --scale 0.2 --label trial-r1

--url points at vLLM directly or at a gateway (LiteLLM). The server metrics (preemptions,
cache, speculative decoding, KV cache) are read from --metrics: by default <url>/metrics, which
is right when --url is vLLM; with a gateway, give the vLLM /metrics URLs (one per replica).
--url, --model and --metrics default to the variables BENCH_URL, BENCH_MODEL and BENCH_METRICS (as for sweep.sh
and agentx.sh); an option given on the command line wins.
API key: environment variable BENCH_API_KEY (never as an argument, so it does not appear in
logs or in the process list).

For each level N, N "workers" chain tasks (agents taken in turn)
for --duration seconds; no new task starts after that, and the tasks in progress
finish. Results: <output>/<label>/n<N>/{calls,tasks}.jsonl and level.json.
"""
import argparse
import itertools
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import zlib

from agents_test import agents as module_agents
from agents_test.agents import AGENTS
from agents_test.common import CallLog, Model
from agents_test.metrics import Sampler
from tools import settings
from tools.server_config import collect as collect_server_config, summary


def build_parser():
    """The command-line parser.

    --url, --model and --metrics default to the variables BENCH_URL, BENCH_MODEL and BENCH_METRICS, like the shell
    scripts (an empty variable counts as not set), then to the values below. An option given on the command line wins."""
    a = argparse.ArgumentParser()
    a.add_argument("--url", default=settings.read_setting("URL", "http://127.0.0.1:8000"),
                   help="target (default: BENCH_URL, else http://127.0.0.1:8000)")
    a.add_argument("--model", default=settings.read_setting("MODEL", "bench"),
                   help="served model name, --served-model-name (default: BENCH_MODEL, else bench)")
    a.add_argument("--agents", default="code_review,pentest,report")
    a.add_argument("--levels", default="1 2 4 8", help="concurrent tasks, separated by spaces")
    a.add_argument("--duration", type=float, default=900, help="seconds per level")
    a.add_argument("--warmup", type=float, default=300,
                   help="seconds excluded from the analysis at the start of each level (at least the duration of one task)")
    a.add_argument("--spread", type=float, default=30, help="worker start times spread over this many seconds")
    a.add_argument("--scale", type=float, default=1.0, help="multiplies all sizes (0.2 for a model with a 32,768-token context)")
    a.add_argument("--mode", choices=["scripted", "autonomous"], default="scripted")
    a.add_argument("--latency-factor", type=float, default=1.0, help="multiplies the simulated tool latencies")
    a.add_argument("--metrics", default=settings.read_setting("METRICS"),
                   help="vLLM /metrics URLs, separated by spaces (default: BENCH_METRICS, else <url>/metrics)")
    a.add_argument("--no-metrics", action="store_true", help="do not read /metrics")
    a.add_argument("--output", default="results-agents")
    a.add_argument("--label", required=True)
    a.add_argument("--overwrite", action="store_true", help="replace existing results with the same label")
    a.add_argument("--label-agents", action="store_true",
                   help="add metadata.agent=<profile> to each request (per-agent view via LiteLLM)")
    return a


def main():
    """Command-line entry point: run each level of concurrent tasks, then write the logs and level.json.

    Results go to <output>/<label>/n<N>/.

    Raises:
        SystemExit: on invalid options or an existing label without --overwrite (exit code 130 on Ctrl-C)."""
    settings.warn_ignored_settings()
    p = build_parser().parse_args()
    module_agents.LATENCY_FACTOR = p.latency_factor
    key = settings.api_key()
    # One key per profile (LiteLLM virtual keys, per-agent view by api_key_alias):
    # BENCH_AGENT_KEYS="code_review=sk-…,pentest=sk-…,report=sk-…"; a profile that is absent keeps the shared key.
    agent_keys = {}
    for pair in (os.environ.get("BENCH_AGENT_KEYS") or "").split(","):
        if "=" in pair:
            k_, v_ = pair.split("=", 1)
            agent_keys[k_.strip()] = v_.strip()
    if p.warmup >= p.duration:
        raise SystemExit(f"--warmup ({p.warmup:.0f} s) must be less than --duration ({p.duration:.0f} s): "
                         "otherwise the measurement window is empty.")
    urls_metrics = [] if p.no_metrics else (p.metrics.split() if p.metrics else [p.url.rstrip("/") + "/metrics"])
    agents = [x.strip() for x in p.agents.split(",") if x.strip()]
    unknown = [x for x in agents if x not in AGENTS]
    if unknown:
        raise SystemExit(f"Unknown agents: {unknown} (choices: {list(AGENTS)})")

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$", p.label) or ".." in p.label:
        raise SystemExit("--label: letters, digits, dot, hyphen, underscore, starting with a letter "
                         "or a digit; no / and no ..")
    already = os.path.join(p.output, p.label)
    if os.path.isdir(already) and os.listdir(already):
        if not p.overwrite:
            raise SystemExit(f"{already} already exists: change the label, or add --overwrite to replace it.")
        import shutil; shutil.rmtree(already)
    os.makedirs(already, exist_ok=True)
    try:  # snapshot of the server configuration (version, cache, parallelism from the fingerprint)
        cfg = collect_server_config(p.url, urls_metrics, key, p.model)
        with open(os.path.join(already, "server_config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1, ensure_ascii=False)
        print(summary(cfg), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(server configuration not collected: {e})", file=sys.stderr)
    for n in [int(x) for x in p.levels.split()]:
        level_dir = os.path.join(p.output, p.label, f"n{n}")
        os.makedirs(level_dir, exist_ok=True)
        log = CallLog(level_dir)
        t_start = time.time(); t_end = t_start + p.duration
        counter = itertools.count()
        stop = threading.Event()
        sampler = Sampler(urls_metrics, t_start + p.warmup, t_end).start()  # /metrics: no key
        print(f"== {p.label}: {n} concurrent task(s), {p.duration:.0f} s ==", flush=True)

        def worker(w):  # the worker threads are joined inside this loop iteration, so the loop variables it uses stay valid
            models = {ag: Model(p.url, p.model, log, p.mode, api_key=agent_keys.get(ag) or key or "bench",
                                  label=p.label_agents) for ag in agents}
            k = w  # offset: the workers do not all start with the same agent
            # staggered starts, to avoid synchronized waves
            stop.wait(random.Random(f"{p.label}-{n}-{w}").uniform(0, min(p.spread, p.duration / 4)))
            while time.time() < t_end and not stop.is_set():
                agent = agents[k % len(agents)]; k += 1
                num = next(counter); task = f"{p.label}-n{n}-{num}"
                t0 = time.time(); err = None
                try:
                    AGENTS[agent](models[agent], task, seed=zlib.crc32(f"{p.label}-n{n}-{num}".encode()), scale=p.scale, mode=p.mode)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:300]
                    traceback.print_exc()
                log.log_task(agent=agent, task=task, worker=w, t_start=t0, t_end=time.time(), error=err)
                if err:
                    time.sleep(2)  # avoid a rapid error loop if the server is down

        threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n)]
        for f in threads: f.start()
        try:
            while any(f.is_alive() for f in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop.set(); log.close()
            print("\nInterrupted: tasks in progress are abandoned; this level is incomplete.", flush=True)
            sys.exit(130)
        log.close()
        measurements = sampler.result()
        if urls_metrics and measurements["reads"] < 2:
            print(f"   WARNING: server metrics not read ({measurements['failed_reads']} failure(s)) on {urls_metrics}", flush=True)
        with open(os.path.join(level_dir, "level.json"), "w") as fh:
            json.dump({"concurrent_tasks": n, "t_start": t_start, "t_end": t_end, "t_end_actual": time.time(),
                       "warmup": p.warmup, "scale": p.scale, "latency_factor": p.latency_factor, "mode": p.mode, "agents": agents,
                       "url": p.url, "model": p.model, "metrics": measurements}, fh, indent=1)
        print(f"   finished in {time.time() - t_start:.0f} s -> {level_dir}", flush=True)


if __name__ == "__main__":
    main()
