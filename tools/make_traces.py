"""Generate synthetic agentic traces in the Weka format, for `agentx.sh` (BENCH_AGENTX_DIR).

The sets published by SemiAnalysis hold sessions of up to about one million tokens, which a small server cannot
serve. This script writes sessions of a size you choose, shaped like the real ones, so that the same AgentX-style
replay can run on a small server and on a large one.

What a session looks like (the shape of the published set `cc-traces-weka-062126`):
- a very small first call, then a jump to the working context, then calls that each extend the previous context
  with new cache blocks (about 90% of consecutive calls repeat the whole list of hash IDs of the previous one);
- `in` is 64 x the number of hash IDs, as in the published set (AIPerf also accepts other values);
- most calls are streaming, with a client think time between calls (a median of a few seconds, long tail);
- some sessions spawn sub-agents, which start from a shared prefix and run their own short calls.

The traces hold no text: AIPerf builds the prompts from the sizes and the hash IDs.
The result is NOT the published AgentX set: AIPerf marks such a run `submission_valid=false`. Use it to compare
machines and configurations with each other, not with the published results.

Usage:
    python tools/make_traces.py --preset small --output traces-small
    BENCH_AGENTX_DIR=traces-small ./agentx.sh small-r1 "1 2 4"

Note: from 8 traces on, AIPerf rebuilds the prompts with several processes, and this step can hang. If the run never
starts, kill the AIPerf processes and set AIPERF_DATASET_WEKA_PARALLEL_WORKERS=1 (serial loading).

The format, and how to build traces from your own logs, are described in tools/traces.md.
"""
import argparse
import json
import math
import pathlib
import random
import sys
from dataclasses import dataclass

BLOCK = 64                 # tokens per cache block, as in the published traces
MODEL = "synthetic"        # model name written in the traces (AIPerf sends its own --model)


@dataclass(frozen=True)
class Preset:
    """Size of a set of traces. `max_context` bounds input + output of every call."""
    name: str
    max_context: int       # tokens: no call goes above this
    calls_median: int      # median number of main-agent calls per session
    calls_max: int


PRESETS = {p.name: p for p in (
    Preset("tiny", 8_000, 30, 80),
    Preset("small", 28_000, 40, 120),
    Preset("medium", 120_000, 50, 200),
    Preset("large", 250_000, 60, 300),
)}


def lognormal(rng, median, sigma):
    """A value from a log-normal distribution with the given median."""
    return rng.lognormvariate(math.log(median), sigma)


class Session:
    """Builds one trace. Hash IDs are allocated from a counter local to the trace."""

    def __init__(self, rng, preset, max_context, subagents):
        self.rng = rng
        self.preset = preset
        self.max_context = max_context
        self.subagents = subagents
        self.counter = 0
        self.out_cap = min(8_192, max_context // 10)

    def fresh(self, n):
        """n new cache block IDs (none if n is not positive)."""
        n = max(0, n)
        ids = list(range(self.counter, self.counter + n))
        self.counter += n
        return ids

    def output_tokens(self, tokens_in, scale):
        """Output tokens of a call: log-normal, never beyond the room left in the context."""
        median = min(700, max(80, scale * 0.006))
        out = int(lognormal(self.rng, median, 0.9))
        return max(16, min(out, self.out_cap, self.max_context - tokens_in))

    def think(self):
        """Client delay between two calls: a median of 4.8 s, long tail, capped at one minute."""
        return min(60.0, lognormal(self.rng, 4.8, 1.1))

    def call(self, kind, t, hash_ids, think, scale):
        """One call of type `kind` ("n" or "s") at time t (seconds), with the cache block IDs `hash_ids`, the
        client delay `think` before it (None for the first call) and `scale`, the size that sets the output length.
        Returns the entry and the time at which the call ends."""
        tokens_in = BLOCK * len(hash_ids)
        out = self.output_tokens(tokens_in, scale)
        ttft = 1.5 + tokens_in / 30_000 + self.rng.uniform(0, 1.0)
        api_time = ttft + out / 45
        entry = {"t": round(t, 3), "type": kind, "model": MODEL, "in": tokens_in, "out": out,
                 "hash_ids": list(hash_ids), "api_time": round(api_time, 3)}
        if think is not None:
            entry["think_time"] = round(think, 3)
        if kind == "s":
            entry["ttft"] = round(ttft, 3)
        return entry, t + api_time

    def subagent(self, start, prefix, number):
        """One sub-agent: a short run of calls that starts from the shared prefix."""
        context = list(prefix) + self.fresh(self.rng.randint(4, 8))
        requests, t, gap = [], start, 0.0
        for _ in range(self.rng.randint(2, 10)):
            entry, t_end = self.call("n", t, context, gap, 4_000)
            requests.append(entry)
            if len(context) * BLOCK + 2_000 > self.max_context:
                break
            context = context + self.fresh(self.rng.randint(2, 10))
            gap = self.rng.uniform(0, 3)
            t = t_end + gap
        finish = requests[-1]["t"] + requests[-1]["api_time"]
        entry = {"t": round(start, 3), "type": "subagent",
                 "agent_id": f"subagent_{number:03d}_{self.rng.getrandbits(32):08x}", "subagent_type": "Subagent",
                 "duration_ms": int((finish - start) * 1000),
                 "total_tokens": sum(r["in"] + r["out"] for r in requests), "tool_use_count": None,
                 "status": "completed", "requests": requests, "models": [MODEL]}
        return entry, finish

    def subagent_group(self, spawn, prefix, number):
        """One to three sub-agents spawned together. Returns the entries, the end of the last one, the new count."""
        entries, end, offset = [], spawn, 0.0
        for k in range(self.rng.randint(1, 3)):
            number += 1
            if k:
                offset += self.rng.uniform(1, 8)
            entry, finish = self.subagent(spawn + offset, prefix, number)
            entries.append(entry)
            end = max(end, finish)
        return entries, end, number

    def build(self, index):
        """The trace of one session."""
        rng, preset = self.rng, self.preset
        peak = int(self.max_context * math.exp(rng.uniform(math.log(0.25), 0)))   # sessions differ in size
        if index == 1:                                  # the first session reaches the limit, with no compaction
            peak = self.max_context
        peak_blocks = max(30, (peak - self.out_cap) // BLOCK)
        ceiling = (self.max_context - self.out_cap) // BLOCK                      # most blocks a call may carry
        calls = max(12, min(preset.calls_max, round(lognormal(rng, preset.calls_median, 0.7))))
        start_blocks = max(20, round(rng.uniform(0.15, 0.35) * peak_blocks))
        weights = [rng.lognormvariate(0, 0.9) for _ in range(calls - 2)]
        room = max(0, peak_blocks - start_blocks)
        growth = [max(1, round(room * w / sum(weights))) for w in weights]
        shared = self.fresh(max(8, int(0.04 * self.max_context / BLOCK)))         # the sub-agents' common prefix

        first, t_end = self.call("s", 0.0, self.fresh(rng.randint(6, 12)), None, peak)   # the tiny first call
        requests = [first]
        context = self.fresh(start_blocks)
        gap = rng.uniform(0, 0.3)
        t, think, number = t_end + gap, gap, 0
        for k in range(calls - 1):
            entry, t_end = self.call("s", t, context, think, peak)
            requests.append(entry)
            if len(context) >= ceiling:
                break
            t = t_end
            if self.subagents and rng.random() < 0.05:
                entries, t, number = self.subagent_group(t_end + rng.uniform(0.1, 1.0), shared, number)
                requests.extend(entries)
            if k >= 10 and rng.random() < 0.03 and index != 1:                 # compaction: a fresh, shorter context
                context = self.fresh(max(start_blocks, int(0.3 * len(context))))
            else:
                context = context + self.fresh(min(growth[k] if k < len(growth) else 1, ceiling - len(context)))
            think = self.think()
            t += think
        return {"id": f"synthetic-{index:04d}-{rng.getrandbits(48):012x}", "models": [MODEL], "block_size": BLOCK,
                "hash_id_scope": "local", "requests": requests}


def make_set(preset, sessions, seed, max_context=None, subagents=True):
    """The list of `sessions` traces of a `Preset`, with `max_context` replacing its longest call if given and
    sub-agents unless `subagents` is False. The same arguments always give the same set."""
    rng = random.Random(seed)
    limit = max_context or preset.max_context
    return [Session(rng, preset, limit, subagents).build(i + 1) for i in range(sessions)]


def calls_of(trace):
    """Every call of a trace: the main-agent calls, then the calls inside each sub-agent."""
    main = [r for r in trace["requests"] if r["type"] != "subagent"]
    sub = [c for r in trace["requests"] if r["type"] == "subagent" for c in r["requests"]]
    return main, sub


def summarize(traces):
    """Counts of a set: sessions, main-agent and sub-agent calls, longest call and total tokens."""
    main_calls = sub_calls = longest = total = 0
    for trace in traces:
        main, sub = calls_of(trace)
        main_calls += len(main)
        sub_calls += len(sub)
        for call in main + sub:
            longest = max(longest, call["in"] + call["out"])
            total += call["in"] + call["out"]
    return {"sessions": len(traces), "main_calls": main_calls, "subagent_calls": sub_calls,
            "longest_call": longest, "total_tokens": total}


def validate(traces):
    """Check the traces against AIPerf's own model.

    Raises:
        SystemExit: if AIPerf cannot be imported in this environment.
        pydantic.ValidationError: if a trace does not follow the format."""
    try:
        from aiperf.dataset.loader.weka_trace_models import WekaTrace
    except ImportError as error:
        raise SystemExit(f"--validate needs AIPerf in this environment ({error}). Nothing was written.") from None
    for trace in traces:
        WekaTrace.model_validate(trace)


def main(argv=None):
    """Command-line entry point: write a set of traces, print a summary. Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small",
                        help="size of the set: " + ", ".join(f"{p.name} (up to {p.max_context:,} tokens)"
                                                              for p in PRESETS.values()))
    parser.add_argument("--output", required=True,
                        help="folder to write, created if missing (one trace-NNNN.json file per session)")
    parser.add_argument("--sessions", type=int, default=30, help="number of sessions, at least 1 (default 30)")
    parser.add_argument("--seed", type=int, default=1, help="the same seed gives the same set (default 1)")
    parser.add_argument("--max-context", type=int, default=None,
                        help="tokens (4,096 to 1,000,000): replaces the preset's limit for one call (input + output)")
    parser.add_argument("--no-subagents", action="store_true", help="sessions without sub-agents")
    parser.add_argument("--validate", action="store_true",
                        help="check the traces with AIPerf's model before writing; stops without writing if AIPerf is missing")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the trace-*.json files of the folder; without it, a folder with any .json file is refused")
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    if args.sessions < 1:
        raise SystemExit("--sessions must be at least 1.")
    if args.max_context is not None and not 4_096 <= args.max_context <= 1_000_000:
        raise SystemExit("--max-context must be between 4,096 and 1,000,000 tokens.")
    out = pathlib.Path(args.output)
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} is not a folder.")
    if out.exists() and any(out.glob("*.json")) and not args.overwrite:
        raise SystemExit(f"{out} already holds traces: use --overwrite, or another folder.")
    traces = make_set(preset, args.sessions, args.seed, args.max_context, not args.no_subagents)
    if args.validate:
        validate(traces)
        print("Traces validated with AIPerf's model.")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("trace-*.json"):                      # agentx.sh replays every .json of the folder
        if old.is_file():
            old.unlink()
    others = [f.name for f in out.glob("*.json")]
    if others:
        print(f"Warning: {len(others)} other .json file(s) in {out} will be replayed too.", file=sys.stderr)
    for i, trace in enumerate(traces, 1):
        (out / f"trace-{i:04d}.json").write_text(json.dumps(trace, separators=(",", ":")))
    s = summarize(traces)
    limit = args.max_context or preset.max_context
    print(f"{s['sessions']} sessions in {out}: {s['main_calls']:,} main calls, {s['subagent_calls']:,} sub-agent calls, "
          f"longest call {s['longest_call']:,} tokens, {s['total_tokens']:,} tokens in all.")
    print(f"vLLM needs --max-model-len of at least {limit + 2_048:,} (margin for the chat template).")
    print(f"Run: BENCH_AGENTX_DIR={out} ./agentx.sh <label> \"1 2 4\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
