# Traces for `agentx.sh`: format, generator, your own

`agentx.sh` replays recorded agent sessions. By default it uses the sessions published by SemiAnalysis, but it also
replays a folder of your own traces (`BENCH_AGENTX_DIR`). This page says what a trace is, how to generate a set of the
size you need, and how to build traces from your own agent's logs.

A run on a local folder is marked `submission_valid=false` by AIPerf: it is not comparable to the published AgentX
results, but it is comparable between your own runs.

## What a trace is

One JSON file per session, in the "Weka" format of the kv-cache-tester project (the one the SemiAnalysis traces use).
A trace holds no prompt or response text, only sizes, timings and cache block hashes. AIPerf builds the prompts from
them.

At the top of the file:

| Field | Meaning |
|---|---|
| `id` | name of the session |
| `models` | models used in the session |
| `block_size` | tokens per cache block (any value above 0; 64 in the published traces) |
| `hash_id_scope` | always `"local"`: hash IDs are only compared inside one trace |
| `requests` | the calls, in time order |
| `tool_tokens`, `system_tokens` | optional: size of the tools and system prefixes |
| `totals` | optional: a summary of the trace, kept as it is |

Each call in `requests`:

| Field | Meaning |
|---|---|
| `t` | seconds since the start of the session |
| `type` | `"n"` for a normal call, `"s"` for a streaming call |
| `model` | model name |
| `in`, `out` | input and output tokens |
| `hash_ids` | one cache block ID per `block_size` tokens of the input; optional, but without it the cache reuse is not reproduced |
| `think_time` | optional: the client's delay before the call, in seconds |
| `api_time` | optional: how long the call took, in seconds |
| `ttft` | optional, streaming calls only: time to first token, in seconds |
| `stop` | optional: `""`, `"tool_use"` or `"end_turn"` |
| `input_types`, `output_types` | optional content-type notes |

All the top-level fields are required except `tool_tokens`, `system_tokens` and `totals`. In a call, `t`, `type`, `model`,
`in` and `out` are required and the other fields are optional.

A field that is not in these tables or in the sub-agent description below is refused.

**Hash IDs** are what makes the cache reuse reproducible. Two calls that start with the same text must start with the
same IDs.

A call that continues the previous context repeats the IDs of the previous call, then adds new ones for the new blocks.
If the previous call ended on a block that was not full, that last block gets a new ID. In the published traces about
90% of consecutive calls repeat the previous IDs.

**Sub-agents** are optional. A sub-agent is an entry of type `"subagent"` in `requests`, with:

- `t` (the spawn time), its own list of `requests`, and the required fields `agent_id`, `subagent_type`, `status` and
  `models`;
- optionally `duration_ms`, `total_tokens`, `tool_use_count` (these three may be `null`), `tool_tokens` and
  `system_tokens`.

The next call of the main agent comes after the sub-agent has finished.

Here is a minimal trace with two calls:

```json
{
  "id": "session-1",
  "models": ["my-model"],
  "block_size": 64,
  "hash_id_scope": "local",
  "requests": [
    {"t": 0.0, "type": "n", "model": "my-model", "in": 400, "out": 120, "hash_ids": [0, 1, 2, 3, 4, 5, 6],
     "api_time": 6.0},
    {"t": 8.5, "type": "s", "model": "my-model", "in": 670, "out": 120,
     "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "api_time": 6.0, "think_time": 2.5, "ttft": 1.2}
  ]
}
```

## Generate a set of a chosen size

When you only need agent-like load of a given size, for example to run the same test on a small server and on a large
one, generate the set:

```bash
python tools/make_traces.py --preset small --output traces-small
```

Then run AgentX on that folder:

```bash
BENCH_AGENTX_DIR=traces-small ./agentx.sh small-r1 "1 2 4"
```

| Preset | Longest call | vLLM needs `--max-model-len` of at least |
|---|---|---|
| `tiny` | 8,000 tokens | 10,048 |
| `small` | 28,000 tokens | 30,048 |
| `medium` | 120,000 tokens | 122,048 |
| `large` | 250,000 tokens | 252,048 |

The script prints the value for your set. It adds 2,048 tokens for the chat template.

Options:

- `--output`: the folder to write (required), with one `trace-NNNN.json` file per session;
- `--preset`: `tiny`, `small`, `medium` or `large` (default `small`);
- `--sessions`: the number of sessions (default 30);
- `--seed`: the same seed gives the same set (default 1);
- `--max-context`: replaces the preset's longest call, 4,096 tokens at least;
- `--no-subagents`: sessions without sub-agents;
- `--validate`: checks the traces with AIPerf's model before writing (needs AIPerf in the environment);
- `--overwrite`: writes into a folder that already holds traces.

The shape follows the published set: a very small first call, a jump to the working context, then calls that extend the
previous context, with think times that have a long tail and sub-agents that start from a shared prefix. The sizes of
the sessions differ, and the first session reaches the preset's limit.

## Build traces from your own agent's logs

No tool in this repository reads your logs: you write the conversion. For each call of each session you need:

- the time of the request;
- the size of the input and of the output;
- how long the call took.

To reproduce the cache reuse you also need the prompt itself.

### Route 1: logs that keep the prompts

Tokenize each prompt with the tokenizer of your model, cut the tokens into blocks of 64, and give each block an ID that
depends on the whole prefix up to that block. Then identical prefixes get identical IDs, as in the cache of vLLM.

```python
import hashlib

BLOCK = 64


def block_ids(tokens, ids):
    """Cache block IDs of a prompt: one per block of 64 tokens. Identical prefixes give identical IDs."""
    out, chain = [], b""
    for i in range(0, len(tokens), BLOCK):
        chain = hashlib.sha256(chain + repr(tokens[i:i + BLOCK]).encode()).digest()
        out.append(ids.setdefault(chain, len(ids)))
    return out


def build_trace(session_id, model, calls):
    """calls: one dict per request of the session, in time order:
    {"t": seconds since the start, "tokens": token IDs of the whole prompt, "out": output tokens, "api_time": seconds}"""
    ids, requests, previous_end = {}, [], None
    for call in calls:
        entry = {"t": call["t"], "type": "n", "model": model, "in": len(call["tokens"]), "out": call["out"],
                 "hash_ids": block_ids(call["tokens"], ids), "api_time": call["api_time"]}
        if previous_end is not None:
            entry["think_time"] = round(max(0.0, call["t"] - previous_end), 3)
        previous_end = call["t"] + call["api_time"]
        requests.append(entry)
    return {"id": session_id, "models": [model], "block_size": BLOCK, "hash_id_scope": "local", "requests": requests}
```

Write one file per session, all in the same folder:

```python
import json

with open("my-traces/session-1.json", "w") as file:
    json.dump(build_trace("session-1", "my-model", calls), file)
```

Points to know about this code:

- The IDs are small numbers given in order of first appearance, one set of IDs per session.
- A last block that is not full gets its own ID. A call that continues the context then gets a new ID for that block,
  because its content is not the same any more. This is how a real cache behaves.
- `in` is the real number of input tokens. It does not have to be a multiple of 64.
- The code uses `"n"` for every call. Use `"s"` and add `ttft` if you recorded streaming.

This code is tested against AIPerf's model on a synthetic session (`tests/test_traces_doc.py`). It has not been run on
logs from a real agent.

### Route 2: logs with sizes only

If your logs have the token counts but not the prompts, assume that each call continues the previous one: the call
repeats the IDs of the previous call, then adds `ceil((new input - previous input) / 64)` new IDs. You lose what you
cannot see (a context that was compacted, two branches of a conversation), so the cache reuse is only an estimate.
`tools/make_traces.py` builds its sessions this way. You can read its `Session` class as an example.

### What to decide

- **Sub-agents**: the simplest is to leave them out and keep only the main agent's calls. Include them as `"subagent"`
  entries if their load matters to you.
- **Think time**: the delay between the end of a call and the next request. Keep the real values: they decide how many
  calls are in flight at the same time.
- **Long sessions**: `--max-model-len` must cover the longest call. `BENCH_MAX_CONTEXT` discards the sessions that go
  above a limit ([README, section 6.6](../README.md#66-context-shorter-than-the-traces)).

## Check your traces

Validate every file with AIPerf's own model, in the environment where AIPerf is installed:

```bash
python - <<'EOF'
import json
import pathlib
from aiperf.dataset.loader.weka_trace_models import WekaTrace

for file in sorted(pathlib.Path("my-traces").glob("*.json")):
    WekaTrace.model_validate(json.loads(file.read_text()))
    print("ok", file.name)
EOF
```

An error names the field that is wrong. Then do a short run against the fake server, which needs no GPU (start it as in
the README, Appendix E). `BENCH_SHORT_RUN=1` allows a level shorter than 900 s:

```bash
BENCH_URL=http://127.0.0.1:8001 BENCH_MODEL=bench BENCH_API_KEY=x BENCH_AGENTX_DIR=my-traces \
  BENCH_SHORT_RUN=1 BENCH_DURATION=60 ./agentx.sh check-r1 "1"
```

## Good to know

- **Loading can hang.** From 8 traces on, AIPerf rebuilds the prompts with several processes, and this step can stay
  idle forever (seen once, on a small machine). If the run never starts, kill the AIPerf processes (a stuck one also
  blocks the next runs on a cache lock) and set `AIPERF_DATASET_WEKA_PARALLEL_WORKERS=1` to load in a single process.
- **Not verified.** The generator and the replay were tried against the fake server only, not against a real vLLM. The
  exact definition of the hash IDs is in the kv-cache-tester documentation (`trace_replay_tester.md`), which was not
  consulted. The scheme of route 1 keeps the property that matters here (identical prefixes give identical IDs), but it
  may not be identical to theirs.
