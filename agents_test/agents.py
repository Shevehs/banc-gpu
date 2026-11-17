"""Three test agent profiles that reproduce the SHAPE of an agentic workload.

They do no real offensive work: their tools are simulated (common.make_text) and their
content is fabricated. Only the sizes (system prompt, tool definitions, tool outputs,
model outputs), the number of steps, the parallelism and the tool latencies matter.

Default sizes (ASSUMPTIONS, to adapt to the target workload): code review ≈ 135K tokens per
sub-agent, 3 sub-agents; step-by-step pentest ≈ 69K, 1 thread; report ≈ 135K, 1 thread
(estimated with the Qwen vocabulary).
The sizes of the profiles are in profiles.py.
All sizes are multiplied by the scale factor (--scale, e.g. 0.2 on a small server).
"""
import random
import threading
import time

from agents_test.common import tool_definitions, make_text
from agents_test.profiles import PROFILES

# Fixed seeds: the system prompt and the tool definitions are identical from one task to the
# next (as with a real agent), so the prefix cache can reuse them.
FIXED_SEEDS = {"code_review": 3_000_000, "pentest": 2_000_000, "report": 9_000_000}
TOOLS = {"code_review": ["list_files", "read_file", "search_pattern", "static_analysis"],
          "pentest": ["scan_ports", "identify_service", "read_log", "test_auth", "search_vuln", "note"]}
TEXT_KINDS = {"list_files": "code", "read_file": "code", "search_pattern": "code", "static_analysis": "code",
          "scan_ports": "scan", "identify_service": "scan", "read_log": "log",
          "test_auth": "log", "search_vuln": "scan", "note": "notes"}


# Multiplies the simulated tool latencies (set by the launcher, --latency-factor option).
LATENCY_FACTOR = 1.0


def _scale(x, scale, minimum=32):
    return max(minimum, int(round(x * scale)))


def _system_prompt(agent, p, scale, mode):
    """Fixed system prompt. In scripted mode, the tool definitions are included as text
    (same prompt size, without depending on the server's tool-call parser)."""
    g = FIXED_SEEDS[agent]
    s = f"You are the {p['name']} test agent. Answer in plain text.\n" + make_text("notes", _scale(p["system"], scale), g)
    tools = None
    if agent in TOOLS:
        tools = tool_definitions(TOOLS[agent], _scale(p["tokens_per_tool"], scale), g + 1000)
        if mode == "scripted":
            s += "\nAvailable tools:\n" + "\n".join(f"{o['function']['name']}: {o['function']['description']}" for o in tools)
            tools = None
    return s, tools


def _run_tool(name, n_tokens, seed, latency_s):
    """Run a simulated tool: wait for the expected latency and return text of a fixed size."""
    # variable latency (± 30 %) around the expected value, to avoid an overly regular load
    time.sleep(latency_s * LATENCY_FACTOR * random.uniform(0.7, 1.3))
    return make_text(TEXT_KINDS.get(name, "notes"), n_tokens, seed)


def _agent_loop(m, messages, tools, n_steps, n_output, n_result, name_tool, ctx, seed, latency_s, mode):
    """Tool → model loop. Scripted: the tool is forced at each step.
    Autonomous: run the tools chosen by the model, and stop when it calls none."""
    last = ""
    for e in range(n_steps):
        if mode == "scripted":
            result = _run_tool(name_tool, n_result, seed + e, latency_s)
            messages.append({"role": "user", "content": f"Result of {name_tool}:\n{result}"})
            last, _ = m.call(messages, n_output, {**ctx, "step": e})
            messages.append({"role": "assistant", "content": last})
        else:
            last, calls = m.call(messages, n_output, {**ctx, "step": e}, tools)
            if not calls:
                break
            ids = [f"call_{e}_{k}" for k in range(len(calls))]
            messages.append({"role": "assistant", "content": last or None, "tool_calls": [
                {"id": i, "type": "function", "function": {"name": a["name"] or "note", "arguments": a["args"] or "{}"}}
                for i, a in zip(ids, calls, strict=True)]})
            for i, a in zip(ids, calls, strict=True):
                messages.append({"role": "tool", "tool_call_id": i,
                                 "content": _run_tool(a["name"], n_result, seed + 100 * e + len(i), latency_s)})
    return last


def code_review(m, task, seed, scale, mode):
    """Code-review profile: file list, plan, parallel analysis sub-agents, then a summary.

    `m` is the Model that logs the calls, `task` the task id, `seed` seeds the fabricated text, `scale`
    multiplies all sizes and `mode` is "scripted" or "autonomous". Returns nothing."""
    p = PROFILES["code_review"]; ctx = {"agent": "code_review", "task": task}
    sysmsg, tools = _system_prompt("code_review", p, scale, mode)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": "Security review of the repository. File list:\n"
             + _run_tool("list_files", _scale(p["list_files"], scale), seed, p["tool_latency_s"])}]
    plan, _ = m.call(msgs, _scale(p["plan"], scale), {**ctx, "thread": 0, "step": "plan"})
    msgs.append({"role": "assistant", "content": plan})
    summaries = [None] * p["sub_agents"]

    def sub_agent(k):
        s_sys = "Analysis sub-agent.\n" + make_text("notes", _scale(p["sub_agent_system"], scale), FIXED_SEEDS["code_review"] + 7)
        s_msgs = [{"role": "system", "content": s_sys},
                  {"role": "user", "content": f"Analyze part {k + 1} of the repository according to this plan:\n{plan}"}]
        summaries[k] = _agent_loop(m, s_msgs, tools, p["steps"], _scale(p["analysis"], scale), _scale(p["read"], scale),
                             "read_file", {**ctx, "thread": k + 1}, seed * 10 + k * 1000, p["tool_latency_s"], mode)

    threads = [threading.Thread(target=sub_agent, args=(k,)) for k in range(p["sub_agents"])]
    for f in threads: f.start()
    for f in threads: f.join()
    msgs.append({"role": "user", "content": "Sub-agent summaries:\n" + "\n---\n".join(r or "" for r in summaries)})
    m.call(msgs, _scale(p["summary"], scale), {**ctx, "thread": 0, "step": "summary"})


def pentest(m, task, seed, scale, mode):
    """Step-by-step pentest profile: one thread of simulated tool steps, then the final report.
    Same parameters as `code_review`."""
    p = PROFILES["pentest"]; ctx = {"agent": "pentest", "task": task, "thread": 0}
    sysmsg, tools = _system_prompt("pentest", p, scale, mode)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": "Assess the simulated test environment, step by step."}]
    _agent_loop(m, msgs, tools, p["steps"], _scale(p["thinking"], scale), _scale(p["tool_result"], scale),
            "scan_ports", ctx, seed, p["tool_latency_s"], mode)
    msgs.append({"role": "user", "content": "Write the final report."})
    m.call(msgs, _scale(p["report"], scale), {**ctx, "step": "report"})


def report_writer(m, task, seed, scale, mode):
    """Report-writing profile: long engagement notes, an outline, then the full report (the second call shares
    its beginning with the first). Same parameters as `code_review`."""
    p = PROFILES["report"]; ctx = {"agent": "report", "task": task, "thread": 0}
    sysmsg, _ = _system_prompt("report", p, scale, mode)
    time.sleep(p["tool_latency_s"] * LATENCY_FACTOR)  # reading the engagement notes (simulated)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": "Engagement notes:\n" + make_text("notes", _scale(p["notes"], scale), seed)
             + "\nPropose the report outline."}]
    plan, _ = m.call(msgs, _scale(p["plan"], scale), {**ctx, "step": "plan"})
    # Second call: same beginning as the first, so it can be reused by the prefix cache.
    msgs += [{"role": "assistant", "content": plan}, {"role": "user", "content": "Write the full report."}]
    m.call(msgs, _scale(p["writing"], scale), {**ctx, "step": "writing"})


AGENTS = {"code_review": code_review, "pentest": pentest, "report": report_writer}
