"""Sizes of the three test agent profiles, at scale 1 (see agents.py). Kept apart, without imports, so that the
report can read them without the OpenAI client."""

PROFILES = {
    "code_review": dict(name="Code review", system=3000, tools=4, tokens_per_tool=300,
                list_files=1500, plan=300, sub_agents=3, sub_agent_system=2000,
                steps=6, read=22000, analysis=400, summary=1500, tool_latency_s=5.0),
    "pentest": dict(name="Step-by-step pentest", system=3000, tools=6, tokens_per_tool=300,
                steps=15, tool_result=4000, thinking=250, report=800, tool_latency_s=8.0),
    "report": dict(name="Report writing", system=2000, notes=125000, plan=800,
                 writing=4000, tool_latency_s=2.0),
}


def calls_per_task(agent):
    """Number of model calls one task of `agent` makes when it goes through every step of its profile:
    code review = plan + steps of each sub-agent + summary, pentest = steps + report, report = outline + writing."""
    p = PROFILES[agent]
    return {"code_review": 2 + p.get("sub_agents", 0) * p.get("steps", 0),
            "pentest": 1 + p.get("steps", 0), "report": 2}[agent]
