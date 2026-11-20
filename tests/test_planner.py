"""Planned runs: the queue (interface/planner.py) and its use by the interface server."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from interface import planner, server

NOW = 1_800_000_000.0
FORM = {"type": "synthetic", "url": "http://127.0.0.1:8000", "model": "bench", "label": "plan-r1", "levels": "1",
        "isl": 100, "osl": 10, "root": "results"}


def item(label="a", start_at=None, created=NOW, state="waiting"):
    return dict(id=label, type="synthetic", label=label, params={}, created=created, start_at=start_at, state=state,
                late=False, run_id=None, code=None, error=None)


# ------------------------------------------------------------------ planner.py
def test_clean_params_keeps_only_the_fields_a_run_reads():
    p = {**FORM, "api_key": "sk-secret", "agent_keys": "report=sk-x", "start_at": 5, "prometheus_key": "x", "other": 1}
    cleaned = planner.clean_params(p)
    assert set(cleaned) == set(FORM) and cleaned["label"] == "plan-r1"
    assert "api_key" in p                                            # the original is untouched


def test_every_field_the_builders_read_is_kept():
    """A field that a run reads but the file drops would change a planned run: compare with server.py itself."""
    import re
    source = open(server.__file__, encoding="utf-8").read()
    read = set(re.findall(r'p\.get\("([a-z_0-9]+)"', source)) | set(re.findall(r'p\["([a-z_0-9]+)"\]', source))
    not_run_fields = {"api_key", "agent_keys", "start_at", "by", "gpu", "litellm", "period", "prometheus", "prometheus_key",
                      "target", "ttft_max", "view", "vllm"}      # secrets, the start time, and the observation form
    assert read - not_run_fields <= planner.PARAM_FIELDS


def test_make_item_accepts_a_time_or_none():
    assert planner.make_item(FORM, NOW + 3600, NOW, 1)["start_at"] == NOW + 3600
    after = planner.make_item(FORM, None, NOW, 2)
    assert after["start_at"] is None and after["state"] == "waiting" and after["label"] == "plan-r1"


@pytest.mark.parametrize("start", [NOW - 3600, NOW + 31 * 24 * 3600, "tomorrow", True, [1], float("nan"), float("inf"), -float("inf"), 10 ** 400])
def test_make_item_refuses_a_bad_start_time(start):
    with pytest.raises(ValueError):
        planner.make_item(FORM, start, NOW, 1)


def test_make_item_never_keeps_a_key():
    made = planner.make_item({**FORM, "api_key": "sk-secret"}, None, NOW, 1)
    assert "sk-secret" not in json.dumps(made)


def test_due_takes_the_earliest_item_whose_time_has_come():
    items = [item("late", NOW + 600), item("b", NOW - 30), item("a", NOW - 90), item("done", NOW - 500, state="done")]
    assert planner.due(items, NOW)["label"] == "a"
    assert planner.due([item("future", NOW + 1)], NOW) is None
    assert planner.due([], NOW) is None


def test_an_item_without_a_time_counts_from_its_creation():
    items = [item("second", None, created=NOW + 10), item("first", None, created=NOW)]
    assert planner.due(items, NOW + 20)["label"] == "first"


def test_is_late_only_after_a_margin():
    assert not planner.is_late(item(start_at=NOW), NOW + 60)
    assert planner.is_late(item(start_at=NOW), NOW + 600)
    assert not planner.is_late(item(start_at=None), NOW + 600)


def test_trim_keeps_waiting_items_and_a_limited_history():
    finished = [item(f"f{n}", created=NOW + n, state="done") for n in range(planner.KEEP_FINISHED + 5)]
    kept = planner.trim([item("w"), item("r", state="running")] + finished)
    assert {i["label"] for i in kept if i["state"] in ("waiting", "running")} == {"w", "r"}
    assert len([i for i in kept if i["state"] == "done"]) == planner.KEEP_FINISHED
    assert "f0" not in {i["label"] for i in kept}


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "planned.json"
    items = [planner.make_item(FORM, NOW + 60, NOW, 1)]
    planner.save(path, items, "a run failed")
    loaded, paused = planner.load(path)
    assert loaded == items and paused == "a run failed"
    assert not (tmp_path / "planned.json.tmp").exists()


def test_a_run_in_progress_when_the_interface_stopped_is_interrupted_and_pauses(tmp_path):
    path = tmp_path / "planned.json"
    planner.save(path, [item("x", state="running"), item("y")], None)
    loaded, paused = planner.load(path)
    assert [i["state"] for i in loaded] == ["interrupted", "waiting"]
    assert paused and "x" in paused


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"items": [{"nothing": 1}]}', '{"items": 5}', '{"items": "x"}',
                                      '{"items": [{"id": "a", "params": {}}]}', '{"items": [{"id": 1, "params": {}, "state": "waiting"}]}',
                                      '{"items": [{"id": "a", "type": "synthetic", "label": "l", "params": [], "state": "waiting", "created": 1}]}',
                                      '{"items": [{"id": "a", "type": "synthetic", "label": "l", "params": {}, "state": "nope", "created": 1}]}',
                                      '{"items": [{"id": "a", "type": "synthetic", "label": "l", "params": {}, "state": "waiting", "created": "x"}]}',
                                      '{"items": [{"id": "a", "type": "synthetic", "label": "l", "params": {}, "state": "waiting", "created": 1, "start_at": "soon"}]}',
                                      '{"items": [{"id": "a", "type": "synthetic", "label": "l", "params": {}, "state": "waiting", "created": NaN}]}'])
def test_a_missing_or_broken_file_gives_an_empty_queue(tmp_path, content):
    path = tmp_path / "planned.json"
    if content is not None:
        path.write_text(content)
    assert planner.load(path)[0] == []


# ------------------------------------------------------------------ interface/server.py
@pytest.fixture
def queue(monkeypatch, tmp_path):
    """An empty queue saved in a temporary file, no run in progress, results under a temporary folder."""
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "PLANNED_FILE", tmp_path / "planned.json")
    monkeypatch.setattr(server, "PLANNED", [])
    monkeypatch.setattr(server, "PAUSE", [None])
    monkeypatch.setattr(server, "RUNS", {})
    monkeypatch.setattr(server, "START_LOCK", asyncio.Lock())      # one lock per test: it binds to the loop that uses it
    return tmp_path


def post(body):
    class Req:
        async def json(self):
            return body
    resp = asyncio.run(server.plan_run(Req()))
    return resp.status, json.loads(resp.text)


def test_plan_run_queues_a_valid_run_and_saves_it(queue):
    status, answer = post({**FORM, "start_at": time.time() + 3600})
    assert status == 200 and len(server.PLANNED) == 1
    assert server.PLANNED[0]["id"] == answer["id"]
    assert json.loads((queue / "planned.json").read_text())["items"][0]["label"] == "plan-r1"


def test_plan_run_refuses_what_a_run_started_now_would_refuse(queue):
    status, answer = post({**FORM, "levels": "x", "start_at": None})
    assert status == 400 and "levels" in answer["error"] and server.PLANNED == []


def test_plan_run_refuses_a_bad_start_time(queue):
    status, answer = post({**FORM, "start_at": "tomorrow"})
    assert status == 400 and "start time" in answer["error"]


def test_a_key_typed_in_the_form_is_refused_unless_env_has_the_key(queue, monkeypatch):
    typed = {**FORM, "api_key": "sk-typed-in-the-form", "start_at": None}
    status, answer = post(typed)
    assert status == 400 and ".env" in answer["error"]
    monkeypatch.setenv("BENCH_URL", FORM["url"])
    monkeypatch.setenv("BENCH_API_KEY", "sk-typed-in-the-form")
    status, _ = post(typed)
    assert status == 200
    assert "sk-typed-in-the-form" not in (queue / "planned.json").read_text()      # the file never holds a key


def test_per_agent_keys_cannot_be_planned(queue):
    form = {**FORM, "type": "agents", "agents": ["report"], "mode": "scripted", "duration": 60, "agent_keys": "report=sk-x"}
    status, answer = post(form)
    assert status == 400 and "per-agent keys" in answer["error"]


def test_a_label_that_exists_is_refused_unless_replacing(queue):
    (queue / "results" / "plan-r1").mkdir(parents=True)
    (queue / "results" / "plan-r1" / "c1").write_text("x")
    status, answer = post({**FORM, "start_at": None})
    assert status == 400 and "already exists" in answer["error"]
    status, _ = post({**FORM, "start_at": None, "overwrite": True})
    assert status == 200


def test_the_same_label_cannot_be_queued_twice(queue):
    assert post({**FORM, "start_at": None})[0] == 200
    status, answer = post({**FORM, "start_at": NOW + 99999})
    assert status == 400 and "already has this label" in answer["error"]
    assert post({**FORM, "label": "plan-r2", "start_at": None})[0] == 200


def started_by_a_fake(monkeypatch):
    """Replace start_run: it records the run as in progress instead of starting a process."""
    started = []

    async def fake(p):
        eid = f"run-{len(started) + 1}"
        server.RUNS[eid] = dict(id=eid, type=p["type"], label=p["label"], start=NOW, end=None, code=None)
        started.append(p["label"])
        return eid
    monkeypatch.setattr(server, "start_run", fake)
    return started


def test_the_queue_runs_one_at_a_time_in_order(queue, monkeypatch):
    started = started_by_a_fake(monkeypatch)
    for n in (1, 2):
        assert post({**FORM, "label": f"plan-r{n}", "start_at": None})[0] == 200
    asyncio.run(server.tick(NOW))
    assert started == ["plan-r1"] and server.PLANNED[0]["state"] == "running"
    asyncio.run(server.tick(NOW + 5))
    assert started == ["plan-r1"]                                  # one run in progress: the second waits
    server.RUNS["run-1"]["code"] = 0
    asyncio.run(server.tick(NOW + 10))
    assert [i["state"] for i in server.PLANNED] == ["done", "running"] and started == ["plan-r1", "plan-r2"]


def test_a_run_waits_for_its_time(queue, monkeypatch):
    started = started_by_a_fake(monkeypatch)
    soon = time.time() + 3600
    assert post({**FORM, "start_at": soon})[0] == 200
    asyncio.run(server.tick(soon - 60))
    assert started == []
    asyncio.run(server.tick(soon + 1))
    assert started == ["plan-r1"] and server.PLANNED[0]["late"] is False
    server.RUNS["run-1"]["code"] = 0
    server.PLANNED[0]["state"], server.PLANNED[0]["run_id"] = "waiting", None
    server.RUNS.clear()
    asyncio.run(server.tick(soon + 3000))
    assert server.PLANNED[0]["late"] is True


def test_a_failed_run_pauses_the_queue_until_it_is_resumed(queue, monkeypatch):
    started = started_by_a_fake(monkeypatch)
    for n in (1, 2):
        assert post({**FORM, "label": f"plan-r{n}", "start_at": None})[0] == 200
    asyncio.run(server.tick(NOW))
    server.RUNS["run-1"]["code"] = 1
    asyncio.run(server.tick(NOW + 5))
    assert server.PLANNED[0]["state"] == "failed" and server.PAUSE[0] and "plan-r1" in server.PAUSE[0]
    asyncio.run(server.tick(NOW + 10))
    assert started == ["plan-r1"]                                  # paused: the second does not start
    asyncio.run(server.resume_planned(None))
    asyncio.run(server.tick(NOW + 15))
    assert started == ["plan-r1", "plan-r2"]


def test_a_planned_run_that_cannot_start_fails_and_pauses(queue, monkeypatch):
    async def refuses(p):
        raise server.Refusal("the target URL is not allowed")
    monkeypatch.setattr(server, "start_run", refuses)
    assert post({**FORM, "start_at": None})[0] == 200
    asyncio.run(server.tick(NOW))
    assert server.PLANNED[0]["state"] == "failed" and "not allowed" in server.PLANNED[0]["error"]
    assert server.PAUSE[0] and "could not start" in server.PAUSE[0]


def test_a_run_started_by_hand_is_not_interrupted_by_the_queue(queue, monkeypatch):
    started = started_by_a_fake(monkeypatch)
    server.RUNS["manual"] = dict(id="manual", type="synthetic", label="m", start=NOW, end=None, code=None)
    assert post({**FORM, "start_at": None})[0] == 200
    asyncio.run(server.tick(NOW))
    assert started == []


def test_cancel_only_a_waiting_run(queue):
    answer = post({**FORM, "start_at": None})[1]
    req = SimpleNamespace(match_info={"id": answer["id"]})
    assert asyncio.run(server.cancel_planned(req)).status == 200
    assert server.PLANNED[0]["state"] == "cancelled"
    assert asyncio.run(server.cancel_planned(req)).status == 400
    assert asyncio.run(server.cancel_planned(SimpleNamespace(match_info={"id": "nope"}))).status == 400


def test_the_list_hides_the_form_fields(queue):
    post({**FORM, "start_at": None})
    data = json.loads(asyncio.run(server.list_planned(None)).text)
    assert data["paused"] is None and "params" not in data["items"][0] and data["items"][0]["label"] == "plan-r1"


def test_load_keeps_the_valid_items_once_and_drops_the_others(tmp_path):
    good = item("good", NOW + 5)
    good["params"] = {"label": "good"}
    data = {"items": [good, dict(good), {"id": "bad"}, "text", None, {**good, "id": "g2", "label": "g2", "state": "running"}], "paused": 7}
    path = tmp_path / "planned.json"
    path.write_text(json.dumps(data))
    loaded, paused = planner.load(path)
    assert [i["id"] for i in loaded] == ["good", "g2"]               # the repeated id and the invalid ones are gone
    assert loaded[1]["state"] == "interrupted" and paused and "g2" in paused


def test_trim_removes_the_oldest_finished_items_from_the_list_it_returns():
    items = [item(f"f{n}", created=NOW + n, state="done") for n in range(planner.KEEP_FINISHED + 2)]
    kept = planner.trim(items)
    assert [i["label"] for i in kept][0] == "f2" and len(kept) == planner.KEEP_FINISHED


def test_saving_the_queue_also_trims_it_in_memory(queue):
    server.PLANNED[:] = [item(f"f{n}", created=NOW + n, state="done") for n in range(planner.KEEP_FINISHED + 4)]
    server.save_planned()
    assert len(server.PLANNED) == planner.KEEP_FINISHED


def test_a_key_typed_in_the_form_must_be_the_key_of_env(queue, monkeypatch):
    monkeypatch.setenv("BENCH_URL", FORM["url"])
    monkeypatch.setenv("BENCH_API_KEY", "sk-from-env")
    status, answer = post({**FORM, "api_key": "sk-another", "start_at": None})
    assert status == 400 and "differs" in answer["error"]
    assert post({**FORM, "api_key": "sk-from-env", "start_at": None})[0] == 200


def test_a_start_that_fails_for_any_reason_fails_the_item_and_pauses(queue, monkeypatch):
    async def breaks(p):
        raise OSError("bash is missing")
    monkeypatch.setattr(server, "start_run", breaks)
    assert post({**FORM, "start_at": None})[0] == 200
    asyncio.run(server.tick(NOW))
    assert server.PLANNED[0]["state"] == "failed" and "bash is missing" in server.PLANNED[0]["error"]
    assert server.PAUSE[0]
    asyncio.run(server.tick(NOW + 5))                              # paused: it is not tried again every few seconds


def test_an_item_that_cannot_say_its_folder_does_not_block_planning(queue):
    server.PLANNED.append({**item("broken"), "params": {"label": "../x", "root": 5}})
    assert post({**FORM, "start_at": None})[0] == 200


def test_the_scheduler_does_nothing_while_the_interface_stops(queue, monkeypatch):
    started = started_by_a_fake(monkeypatch)
    assert post({**FORM, "start_at": None})[0] == 200
    server.STOPPING[0] = True
    try:
        asyncio.run(server.tick(NOW))
    finally:
        server.STOPPING[0] = False
    assert started == [] and server.PLANNED[0]["state"] == "waiting"


def test_a_stopped_run_is_not_recorded_as_failed_during_shutdown(queue, monkeypatch):
    started_by_a_fake(monkeypatch)
    assert post({**FORM, "start_at": None})[0] == 200
    asyncio.run(server.tick(NOW))
    server.RUNS["run-1"]["code"] = -15
    server.STOPPING[0] = True
    try:
        asyncio.run(server.tick(NOW + 5))
    finally:
        server.STOPPING[0] = False
    assert server.PLANNED[0]["state"] == "running" and server.PAUSE[0] is None      # it becomes "interrupted" at the next load


def test_two_starts_at_once_give_one_run(queue, monkeypatch):
    """The check "no run in progress" and the start go together: a manual launch cannot slip in between."""
    spawned = []

    async def fake_exec(*cmd, **kw):
        await asyncio.sleep(0.05)
        spawned.append(cmd)
        return SimpleNamespace(pid=1, wait=lambda: asyncio.sleep(3600))
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server, "LOGS_DIR", queue / "logs")
    monkeypatch.setattr(server, "follow", lambda *a: asyncio.sleep(0))
    body = {**FORM, "tokenizer": "", "requests_per_concurrency": 2}

    async def both():
        return await asyncio.gather(server.start_run(body), server.start_run({**body, "label": "other"}), return_exceptions=True)
    results = asyncio.run(both())
    assert len(spawned) == 1 and sum(isinstance(r, server.Busy) for r in results) == 1


class GuardReq:
    def __init__(self, path, method="POST", token="t", content_type="application/json", origin=None, host="127.0.0.1:8765"):
        self.path, self.method, self.content_type, self.host = path, method, content_type, host
        self.headers = {"X-Bench-Token": token, **({"Origin": origin} if origin else {})}


@pytest.mark.parametrize("path", ["/api/planned", "/api/planned/resume", "/api/planned/p1-001/cancel"])
def test_the_guard_protects_the_planned_routes(monkeypatch, path):
    monkeypatch.setattr(server, "TOKEN", "t")
    monkeypatch.setattr(server, "PUBLIC_PORT", [8765])

    async def handler(req):
        return server.web.json_response({"ok": True})

    def status(**kw):
        return asyncio.run(server.guard(GuardReq(path, **kw), handler)).status
    assert status() == 200
    assert status(token="wrong") == 401 and status(token="") == 401
    assert status(content_type="text/plain") == 415
    assert status(origin="http://evil.example") == 403
    assert status(host="evil.example") == 403


def test_the_planned_routes_are_registered():
    source = open(server.__file__, encoding="utf-8").read()
    for route in ('web.post("/api/planned", plan_run)', 'web.get("/api/planned", list_planned)',
                  'web.post("/api/planned/resume", resume_planned)', 'web.post("/api/planned/{id}/cancel", cancel_planned)'):
        assert route in source


def test_load_cleans_the_fields_of_a_hand_edited_file(tmp_path):
    path = tmp_path / "planned.json"
    edited = item("e")
    edited["params"] = {"label": "e", "api_key": "sk-put-back", "agent_keys": "report=sk-x"}
    path.write_text(json.dumps({"items": [edited], "paused": None}))
    assert planner.load(path)[0][0]["params"] == {"label": "e"}


def test_a_deeply_nested_file_gives_an_empty_queue(tmp_path):
    path = tmp_path / "planned.json"
    path.write_text("[" * 100000)
    assert planner.load(path) == ([], None)
