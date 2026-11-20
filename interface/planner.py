"""Planned runs: a queue of runs that start later, one after another (interface/README.md).

Pure functions on a list of items, saved in a JSON file so that the queue survives a restart of the interface.
Nothing here starts a process: interface/server.py asks `due()` which item is next and starts it itself.

An item is a dict: id, type, label, params (the form fields, without any key), created, start_at (epoch seconds, or
None for "after the current run"), state, late, run_id, code, error.
States: waiting, running, done, failed, cancelled, interrupted."""
import json
import os
import time

# Form fields that are secrets: never written to the file. A planned run uses the keys of .env instead.
SECRET_FIELDS = ("api_key", "agent_keys")
KEEP_FINISHED = 30        # finished items kept in the file and the list
LATE_AFTER = 120          # seconds after its start time at which a run counts as late
MAX_AHEAD = 30 * 24 * 3600


def clean_params(p):
    """The form fields `p` without the secret ones (a copy)."""
    return {k: v for k, v in p.items() if k not in SECRET_FIELDS and k != "start_at"}


def make_item(p, start_at, now, number):
    """New waiting item for the form fields `p`, to start at `start_at` (epoch seconds) or, if None, as soon as the
    run in progress ends.

    Raises:
        ValueError: if `start_at` is not a number, is more than a minute in the past or more than 30 days ahead."""
    if start_at is not None:
        if isinstance(start_at, bool) or not isinstance(start_at, (int, float)):
            raise ValueError("start time: a number of seconds expected")
        if start_at < now - 60:
            raise ValueError("start time: it is in the past")
        if start_at > now + MAX_AHEAD:
            raise ValueError("start time: more than 30 days ahead")
        start_at = float(start_at)
    return dict(id=f"p{int(now)}-{number:03d}", type=p.get("type"), label=p.get("label"), params=clean_params(p),
                created=now, start_at=start_at, state="waiting", late=False, run_id=None, code=None, error=None)


def order_key(item):
    """Waiting items start in the order of their planned time; one without a time counts from its creation."""
    return (item["start_at"] if item["start_at"] is not None else item["created"], item["created"])


def due(items, now):
    """The waiting item that must start now, or None: the earliest one whose start time has come."""
    ready = [i for i in items if i["state"] == "waiting" and (i["start_at"] is None or i["start_at"] <= now)]
    return min(ready, key=order_key) if ready else None


def is_late(item, now):
    """True if the item starts more than LATE_AFTER seconds after its planned time."""
    return item["start_at"] is not None and now - item["start_at"] > LATE_AFTER


def trim(items):
    """The items without the oldest finished ones (all waiting and running items are kept)."""
    active = [i for i in items if i["state"] in ("waiting", "running")]
    finished = [i for i in items if i["state"] not in ("waiting", "running")]
    finished.sort(key=lambda i: i["created"])
    return [i for i in items if i in active or i in finished[-KEEP_FINISHED:]]


def public_view(item):
    """The item without its form fields (what the page shows)."""
    return {k: v for k, v in item.items() if k != "params"}


def load(path):
    """Items and pause reason saved in `path`: (items, paused). An item that was running when the interface stopped
    becomes `interrupted` and pauses the queue. A missing or unreadable file gives ([], None)."""
    try:
        data = json.loads(open(path, encoding="utf-8").read())
        items = [i for i in data.get("items", []) if isinstance(i, dict) and "id" in i and "params" in i]
        paused = data.get("paused")
    except (OSError, ValueError, AttributeError):
        return [], None
    for i in items:
        if i["state"] == "running":
            i["state"] = "interrupted"
            paused = paused or f"{i['label']} was interrupted when the interface stopped"
    return items, paused


def save(path, items, paused):
    """Write the items and the pause reason to `path` (through a temporary file, so a crash never leaves half a file)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"items": trim(items), "paused": paused, "saved": time.time()}, fh)
    os.replace(tmp, path)
