"""Planned runs: a queue of runs that start later, one after another (interface/README.md).

Pure functions on a list of items, saved in a JSON file so that the queue survives a restart of the interface.
Nothing here starts a process: interface/server.py asks `due()` which item is next and starts it itself.

An item is a dict: id, type, label, params (the form fields, without any key), created, start_at (epoch seconds, or
None for "after the current run"), state, late, run_id, code, error.
States: waiting, running, done, failed, cancelled, interrupted."""
import json
import math
import os
import time

# The form fields the three run builders of interface/server.py read: only these are written to the file (a list of
# what is kept, not of what is dropped: a key, or any other field, never reaches the disk). A planned run uses the
# keys of .env instead of the typed ones.
PARAM_FIELDS = frozenset((
    "type", "label", "url", "model", "metrics", "tokenizer", "root", "levels", "overwrite", "extra", "isl", "osl",
    "temperature", "duration", "requests_per_concurrency", "generate", "traces_dir", "variant", "short_run",
    "max_context", "agents", "mode", "warmup", "scale", "latency_factor", "label_agents"))
STATES = ("waiting", "running", "done", "failed", "cancelled", "interrupted")
KEEP_FINISHED = 30        # finished items kept in the file and the list
LATE_AFTER = 120          # seconds after its start time at which a run counts as late
MAX_AHEAD = 30 * 24 * 3600


def clean_params(p):
    """The form fields of `p` that a run reads (a copy): no key, no start time, nothing else."""
    return {k: v for k, v in p.items() if k in PARAM_FIELDS}


def is_number(v):
    """True for a finite int or float (not a bool, not NaN, not infinity)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:                                    # an integer too large for a float
        return False


def make_item(p, start_at, now, number):
    """New waiting item for the form fields `p`, to start at `start_at` (epoch seconds) or, if None, as soon as the
    run in progress ends.

    Raises:
        ValueError: if `start_at` is not a number, is more than a minute in the past or more than 30 days ahead."""
    if start_at is not None:
        if not is_number(start_at):
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
    finished = sorted((i for i in items if i["state"] not in ("waiting", "running")), key=lambda i: i["created"])
    dropped = {id(i) for i in finished[:max(0, len(finished) - KEEP_FINISHED)]}
    return [i for i in items if id(i) not in dropped]


def public_view(item):
    """The item without its form fields (what the page shows)."""
    return {k: v for k, v in item.items() if k != "params"}


def valid_item(i):
    """True if `i` has every field the queue reads, with the right type (the file may have been edited by hand)."""
    return (isinstance(i, dict) and all(isinstance(i.get(k), str) for k in ("id", "type", "label"))
            and isinstance(i.get("params"), dict) and i.get("state") in STATES and is_number(i.get("created"))
            and (i.get("start_at") is None or is_number(i["start_at"])))


def load(path):
    """Items and pause reason saved in `path`: (items, paused). An item that is not valid, or whose id repeats, is
    left out. An item that was running when the interface stopped becomes `interrupted` and pauses the queue.
    A missing, unreadable or broken file gives ([], None)."""
    try:
        data = json.loads(open(path, encoding="utf-8").read())
        raw = data.get("items", [])
        paused = data.get("paused")
        paused = paused if isinstance(paused, str) else None
        items, seen = [], set()
        for i in raw if isinstance(raw, list) else []:
            if valid_item(i) and i["id"] not in seen:
                seen.add(i["id"])
                i["params"] = clean_params(i["params"])        # a hand-edited file cannot bring a key back
                for k, default in (("late", False), ("run_id", None), ("code", None), ("error", None)):
                    i.setdefault(k, default)
                items.append(i)
    except (OSError, ValueError, AttributeError, TypeError, RecursionError):
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
