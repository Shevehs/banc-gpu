"""observation/query.py: parsing, Prometheus client, windows, discard rules, merging, views and command line."""
import argparse
import csv
import time

import pytest

from observation import query
from observation.analysis import COUNTERS, HISTOGRAMS
from tests import fixtures, helpers

DAY = fixtures.DAY


# ---------------------------------------------------------------- helpers
def window(start=DAY, length=300.0, requests=100, tpot=None, **kw):
    """A window as collect() returns it: usable by default (one target, scraped all along, no restart)."""
    tpot = fixtures.cumulative_histogram(requests, tpot or fixtures.TPOT_FAST)
    o = {"start": start, "end": start + length,
         "counters": {k: None for k in COUNTERS} | {"output_tokens": requests * 400.0, "completed_requests": float(requests),
                                                    "preemptions": 0.0, "failed_requests": 0.0},
         "histograms": {"ttft": fixtures.cumulative_histogram(requests, fixtures.TTFT_OK), "tpot": tpot},
         "gauges": {"kv_max": 0.5, "waiting_max": 0.0, "running_avg": 4.0, "running_max": 6.0},
         "targets": 1, "targets_always_up": 1, "targets_at_start": 1, "resets": 0, "scrapes": length / 15,
         "targets_at_boundary": 1, "targets_at_end": 1}
    o.update(kw)
    return o


def arguments(**kw):
    """The namespace of the command line, as the views and evaluate() read it."""
    base = {"target": 30.0, "gpu": 2, "ttft_max": None, "period": "range", "by": "metadata_agent"}
    return argparse.Namespace(**(base | kw))


def agent(n, output=400.0, speed_ok=True):
    """Per-agent totals of one window for `n` requests (every request streamed unless speed_ok is False)."""
    return fixtures.agent_window(output=n * output, input_tokens=n * 1000.0, requests=n, ttft_seconds=n * 0.2,
                                 latency_seconds=n * 5.0, streamed=None if speed_ok else n - 1)


# ---------------------------------------------------------------- formats and parsing
def test_fmt_and_pct():
    assert query.fmt(None) == "—" and query.fmt("") == "—" and query.fmt(0) == "0.0"
    assert query.fmt(1.26) == "1.3" and query.fmt(1.26, 0) == "1" and query.fmt("2.5", 2) == "2.50"
    assert query.pct(None) == "—" and query.pct("") == "—" and query.pct(0.456) == "46%" and query.pct(0) == "0%"


@pytest.mark.parametrize("text,seconds", [("300", 300), ("300s", 300), ("5m", 300), ("2h", 7200), ("1d", 86400),
                                          ("1.5h", 5400), (" 15 s ", 15), (30, 30), ("0", 0)])
def test_parse_duration(text, seconds):
    assert query.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "m", "5x", "-5m", "5 m 3", "1e3", "five", "5mm"])
def test_parse_duration_rejects(text):
    with pytest.raises(SystemExit) as e:
        query.parse_duration(text)
    assert "invalid duration" in str(e.value)


def test_parse_instant_timestamp_and_local_formats(utc_timezone):
    assert query.parse_instant("1715644800") == 1715644800.0
    assert query.parse_instant("1715644800.5") == 1715644800.5
    assert query.parse_instant("2024-05-14 09:30") == DAY + 9.5 * 3600
    assert query.parse_instant("2024-05-14 09:30:15") == DAY + 9.5 * 3600 + 15
    assert query.parse_instant("2024-05-14T09:30") == DAY + 9.5 * 3600
    assert query.parse_instant("2024-05-14T09:30:15") == DAY + 9.5 * 3600 + 15
    assert query.parse_instant("  2024-05-14 00:00  ") == DAY


def test_parse_instant_follows_the_local_time_zone(monkeypatch):
    monkeypatch.setenv("TZ", "Etc/GMT-2")        # UTC+2 (POSIX sign inverted)
    time.tzset()
    try:
        assert query.parse_instant("2024-05-14 02:00") == DAY
    finally:
        monkeypatch.undo()
        time.tzset()


@pytest.mark.parametrize("text", ["", "yesterday", "2024-05-14", "14/05/2024 09:30", "2024-13-01 00:00", "-5"])
def test_parse_instant_rejects(text):
    with pytest.raises(SystemExit) as e:
        query.parse_instant(text)
    assert "invalid date" in str(e.value)


@pytest.mark.parametrize("selector", ['job="vllm"', ' job="vllm" ', 'job="vllm",instance=~"10.0.0.*"',
                                      'job!="x" , env=~"a|b"', 'job_1!~"z"'])
def test_check_selector_accepts(selector):
    assert query.check_selector(selector, "--vllm") == selector.strip()


@pytest.mark.parametrize("selector", [None, "", "job=vllm", 'job="vllm"}', '{job="vllm"}', 'job="a" b="c"',
                                      'job="a\\"b"', '1job="x"', 'job="x",', 'job=="x"', 'up) or vector(1) #'])
def test_check_selector_rejects(selector):
    with pytest.raises(SystemExit) as e:
        query.check_selector(selector, "--litellm")
    assert str(e.value).startswith("--litellm: label selector expected")


@pytest.mark.parametrize("label,ok", [("api_key_alias", True), ("team_alias", True), ("user", True), ("end_user", True),
                                      ("metadata_agent", True), ("metadata_" + "x" * 40, True),
                                      ("metadata_" + "x" * 41, False), ("metadata_", False), ("model", False),
                                      ("agent", False), ("user ", False), ("metadata_a-b", False)])
def test_by_label_pattern(label, ok):
    assert bool(query.LABEL.match(label)) is ok


# ---------------------------------------------------------------- PromQL
def test_increase_with_and_without_grouping():
    plain = query.increase('x{job="v"}', 300)
    assert plain.startswith("sum(") and "(x{job=\"v\"} - x{job=\"v\"} offset 300s) >= 0" in plain
    assert 'unless x{job="v"} offset 300s' in plain and "(up offset 300s == 1)" in plain
    grouped = query.increase("x", 600.0, "a, le")
    assert grouped.startswith("sum by (a, le) (") and "offset 600s" in grouped


def test_restart_signs_lists_every_witness_and_the_start_time_over_window_plus_one_scrape():
    q = query.restart_signs('{job="v"}', ["a_total", "b_total"], 300, 15)
    assert q.count("resets(") == 2 and 'resets(a_total{job="v"}[315s])' in q
    assert 'changes(process_start_time_seconds{job="v"}[315s])' in q and q.count(" + ") == 2 and "or vector(0)" in q


def test_witnesses_are_counters_the_metrics_module_exposes():
    assert all(n.endswith(("_total", "_count")) for n in query.VLLM_WITNESSES + query.LITELLM_WITNESSES)
    assert set(query.VLLM_WITNESSES) >= {COUNTERS["completed_requests"], COUNTERS["output_tokens"]}


# ---------------------------------------------------------------- Prometheus client
RANGE = "/api/v1/query_range"


def answer(result):
    return (200, {"status": "success", "data": {"resultType": "matrix", "result": result}}, {})


def test_query_range_parses_series_and_drops_non_finite_values(stub):
    s = stub({RANGE: answer([{"metric": {"le": "0.5"}, "values": [[1700000000.4, "3"], [1700000300, "NaN"],
                                                                      [1700000600, "+Inf"], [1700000900, "-Inf"],
                                                                      [1700001200, "7.5"]]},
                             {"values": [[1, "1"]]}])})
    series = query.Prometheus(s.url).query_range("up", 1700000000, 1700001200, 300)
    assert series == [({"le": "0.5"}, {1700000000: 3.0, 1700001200: 7.5}), ({}, {1: 1.0})]


def test_query_range_sends_the_query_the_bounds_and_the_step(stub):
    s = stub({RANGE: answer([])})
    assert query.Prometheus(s.url + "/").query_range('sum(up{job="v"})', 1000.5, 2000, 300.0) == []
    (call,) = s.seen
    path, _, qs = call["path"].partition("?")
    assert path == RANGE
    assert dict(p.split("=", 1) for p in qs.split("&")) == {"query": "sum%28up%7Bjob%3D%22v%22%7D%29", "start": "1000.500",
                                                           "end": "2000.000", "step": "300"}


def test_query_range_bearer_key_only_when_given(stub):
    s = stub({RANGE: answer([])})
    query.Prometheus(s.url).query_range("up", 1, 2, 1)
    query.Prometheus(s.url, key="sk-prom").query_range("up", 1, 2, 1)
    assert "Authorization" not in s.seen[0]["headers"] and s.seen[1]["headers"]["Authorization"] == "Bearer sk-prom"


def test_query_range_http_error_shows_the_status_and_the_body(stub):
    s = stub({RANGE: (400, "parse error: bad query", {})})
    with pytest.raises(SystemExit) as e:
        query.Prometheus(s.url).query_range("sum(", 1, 2, 1)
    assert "HTTP 400" in str(e.value) and "parse error: bad query" in str(e.value) and "sum(" in str(e.value)


def test_query_range_refuses_redirects_so_the_key_never_leaves(stub):
    other = stub({RANGE: answer([])})
    s = stub({RANGE: (302, "", {"Location": other.url + RANGE})})
    with pytest.raises(SystemExit) as e:
        query.Prometheus(s.url, key="sk-prom").query_range("up", 1, 2, 1)
    assert "HTTP 302" in str(e.value) and other.seen == []


def test_query_range_error_status_in_the_json(stub):
    s = stub({RANGE: (200, {"status": "error", "error": "query timed out"}, {})})
    with pytest.raises(SystemExit) as e:
        query.Prometheus(s.url).query_range("up", 1, 2, 1)
    assert "query timed out" in str(e.value)


def test_query_range_unreachable():
    with pytest.raises(SystemExit) as e:
        query.Prometheus(f"http://127.0.0.1:{helpers.free_port()}", timeout_s=2).query_range("up", 1, 2, 1)
    assert "Prometheus unreachable" in str(e.value)


def test_query_range_invalid_json(stub):
    s = stub({RANGE: (200, "<html>", {})})
    with pytest.raises(ValueError):
        query.Prometheus(s.url).query_range("up", 1, 2, 1)


# ---------------------------------------------------------------- collect
def collect_with(prom, windows, w=300, scrape=15, sel_l=None, by=None, first_end=DAY + 300, end=None):
    stub = prom(fixtures.prometheus_responder(first_end, w, windows))
    end = end if end is not None else first_end - w + len(windows) * w
    obs = query.collect(query.Prometheus(stub.url), 'job="vllm"', first_end - w, end, w, scrape, sel_l, by)
    return obs, stub


def test_collect_one_observation_per_window(prom):
    windows = [fixtures.prom_window(300, 100, running_avg=3.0, running_max=5.0, kv=0.5, waiting=1.0),
               fixtures.prom_window(300, 200, running_avg=6.0, running_max=9.0, preemptions=2.0, resets=1)]
    obs, _ = collect_with(prom, windows)
    assert [(o["start"], o["end"]) for o in obs] == [(DAY, DAY + 300), (DAY + 300, DAY + 600)]
    a, b = obs
    assert set(a["counters"]) == set(COUNTERS) and set(a["histograms"]) == set(HISTOGRAMS)
    assert a["counters"]["completed_requests"] == 100 and b["counters"]["completed_requests"] == 200
    assert a["counters"]["output_tokens"] == 40000.0 and b["counters"]["preemptions"] == 2.0
    assert a["histograms"]["tpot"]["0.025"] == pytest.approx(93.0) and a["histograms"]["ttft"]["+Inf"] == 100
    assert a["gauges"] == {"kv_max": 0.5, "waiting_max": 1.0, "running_avg": 3.0, "running_max": 5.0}
    assert (a["targets"], a["targets_always_up"], a["targets_at_start"]) == (1, 1, 1)
    assert (a["targets_at_boundary"], a["targets_at_end"], a["scrapes"]) == (1, 1, 20)
    assert a["resets"] == 0 and b["resets"] == 1
    assert "agents" not in a and "litellm_targets" not in a


def test_collect_asks_only_the_selected_targets_over_the_whole_range(prom):
    obs, stub = collect_with(prom, [fixtures.prom_window(300, 10)] * 3)
    assert len(obs) == 3
    assert {(c["start"], c["end"], c["step"]) for c in stub.seen} == {(DAY + 300, DAY + 900, 300.0)}
    assert all('job="vllm"' in c["query"] for c in stub.seen) and not any("litellm" in c["query"] for c in stub.seen)
    assert any("increase" not in c["query"] and "offset 300s" in c["query"] for c in stub.seen)


def test_collect_failed_requests_none_without_http_series_zero_when_exposed(prom):
    absent, _ = collect_with(prom, [fixtures.prom_window(300, 10, failed=None)])
    assert absent[0]["counters"]["failed_requests"] is None
    exposed, _ = collect_with(prom, [fixtures.prom_window(300, 10, failed=3.0)])
    assert exposed[0]["counters"]["failed_requests"] == 3.0


def test_collect_a_window_without_data_reads_as_unknown(prom):
    obs, _ = collect_with(prom, [fixtures.prom_window(300, 10)], end=DAY + 600)
    gap = obs[1]
    assert gap["targets"] is None and gap["scrapes"] is None and gap["counters"]["completed_requests"] is None
    assert gap["histograms"] == {"ttft": {}, "tpot": {}} and gap["resets"] == 0 and gap["targets_at_end"] == 0


def test_collect_litellm_agents_and_targets(prom):
    windows = [fixtures.prom_window(300, 100, agents={"coder": agent(60), "": agent(5)}, litellm_up=1, litellm_count=2,
                                    litellm_resets=1)]
    (o,), stub = collect_with(prom, windows, sel_l='job="litellm"', by="metadata_agent")
    assert o["litellm_targets"] == (2, 1) and o["litellm_resets"] == 1
    assert sorted(o["agents"]) == ["", "coder"]
    coder = o["agents"]["coder"]
    assert coder["output_tokens"] == 60 * 400.0 and coder["input_tokens"] == 60 * 1000.0
    assert coder["latency_count"] == 60 and coder["ttft_count"] == 60 and coder["latency_sum"] == 300.0
    assert coder["ttft"]["0.5"] == 60 and coder["latency"]["+Inf"] == 60
    assert any("metadata_agent, le" in c["query"] for c in stub.seen)


def test_collect_drops_agents_without_requests_or_tokens(prom):
    idle = fixtures.agent_window(output=0.0, requests=0.0)
    (o,), _ = collect_with(prom, [fixtures.prom_window(300, 1, agents={"idle": idle, "busy": agent(3)})],
                           sel_l='job="litellm"', by="metadata_agent")
    assert list(o["agents"]) == ["busy"]


def test_collect_agent_missing_a_metric_gets_zeroes(prom):
    partial = {"output_tokens": 50.0}      # only the token counter exists for this agent
    (o,), _ = collect_with(prom, [fixtures.prom_window(300, 1, agents={"a": partial})], sel_l='job="litellm"',
                           by="metadata_agent")
    assert o["agents"]["a"]["output_tokens"] == 50.0 and o["agents"]["a"]["latency"] == {} and o["agents"]["a"]["ttft_sum"] == 0.0


def test_collect_range_shorter_than_a_window(stub):
    s = stub({})
    with pytest.raises(SystemExit) as e:
        query.collect(query.Prometheus(s.url), 'job="vllm"', 1000, 1200, 300, 15)
    assert "Range shorter than one window" in str(e.value) and s.seen == []


def test_collect_too_many_windows(stub):
    s = stub({})
    with pytest.raises(SystemExit) as e:
        query.collect(query.Prometheus(s.url), 'job="vllm"', 0, 100_000, 5, 1)
    assert "More than 10,000 windows" in str(e.value) and s.seen == []


# ---------------------------------------------------------------- periods
@pytest.mark.parametrize("hour,minute,label", [(0, 0, "night (0-6 h)"), (5, 59, "night (0-6 h)"), (6, 0, "morning (6-12 h)"),
                                               (11, 59, "morning (6-12 h)"), (12, 0, "afternoon (12-18 h)"),
                                               (17, 59, "afternoon (12-18 h)"), (18, 0, "evening (18-24 h)"),
                                               (23, 59, "evening (18-24 h)")])
def test_period_label_ranges(utc_timezone, hour, minute, label):
    t = DAY + hour * 3600 + minute * 60
    assert query.period_label(t, "range") == label
    assert query.period_label(t, "hour") == f"{hour:02d} h"


def test_period_label_uses_local_time(monkeypatch):
    monkeypatch.setenv("TZ", "Etc/GMT-8")     # UTC+8: 22:00 UTC is 06:00 the next day
    time.tzset()
    try:
        assert query.period_label(DAY + 22 * 3600, "range") == "morning (6-12 h)"
    finally:
        monkeypatch.undo()
        time.tzset()


# ---------------------------------------------------------------- merging
def test_sum_histograms_adds_bound_by_bound():
    assert query.sum_histograms([{"0.1": 1.0, "+Inf": 2.0}, {"0.1": 3.0}, {}]) == {"0.1": 4.0, "+Inf": 2.0}
    assert query.sum_histograms([]) == {}


def test_merge_adds_counters_and_histograms_and_keeps_the_gauge_maximum():
    a = window(DAY, 300, 100, gauges={"kv_max": 0.3, "waiting_max": 0.0, "running_avg": 2.0, "running_max": 4.0})
    b = window(DAY + 300, 600, 300, gauges={"kv_max": 0.9, "waiting_max": 2.0, "running_avg": 6.0, "running_max": 8.0})
    m = query.merge([a, b])
    assert m["counters"]["completed_requests"] == 400 and m["counters"]["output_tokens"] == 160000.0
    assert m["histograms"]["tpot"]["+Inf"] == 400 and m["histograms"]["ttft"]["0.5"] == pytest.approx(384.0)
    assert m["gauges"] == {"kv_max": 0.9, "waiting_max": 2.0, "running_avg": 4.0, "running_max": 8.0}
    assert (m["start"], m["end"]) == (0.0, 900.0)


def test_merge_unreadable_counter_makes_the_sum_unknown():
    a, b = window(), window(DAY + 300, requests=50)
    b["counters"]["output_tokens"] = None
    assert query.merge([a, b])["counters"]["output_tokens"] is None
    assert query.merge([b, a])["counters"]["output_tokens"] is None


def test_merge_failed_requests_none_means_nothing_to_add():
    a, b, c = window(), window(DAY + 300), window(DAY + 600)
    a["counters"]["failed_requests"] = None
    b["counters"]["failed_requests"] = 2.0
    c["counters"]["failed_requests"] = None
    assert query.merge([a, b])["counters"]["failed_requests"] == 2.0
    assert query.merge([b, c])["counters"]["failed_requests"] == 2.0
    assert query.merge([a, c])["counters"]["failed_requests"] is None


def test_merge_gauges_without_values():
    a = window(gauges={"kv_max": None, "waiting_max": None, "running_avg": None, "running_max": None})
    assert query.merge([a, a])["gauges"] == {"kv_max": None, "waiting_max": None, "running_avg": None, "running_max": None}


# ---------------------------------------------------------------- row rendering
def row(**kw):
    base = {"share_of_requests_at_target": 0.95, "share_exact": True, "mean_concurrency": 4.0, "completed_requests": 100.0,
            "failed_requests": 0.0, "output_throughput_per_gpu": 20.0, "interactivity_bound": False,
            "estimated_interactivity_p10": 40.0, "ttft_p95_beyond": False, "estimated_ttft_p95_ms": 500.0,
            "preemptions": 0.0, "prefix_cache": 0.6, "mtp_acceptance": None, "kv_cache_max_fraction": 0.5,
            "verdict": "holds"}
    return base | kw


def test_share_text():
    assert query.share_text(row()) == "95%"
    assert query.share_text(row(share_exact=False)) == "95%*"
    assert query.share_text(row(share_of_requests_at_target=None)) == "—"


def test_format_row_is_aligned_with_the_headers():
    line = query.format_row(row())
    head = query.headers()
    assert len(line.rsplit("  holds", 1)[0]) == len(head.rsplit("  verdict", 1)[0])
    assert line.endswith("  holds") and head.endswith("  verdict")
    assert line.split()[:4] == ["4.0", "100", "0", "20.0"]


def test_format_row_marks_bounds_and_missing_values():
    line = query.format_row(row(interactivity_bound=True, ttft_p95_beyond=True, failed_requests=None, mtp_acceptance=None,
                                share_exact=False))
    assert "<40.0" in line and ">500" in line and "95%*" in line and "—" in line


@pytest.mark.parametrize("histogram,expected", [({"0.1": 1.0, "0.5": 2.0, "+Inf": 2.0}, True), ({}, True),
                                                ({"0.1": 2.0, "0.5": 1.0, "+Inf": 3.0}, False),
                                                ({"0.1": 5.0, "+Inf": 3.0}, False)])
def test_is_increasing(histogram, expected):
    assert query.is_increasing(histogram) is expected


def test_evaluate_a_window_that_holds():
    r = query.evaluate(window(), arguments())
    assert r["verdict"] == "holds" and r["holds"] is True
    assert r["mean_concurrency"] == 4.0 and r["max_concurrency"] == 6.0
    assert r["output_throughput_per_gpu"] == pytest.approx(100 * 400 / 300 / 2)


def test_evaluate_without_requests():
    r = query.evaluate(window(requests=0), arguments())
    assert r["verdict"] == "no requests"


def test_evaluate_inconsistent_histogram_is_undetermined():
    o = window()
    o["histograms"]["tpot"] = {"0.025": 90.0, "0.05": 50.0, "+Inf": 100.0}
    r = query.evaluate(o, arguments())
    assert r["inconsistent_histogram"] is True and r["verdict"] == "undetermined" and r["holds"] is None
    assert r["share_of_requests_at_target"] is None and r["estimated_ttft_p95_ms"] is None


def test_evaluate_ttft_limit():
    assert query.evaluate(window(), arguments(ttft_max=600))["verdict"] == "holds"
    assert query.evaluate(window(), arguments(ttft_max=400))["verdict"] == "does not hold"


def test_evaluate_missing_concurrency_gauge():
    o = window()
    o["gauges"] = {"kv_max": None, "waiting_max": None, "running_avg": None, "running_max": None}
    r = query.evaluate(o, arguments())
    assert r["mean_concurrency"] is None and r["max_concurrency"] is None


# ---------------------------------------------------------------- discarding windows
def test_discard_reason_usable_window():
    assert query.discard_reason(window(), 300, 15, []) is None


@pytest.mark.parametrize("changes,fragment", [
    ({"targets": 0}, "no vLLM target scraped"),
    ({"targets": None}, "no vLLM target scraped"),
    ({"targets_at_start": 0}, "not yet scraped or unreachable at the start"),
    ({"targets_at_start": None}, "not yet scraped or unreachable at the start"),
    ({"targets": 2, "targets_at_start": 2, "targets_at_boundary": 1}, "no scrape just before the start or the end"),
    ({"targets_at_end": 0}, "no scrape just before the start or the end"),
    ({"targets_always_up": 0}, "a vLLM target unreachable during the window"),
    ({"targets_always_up": None}, "a vLLM target unreachable during the window"),
    ({"resets": 1}, "vLLM restarted during the window"),
    ({"scrapes": 15}, "missing scrapes (15 for ≈ 20 expected"),
    ({"scrapes": None}, "missing scrapes (0 for ≈ 20 expected"),
])
def test_discard_reason(changes, fragment):
    o = window(**changes)
    reason = query.discard_reason(o, 300, 15, [])
    assert reason is not None and fragment in reason


def test_discard_reason_scrape_threshold_is_80_percent_of_the_expected_count():
    assert query.discard_reason(window(scrapes=16), 300, 15, []) is None
    assert "missing scrapes" in query.discard_reason(window(scrapes=15.9), 300, 15, [])


def test_discard_reason_expected_scrapes_grow_with_the_targets():
    o = window(targets=2, targets_at_start=2, targets_at_boundary=2, targets_at_end=2, targets_always_up=2, scrapes=20)
    assert "40 expected" in query.discard_reason(o, 300, 15, [])


def test_discard_reason_excluded_ranges_overlap_rule():
    o = window(DAY, 300)
    excluded = [(DAY + 100, DAY + 200)]
    assert query.discard_reason(o, 300, 15, excluded) == "excluded range (stress test)"
    assert query.discard_reason(o, 300, 15, [(DAY - 50, DAY + 1)]) == "excluded range (stress test)"
    assert query.discard_reason(o, 300, 15, [(DAY + 299, DAY + 400)]) == "excluded range (stress test)"
    assert query.discard_reason(o, 300, 15, [(DAY + 300, DAY + 400)]) is None     # starts exactly at the window end
    assert query.discard_reason(o, 300, 15, [(DAY - 100, DAY)]) is None           # ends exactly at the window start


def test_discard_reason_other_reasons_win_over_the_exclusion():
    assert "vLLM restarted" in query.discard_reason(window(resets=2), 300, 15, [(DAY, DAY + 300)])


# ---------------------------------------------------------------- rows and alerts
def build(obs, sel_l=None, excluded=(), **args):
    return query.build_rows(obs, arguments(**args), sel_l, 300, 15, list(excluded))


def test_build_rows_usable_window(utc_timezone):
    rows, alerts, valid = build([window()])
    (r,) = rows
    assert alerts == [] and len(valid) == 1
    assert (r["start"], r["end"], r["duration_s"], r["targets"]) == ("2024-05-14 00:00", "00:05", 300, 1)
    assert r["verdict"] == "holds" and r["discarded"] is False and r["completed_requests"] == 100


def test_build_rows_discarded_window_blanks_the_numbers(utc_timezone):
    rows, alerts, valid = build([window(resets=1)])
    (r,) = rows
    assert valid == [] and r["verdict"] == "discarded" and r["discarded"] is True
    assert r["completed_requests"] is None and r["output_throughput_per_gpu"] is None and r["mean_concurrency"] is None
    assert alerts == ["2024-05-14 00:00: vLLM restarted during the window (counters reset to zero): window discarded."]


def test_build_rows_no_target_is_silent(utc_timezone):
    rows, alerts, valid = build([window(targets=0)])
    assert rows[0]["verdict"] == "discarded" and alerts == [] and valid == []


def test_build_rows_excluded_window_is_a_run_in_progress(utc_timezone):
    rows, alerts, valid = build([window(DAY), window(DAY + 300)], excluded=[(DAY + 100, DAY + 200)])
    assert [r["verdict"] for r in rows] == ["run in progress", "holds"] and len(valid) == 1
    assert alerts == ["2024-05-14 00:00: excluded range (stress test): window discarded."]


def test_build_rows_few_requests_alert_but_not_for_zero(utc_timezone):
    _, alerts, valid = build([window(requests=49), window(DAY + 300, requests=0)])
    assert alerts == ["2024-05-14 00:00: fewer than 50 completed requests: percentiles unreliable."] and len(valid) == 2
    assert build([window(requests=50)])[1] == []


def test_build_rows_inconsistent_histogram_alert(utc_timezone):
    o = window()
    o["histograms"]["ttft"] = {"0.1": 90.0, "0.5": 10.0, "+Inf": 100.0}
    rows, alerts, _ = build([o])
    assert rows[0]["verdict"] == "undetermined"
    assert alerts == ["2024-05-14 00:00: inconsistent histogram (non-increasing buckets): verdict undetermined."]


def test_build_rows_scrape_longer_than_the_actual_interval(utc_timezone):
    _, alerts, valid = build([window(scrapes=40)])        # 40 scrapes in 300 s: every 7.5 s, not 15 s
    assert len(alerts) == 1 and "--scrape (15 s) is longer than the actual Prometheus interval (≈ 8 s)" in alerts[0]
    assert "give --scrape 8s" in alerts[0] and len(valid) == 1
    assert build([window(scrapes=24)])[1] == []           # up to 20 % more scrapes than expected is accepted


def test_build_rows_litellm_alerts_only_with_a_litellm_selector(utc_timezone):
    o = window(litellm_targets=(1, 1), litellm_resets=1, agents={})
    assert build([o])[1] == []
    _, alerts, _ = build([o], sel_l='job="litellm"')
    assert alerts == ["2024-05-14 00:00: LiteLLM restarted during the window: discarded from the per-agent view."]


def test_check_litellm_target_unreachable_drops_the_agents():
    o = window(litellm_targets=(2, 1), agents={"a": agent(5)})
    alerts = query.check_litellm(o, "T", 'job="l"')
    assert o["agents"] is None and len(alerts) == 1 and "a LiteLLM target unreachable" in alerts[0]


def test_check_litellm_restart_drops_the_agents():
    o = window(litellm_targets=(1, 1), litellm_resets=2, agents={"a": agent(5)})
    alerts = query.check_litellm(o, "T", 'job="l"')
    assert o["agents"] is None and alerts == ["T: LiteLLM restarted during the window: discarded from the per-agent view."]


def test_check_litellm_inconsistent_histogram_drops_the_agents():
    bad = agent(5)
    bad["latency"] = {"1": 5.0, "5": 2.0, "+Inf": 5.0}
    o = window(litellm_targets=(1, 1), agents={"a": bad})
    alerts = query.check_litellm(o, "T", 'job="l"')
    assert o["agents"] is None and "inconsistent LiteLLM histogram" in alerts[0]


def test_check_litellm_no_target_scraped_while_vllm_is():
    o = window(litellm_targets=(None, None), agents={})
    alerts = query.check_litellm(o, "T", 'job="litellm"')
    assert alerts == ['T: no LiteLLM target scraped (selector job="litellm").']
    assert query.check_litellm(window(targets=0, litellm_targets=(None, None), agents={}), "T", "x") == []


def test_check_litellm_healthy_window_is_untouched():
    o = window(litellm_targets=(1, 1), litellm_resets=0, agents={"a": agent(5)})
    assert query.check_litellm(o, "T", 'job="l"') == [] and list(o["agents"]) == ["a"]


# ---------------------------------------------------------------- views
def windows_output(rows, capsys, target=30.0):
    query.windows_view(rows, arguments(target=target))
    return capsys.readouterr().out


def rows_of(*specs):
    """Rows with the keys windows_view reads: specs = (verdict, mean_concurrency, extra changes)."""
    return [row(start=f"T{i}", end="00:05", verdict=v, mean_concurrency=c, max_waiting_requests=0.0, discarded=False, **kw)
            for i, (v, c, kw) in enumerate(specs)]


def test_windows_view_highest_concurrency_that_holds_and_warnings(capsys):
    out = windows_output(rows_of(("holds", 4.0, {}), ("holds", 8.0, {}), ("does not hold", 6.0, {}),
                                 ("undetermined", 10.0, {}), ("does not hold", 12.0, {})) , capsys)
    assert "Highest mean concurrency that holds: 8.0 requests in flight (T1, 20.0 tokens/s per GPU)." in out
    assert "windows with equal or lower concurrency do not hold (T2)" in out
    assert "1 busier window(s) with an undetermined verdict" in out
    assert "no busier window fails" not in out


def test_windows_view_no_busier_window_fails(capsys):
    out = windows_output(rows_of(("holds", 4.0, {}), ("holds", 8.0, {})), capsys)
    assert "WARNING: no busier window fails: the limit is not reached in this range." in out


def test_windows_view_nothing_holds(capsys):
    out = windows_output(rows_of(("does not hold", 4.0, {}), ("holds", 3.0, {"completed_requests": 10.0})), capsys, target=25)
    assert "No window of at least 50 requests holds 25 tokens/s for 90 % of the requests" in out
    assert "Highest mean concurrency" not in out


def test_windows_view_queue_and_kv_warnings_and_interpolation_note(capsys):
    rows = rows_of(("holds", 4.0, {"kv_cache_max_fraction": 0.96, "share_exact": False}))
    rows[0]["max_waiting_requests"] = 3.0
    out = windows_output(rows, capsys)
    assert "requests waited in the server queue in windows that hold" in out
    assert "KV cache 95 % full or more in at least one window" in out
    assert "* interpolated share: no histogram bound at 33.3 ms" in out


def test_windows_view_ignores_the_kv_of_discarded_windows(capsys):
    rows = rows_of(("discarded", None, {"kv_cache_max_fraction": 0.99}))
    rows[0]["discarded"] = True
    assert "KV cache 95 %" not in windows_output(rows, capsys)


def test_period_view_groups_windows_by_period_and_sorts_the_labels(capsys, utc_timezone):
    valid = [window(DAY + 13 * 3600), window(DAY + 2 * 3600, requests=10), window(DAY + 3 * 3600, requests=30)]
    query.period_view(valid, arguments())
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("period") and lines[1].startswith("afternoon (12-18 h)") and lines[2].startswith("night (0-6 h)")
    assert lines[1].split()[3] == "1" and lines[2].split()[3] == "2"            # windows per period
    assert lines[2].split()[5] == "40"                                          # requests merged: 10 + 30


def test_period_view_by_hour(capsys, utc_timezone):
    query.period_view([window(DAY + 13 * 3600), window(DAY + 13 * 3600 + 900), window(DAY + 14 * 3600)], arguments(period="hour"))
    labels = [x[:4] for x in capsys.readouterr().out.splitlines()[1:3]]
    assert labels == ["13 h", "14 h"]


def test_agents_view_adds_the_windows_and_names_the_missing_identity(capsys):
    valid = [window(agents={"coder": agent(30), "": agent(2)}), window(DAY + 300, agents={"coder": agent(30), "reviewer": agent(10, 100.0, False)}),
             window(DAY + 600, agents=None), window(DAY + 900)]
    query.agents_view(valid, arguments(), 'job="litellm"')
    out = capsys.readouterr().out
    lines = {l.split()[0] + (" id" if l.startswith("  (no") else ""): l for l in out.splitlines() if l.startswith("  ")}
    assert "coder" in lines and "reviewer" in lines
    coder = lines["coder"].split()
    assert coder[:3] == ["coder", "60", "93%"]                                 # 60 requests; 24000 of 25800 output tokens
    assert "(no identity)" in out and "requests without a value for the label metadata_agent" in out
    assert lines["reviewer"].split()[-1] == "—"                                # not every request streamed: no mean speed


def test_agents_view_without_any_agent(capsys):
    query.agents_view([window(agents={}), window(agents=None)], arguments(by="api_key_alias"), 'job="l"')
    out = capsys.readouterr().out
    assert out.startswith('No per-agent request in the range (LiteLLM {job="l"}, label api_key_alias).')
    assert "Per-agent view read from LiteLLM" in out


# ---------------------------------------------------------------- command line
def run_main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["observation.query", *argv])
    query.main()


def common(prom_url, *extra):
    return ["--prometheus", prom_url, "--target", "30", "--gpu", "2", "--start", "2024-05-14 00:00",
            "--end", "2024-05-14 00:15", "--window", "5m", *extra]


@pytest.mark.parametrize("extra,message", [
    (["--window", "1m", "--scrape", "30s"], "--window must be at least 4 x --scrape (120 s)"),
    (["--view", "agents"], "--view agents: give --litellm"),
    (["--vllm", "job=vllm"], "--vllm: label selector expected"),
    (["--litellm", "nope"], "--litellm: label selector expected"),
    (["--litellm", 'job="l"', "--by", "model"], "--by: api_key_alias"),
    (["--window", "soon"], "invalid duration"),
    (["--scrape", "x"], "invalid duration"),
    (["--start", "tomorrow"], "invalid date"),
    (["--end", "later"], "invalid date"),
    (["--exclude", "2024-05-14 00:00"], '--exclude: form "start/end" expected'),
    (["--exclude", "bad/2024-05-14 01:00"], "invalid date"),
])
def test_main_rejects_invalid_arguments_before_any_request(monkeypatch, stub, utc_timezone, extra, message):
    s = stub({})
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *common(s.url), *extra)
    assert message in str(e.value) and s.seen == []


def test_main_refuses_a_short_window_after_asking_prometheus_for_its_interval(monkeypatch, stub, utc_timezone):
    """Without --scrape the interval is read from Prometheus first, so this refusal comes after that one request."""
    s = stub({})
    with pytest.raises(SystemExit, match=r"4 x the scrape interval \(60 s, default: Prometheus did not say\)"):
        run_main(monkeypatch, *common(s.url), "--window", "59s")
    assert {r["path"].split("?")[0] for r in s.seen} <= {"/api/v1/targets", "/api/v1/status/config"}
    assert not [r for r in s.seen if "query_range" in r["path"]]


def test_main_requires_the_target_and_the_gpu(monkeypatch, capsys):
    for argv in (["--prometheus", "http://x", "--gpu", "1"], ["--prometheus", "http://x", "--target", "30"], ["--target", "30", "--gpu", "1"]):
        with pytest.raises(SystemExit) as e:
            run_main(monkeypatch, *argv)
        assert e.value.code == 2
    assert "required" in capsys.readouterr().err


def test_main_unknown_view_or_period_is_an_argparse_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "--prometheus", "http://x", "--target", "30", "--gpu", "1", "--view", "nope")
    assert e.value.code == 2 and "invalid choice" in capsys.readouterr().err


def test_main_no_target_in_prometheus(monkeypatch, prom, utc_timezone):
    nothing = prom(lambda q, s, e, st: [])
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *common(nothing.url))
    assert 'No vLLM target {job="vllm"} in Prometheus over this range' in str(e.value)


def test_main_prometheus_unreachable(monkeypatch, utc_timezone):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *common(f"http://127.0.0.1:{helpers.free_port()}"))
    assert "Prometheus unreachable" in str(e.value)


def test_main_range_shorter_than_one_window(monkeypatch, stub, utc_timezone):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *common(stub({}).url, "--end", "2024-05-14 00:04"))
    assert "Range shorter than one window" in str(e.value)


def canned(prom, windows=None):
    windows = windows or [fixtures.prom_window(300, 100, running_avg=4.0, running_max=6.0)] * 3
    return prom(fixtures.prometheus_responder(DAY + 300, 300, windows))


def test_main_windows_view_header_rows_and_verdict(monkeypatch, prom, capsys, utc_timezone):
    s = canned(prom)
    run_main(monkeypatch, *common(s.url))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f'Prometheus {s.url}, vLLM {{job="vllm"}}; 2024-05-14 00:00 → 2024-05-14 00:15, windows of 300 s'
    assert lines[1] == "Scrape interval: 15 s (default: Prometheus did not say)"
    assert [l[:23] for l in lines[4:7]] == ["2024-05-14 00:00  00:05", "2024-05-14 00:05  00:10", "2024-05-14 00:10  00:15"]
    assert all(l.endswith("  holds") for l in lines[4:7])


def test_main_output_csv_has_one_row_per_window_even_when_discarded(monkeypatch, prom, tmp_path, utc_timezone):
    windows = [fixtures.prom_window(300, 100), fixtures.prom_window(300, 100, resets=1), fixtures.prom_window(300, 100)]
    out = tmp_path / "t.csv"
    run_main(monkeypatch, *common(canned(prom, windows).url, "--output", str(out), "--view", "period"))
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert [r["verdict"] for r in rows] == ["holds", "discarded", "holds"] and [r["discarded"] for r in rows] == ["False", "True", "False"]
    assert rows[0]["start"] == "2024-05-14 00:00" and rows[0]["duration_s"] == "300" and rows[1]["completed_requests"] == ""
    assert list(rows[0])[:4] == ["start", "end", "duration_s", "targets"]


def test_main_prints_alerts_once_each(monkeypatch, prom, capsys, utc_timezone):
    windows = [fixtures.prom_window(300, 10)] * 3
    run_main(monkeypatch, *common(canned(prom, windows).url, "--view", "period"))
    out = capsys.readouterr().out
    assert out.count("WARNING: 2024-05-14 00:00: fewer than 50 completed requests") == 1
    assert out.count("fewer than 50") == 3            # one per window, each different


def test_main_agents_view_with_litellm(monkeypatch, prom, capsys, utc_timezone):
    windows = [fixtures.prom_window(300, 100, agents={"coder": agent(80), "reviewer": agent(20)})] * 3
    run_main(monkeypatch, *common(canned(prom, windows).url, "--litellm", 'job="litellm"', "--view", "agents"))
    out = capsys.readouterr().out
    assert 'LiteLLM {job="litellm"} by metadata_agent' in out
    assert [l.split()[:3] for l in out.splitlines() if l.startswith(("  coder", "  reviewer"))] == [
        ["coder", "240", "80%"], ["reviewer", "60", "20%"]]


def test_main_period_hour_view(monkeypatch, prom, capsys, utc_timezone):
    run_main(monkeypatch, *common(canned(prom).url, "--view", "period", "--period", "hour"))
    assert "00 h" in capsys.readouterr().out


def test_main_exclude_makes_the_overlapping_windows_runs_in_progress(monkeypatch, prom, capsys, utc_timezone):
    run_main(monkeypatch, *common(canned(prom).url, "--exclude", "2024-05-14 00:03/2024-05-14 00:07"))
    out = capsys.readouterr().out
    assert out.count("run in progress") == 2 and "excluded range (stress test): window discarded" in out


def test_main_sends_the_prometheus_key_from_the_environment(monkeypatch, prom, utc_timezone):
    s = canned(prom)
    monkeypatch.setenv("BENCH_PROMETHEUS_KEY", "sk-prom")
    run_main(monkeypatch, *common(s.url))
    assert s.seen and {c["headers"].get("Authorization") for c in s.seen} == {"Bearer sk-prom"}


def test_main_sends_no_key_by_default(monkeypatch, prom, utc_timezone):
    s = canned(prom)
    run_main(monkeypatch, *common(s.url))
    assert all("Authorization" not in c["headers"] for c in s.seen)


def test_main_custom_selector_scrape_and_since(monkeypatch, prom, capsys, utc_timezone):
    s = prom(lambda q, st, e, step: [])
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--prometheus", s.url, "--target", "30", "--gpu", "1", "--vllm", 'job="a",instance=~"b.*"',
                 "--since", "30m", "--window", "10m", "--scrape", "30s")
    assert s.seen and all('{job="a",instance=~"b.*"' in c["query"] for c in s.seen)
    first = s.seen[0]
    assert first["step"] == 600 and first["end"] - first["start"] == 1200      # 30 minutes: three windows


def test_main_end_in_the_future_is_capped_at_now(monkeypatch, prom, utc_timezone):
    s = prom(lambda q, st, e, step: [])
    before = time.time()
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--prometheus", s.url, "--target", "30", "--gpu", "1", "--start", "2024-05-14 00:00",
                 "--end", "2999-01-01 00:00", "--window", "6h")
    end = s.seen[0]["end"]            # the last whole window before now
    assert end <= time.time() + 1 and time.time() - end < 6 * 3600 + 1 and end > before - 6 * 3600 - 1


def test_main_start_without_end_runs_up_to_now(monkeypatch, prom, utc_timezone):
    s = prom(lambda q, st, e, step: [])
    start = time.time() - 3 * 3600
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--prometheus", s.url, "--target", "30", "--gpu", "1", "--start", str(start), "--window", "1h")
    assert s.seen[0]["end"] <= time.time() + 1 and s.seen[0]["end"] - s.seen[0]["start"] >= 1.9 * 3600


# ---------------------------------------------------------------- as a program
def test_module_runs_as_a_program_and_reports_errors_on_stderr(prom, utc_timezone):
    s = canned(prom)
    ok = helpers.run([helpers.sys.executable, "-B", "-m", "observation.query", *common(s.url), "--view", "period"],
                     env=helpers.clean_environment(TZ="UTC"))
    assert ok.returncode == 0 and "night (0-6 h)" in ok.stdout and ok.stderr == ""
    bad = helpers.run([helpers.sys.executable, "-B", "-m", "observation.query", *common(s.url), "--window", "10s"],
                      env=helpers.clean_environment(TZ="UTC"))
    assert bad.returncode == 1 and "--window must be at least 4 x the scrape interval" in bad.stderr and bad.stdout == ""


def test_help_lists_the_options():
    r = helpers.run([helpers.sys.executable, "-B", "-m", "observation.query", "--help"])
    assert r.returncode == 0
    for option in ("--prometheus", "--target", "--gpu", "--vllm", "--litellm", "--by", "--since", "--start", "--end",
                   "--window", "--scrape", "--ttft-max", "--view", "--period", "--exclude", "--output"):
        assert option in r.stdout


# ---------------------------------------------------------------- the scrape interval is read from Prometheus
@pytest.mark.parametrize("text,seconds", [
    ("15s", 15), ("1m", 60), ("1m30s", 90), ("500ms", 0.5), ("2h", 7200), ("1d", 86400), ("1w", 604800), ("0.5s", 0.5),
    ("1h30m", 5400),
])
def test_parse_prometheus_duration(text, seconds):
    assert query.parse_prometheus_duration(text) == seconds


@pytest.mark.parametrize("text", ["", None, "15", "s", "1x", "1m 30s", "-5s", "fifteen", "15S"])
def test_parse_prometheus_duration_refuses_what_prometheus_does_not_write(text):
    assert query.parse_prometheus_duration(text) is None


@pytest.mark.parametrize("selector,expected", [
    ('job="vllm"', True), ('job="litellm"', False), ('job!="litellm"', True), ('job=~"vl.*"', True), ('job=~"vl"', False),
    ('job!~"lite.*"', True), ('job="vllm",instance="h:8000"', True), ('job="vllm",instance="other"', False),
    ('team="x"', False), ('team!="x"', True), ("", True),
])
def test_matches_selector(selector, expected):
    assert query.matches_selector({"job": "vllm", "instance": "h:8000"}, selector) is expected


def test_matches_selector_regex_must_match_the_whole_value():
    assert query.matches_selector({"job": "vllm-2"}, 'job=~"vllm"') is False
    assert query.matches_selector({"job": "vllm-2"}, 'job=~"vllm.*"') is True


def targets(*pairs):
    """The data of /api/v1/targets for (job, scrapeInterval) pairs."""
    return {"activeTargets": [{"scrapePool": job, "labels": {"job": job, "instance": f"{job}:1"}, "scrapeInterval": every,
                               "health": "up"} for job, every in pairs]}


CONFIG = {"yaml": "global:\n  scrape_interval: 20s\n  scrape_timeout: 10s\nscrape_configs:\n- job_name: vllm\n"}


def detect(prom, endpoints, selectors=('job="vllm"', None)):
    s = prom(lambda *a: [], endpoints)
    return query.detect_scrape_interval(query.Prometheus(s.url), list(selectors)), s


def test_the_interval_of_the_matching_targets_is_used(prom):
    (seconds, where), _ = detect(prom, {"/api/v1/targets": targets(("vllm", "5s"), ("litellm", "30s"))})
    assert seconds == 5 and "targets" in where


def test_with_a_litellm_selector_the_longest_interval_wins(prom):
    (seconds, _), _ = detect(prom, {"/api/v1/targets": targets(("vllm", "5s"), ("litellm", "30s"))},
                             ('job="vllm"', 'job="litellm"'))
    assert seconds == 30


def test_sub_second_and_fractional_intervals_round_up_to_whole_seconds(prom):
    assert detect(prom, {"/api/v1/targets": targets(("vllm", "500ms"))})[0][0] == 1
    assert detect(prom, {"/api/v1/targets": targets(("vllm", "1m30s"))})[0][0] == 90
    assert detect(prom, {"/api/v1/targets": targets(("vllm", "2500ms"))})[0][0] == 3


def test_when_no_target_matches_the_global_interval_is_used(prom):
    (seconds, where), _ = detect(prom, {"/api/v1/targets": targets(("other", "5s")), "/api/v1/status/config": CONFIG})
    assert seconds == 20 and "global" in where


def test_without_the_targets_endpoint_the_global_interval_is_used(prom):
    (seconds, where), _ = detect(prom, {"/api/v1/status/config": CONFIG})
    assert seconds == 20 and "global" in where


def test_the_global_interval_is_read_from_the_global_section_only(prom):
    config = {"yaml": "scrape_configs:\n- job_name: x\n  scrape_interval: 1s\nglobal:\n  evaluation_interval: 1m\n"
                      "  scrape_interval: 45s\n"}
    assert detect(prom, {"/api/v1/status/config": config})[0][0] == 45


@pytest.mark.parametrize("endpoints", [
    {}, {"/api/v1/targets": {"activeTargets": []}, "/api/v1/status/config": {"yaml": "nothing useful"}},
    {"/api/v1/targets": {"activeTargets": [{"scrapePool": "vllm", "scrapeInterval": "soon"}]}},
    {"/api/v1/targets": "not an object"},
])
def test_when_prometheus_cannot_say_the_default_is_15_seconds(prom, endpoints):
    (seconds, where), _ = detect(prom, endpoints)
    assert seconds == query.DEFAULT_SCRAPE_S == 15 and "did not say" in where


def test_an_unreachable_prometheus_gives_the_default():
    assert query.detect_scrape_interval(query.Prometheus("http://127.0.0.1:1"), ['job="vllm"']) == (15, "default: Prometheus did not say")


def test_the_prometheus_key_is_sent_to_the_targets_endpoint(prom):
    s = prom(lambda *a: [], {"/api/v1/targets": targets(("vllm", "5s"))})
    query.detect_scrape_interval(query.Prometheus(s.url, "sk-prom"), ['job="vllm"'])
    assert s.gets and all(h.get("Authorization") == "Bearer sk-prom" for _, h in s.gets)


def test_main_reads_the_interval_from_prometheus_and_says_so(monkeypatch, prom, capsys, utc_timezone):
    s = prom(fixtures.prometheus_responder(DAY + 300, 300, [fixtures.prom_window(300, 100)] * 3),
             {"/api/v1/targets": targets(("vllm", "30s"))})
    run_main(monkeypatch, *common(s.url))
    assert "Scrape interval: 30 s (read from Prometheus: the interval of the selected targets)" in capsys.readouterr().out


def test_main_with_scrape_given_does_not_ask_prometheus(monkeypatch, prom, capsys, utc_timezone):
    s = prom(fixtures.prometheus_responder(DAY + 300, 300, [fixtures.prom_window(300, 100)] * 3),
             {"/api/v1/targets": targets(("vllm", "30s"))})
    run_main(monkeypatch, *common(s.url, "--scrape", "10s"))
    assert "Scrape interval: 10 s (given by --scrape)" in capsys.readouterr().out
    assert s.gets == []


def test_main_refuses_a_window_shorter_than_four_detected_scrapes(monkeypatch, prom):
    s = prom(lambda *a: [], {"/api/v1/targets": targets(("vllm", "90s"))})
    with pytest.raises(SystemExit, match=r"4 x the scrape interval \(360 s, read from Prometheus: the interval of the selected"):
        run_main(monkeypatch, *common(s.url))


# ---------------------------------------------------------------- the rows of each view, as data (--view-output)
def read_csv(path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


@pytest.mark.parametrize("label,mode,hour", [
    ("night (0-6 h)", "range", 0), ("morning (6-12 h)", "range", 6), ("afternoon (12-18 h)", "range", 12),
    ("evening (18-24 h)", "range", 18), ("00 h", "hour", 0), ("07 h", "hour", 7), ("23 h", "hour", 23),
])
def test_period_start_hour(label, mode, hour):
    assert query.period_start_hour(label, mode) == hour


def test_period_rows_are_in_the_order_of_the_day_with_the_merged_indicators(utc_timezone):
    valid = [window(DAY + 13 * 3600), window(DAY + 2 * 3600, requests=10), window(DAY + 3 * 3600, requests=30)]
    rows = query.period_rows(valid, arguments())
    assert [r["period"] for r in rows] == ["night (0-6 h)", "afternoon (12-18 h)"]          # not alphabetical
    assert [(r["start_hour"], r["windows"]) for r in rows] == [(0, 2), (12, 1)]
    assert rows[0]["completed_requests"] == 40                                                # 10 + 30 merged
    assert rows[0]["verdict"] == "holds" and "output_throughput_per_gpu" in rows[0]


def test_period_rows_by_hour_and_without_any_usable_window(utc_timezone):
    rows = query.period_rows([window(DAY + 14 * 3600), window(DAY + 13 * 3600), window(DAY + 13 * 3600 + 900)], arguments(period="hour"))
    assert [(r["period"], r["windows"]) for r in rows] == [("13 h", 2), ("14 h", 1)]
    assert query.period_rows([], arguments()) == []


def test_period_rows_have_the_columns_of_the_windows_plus_three(utc_timezone):
    valid = [window(DAY + 3 * 3600)]
    window_columns = set(query.evaluate(valid[0], arguments()))
    assert set(query.period_rows(valid, arguments())[0]) == window_columns | {"period", "start_hour", "windows"}


def test_agent_rows_add_the_windows_and_name_the_missing_identity():
    valid = [window(agents={"coder": agent(30), "": agent(2)}),
             window(DAY + 300, agents={"coder": agent(30), "reviewer": agent(10, 100.0, False)}), window(DAY + 600, agents=None)]
    rows = {r["agent"]: r for r in query.agent_rows(valid)}
    assert set(rows) == {"coder", "reviewer", ""}
    assert rows["coder"]["requests"] == 60 and rows["coder"]["agent_name"] == "coder"
    assert rows[""]["agent_name"] == "(no identity)"
    assert rows["reviewer"]["mean_speed"] is None                                              # not every request streamed
    assert round(sum(r["output_token_share"] for r in rows.values()), 6) == 1.0
    assert query.agent_rows([window(agents={}), window(agents=None)]) == []


def test_the_text_views_are_unchanged_by_the_row_helpers(capsys, utc_timezone):
    """period_view and agents_view print what they printed before (the golden tests hold the full text)."""
    valid = [window(DAY + 13 * 3600), window(DAY + 2 * 3600, requests=10)]
    query.period_view(valid, arguments())
    assert [l.split()[0] for l in capsys.readouterr().out.splitlines()[1:3]] == ["afternoon", "night"]


def test_main_view_output_for_the_windows_view_is_the_windows(monkeypatch, prom, tmp_path, utc_timezone):
    out, view = tmp_path / "w.csv", tmp_path / "v.csv"
    run_main(monkeypatch, *common(canned(prom).url, "--output", str(out), "--view-output", str(view)))
    assert read_csv(view) == read_csv(out) and len(read_csv(view)) == 3


def test_main_view_output_for_the_period_view_is_one_row_per_period(monkeypatch, prom, tmp_path, utc_timezone):
    out, view = tmp_path / "w.csv", tmp_path / "v.csv"
    run_main(monkeypatch, *common(canned(prom).url, "--view", "period", "--output", str(out), "--view-output", str(view)))
    rows = read_csv(view)
    assert [(r["period"], r["start_hour"], r["windows"]) for r in rows] == [("night (0-6 h)", "0", "3")]
    assert len(read_csv(out)) == 3                                                            # --output still has the windows


def test_main_view_output_for_the_agents_view_is_one_row_per_agent(monkeypatch, prom, tmp_path, utc_timezone):
    view = tmp_path / "v.csv"
    windows = [fixtures.prom_window(300, 100, agents={"coder": agent(80), "reviewer": agent(20)})] * 3
    run_main(monkeypatch, *common(canned(prom, windows).url, "--litellm", 'job="litellm"', "--view", "agents", "--view-output", str(view)))
    rows = read_csv(view)
    assert [(r["agent"], r["requests"]) for r in rows] == [("coder", "240.0"), ("reviewer", "60.0")]
    assert rows[0]["output_token_share"].startswith("0.8")


def test_main_without_view_output_writes_no_view_file(monkeypatch, prom, tmp_path, utc_timezone):
    run_main(monkeypatch, *common(canned(prom).url, "--view", "period"))
    assert list(tmp_path.iterdir()) == []


def test_window_rows_say_why_a_window_was_discarded(monkeypatch, prom, tmp_path, utc_timezone):
    out = tmp_path / "w.csv"
    run_main(monkeypatch, *common(canned(prom).url, "--exclude", "2024-05-14 00:03/2024-05-14 00:07", "--output", str(out)))
    rows = read_csv(out)
    assert [r["discard_reason"] for r in rows] == ["excluded range (stress test)", "excluded range (stress test)", ""]
    assert [r["discarded"] for r in rows] == ["True", "True", "False"]
