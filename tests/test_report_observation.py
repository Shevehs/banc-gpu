import time

from report.observation import merge, overlaps, period_label, sum_histograms


def window(start, end, counters=None, ttft=None, kv=None):
    return {"start": start, "end": end, "counters": counters or {},
            "histograms": {"ttft": ttft or {}}, "gauges": {"kv": kv}}


def test_sum_histograms():
    assert sum_histograms([{"0.1": 1.0, "+Inf": 2.0}, {"0.1": 3.0}]) == {"0.1": 4.0, "+Inf": 2.0}


def test_merge_adds_counters_and_takes_the_max_of_gauges():
    a = window(0, 10, {"output_tokens": 5.0}, {"1": 1.0}, kv=0.3)
    b = window(10, 30, {"output_tokens": 7.0}, {"1": 2.0}, kv=0.9)
    m = merge([a, b])
    assert m["counters"]["output_tokens"] == 12.0
    assert m["histograms"]["ttft"] == {"1": 3.0}
    assert m["gauges"]["kv"] == 0.9
    assert m["end"] == 30


def test_merge_unreadable_counter_makes_the_sum_unknown():
    m = merge([window(0, 1, {"output_tokens": None}), window(1, 2, {"output_tokens": 4.0})])
    assert m["counters"]["output_tokens"] is None


def test_merge_errors_none_means_nothing_to_add():
    m = merge([window(0, 1, {"failed_requests": None}), window(1, 2, {"failed_requests": 2.0})])
    assert m["counters"]["failed_requests"] == 2.0
    n = merge([window(0, 1, {"failed_requests": None}), window(1, 2, {"failed_requests": None})])
    assert n["counters"]["failed_requests"] is None


def test_overlaps():
    assert not overlaps([window(0, 10), window(10, 20)])
    assert overlaps([window(0, 10), window(5, 20)])
    assert overlaps([window(5, 20), window(0, 10)])  # order does not matter


def test_period_by_hour():
    t = time.mktime((2026, 1, 5, 14, 30, 0, 0, 0, -1))
    assert period_label(t, "hour") == "14 h"
