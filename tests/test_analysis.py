from observation.analysis import quantile, share_below, summarize, verdict


def test_summarize_sums_counters_histograms_and_gauges():
    series = [
        ("vllm:generation_tokens", {"engine": "0"}, 10.0),
        ("vllm:generation_tokens", {"engine": "1"}, 5.0),
        ("vllm:time_to_first_token_seconds_bucket", {"le": "0.5"}, 3.0),
        ("vllm:time_to_first_token_seconds_bucket", {"le": "0.5", "engine": "1"}, 2.0),
        ("vllm:kv_cache_usage_perc", {}, 0.4),
        ("vllm:kv_cache_usage_perc", {}, 0.7),
    ]
    c, h, g = summarize(series)
    assert c["vllm:generation_tokens"] == 15.0
    assert h["ttft"]["0.5"] == 5.0
    assert g["kv"] == [0.4, 0.7]


def test_http_errors_only_4xx_5xx_on_v1():
    series = [
        ("http_requests", {"handler": "/v1/chat/completions", "status": "2xx"}, 100.0),
        ("http_requests", {"handler": "/v1/chat/completions", "status": "5xx"}, 3.0),
        ("http_requests", {"handler": "/health", "status": "5xx"}, 9.0),
    ]
    c, _, _ = summarize(series)
    assert c["http_requests"] == 3.0


def test_http_errors_zero_as_soon_as_a_v1_series_exists():
    c, _, _ = summarize([("http_requests", {"handler": "/v1/models", "status": "2xx"}, 4.0)])
    assert c["http_requests"] == 0.0


def test_quantile_upper_bound_of_the_bucket():
    h = {"0.1": 50.0, "0.5": 90.0, "+Inf": 100.0}
    assert quantile(h, 0.5) == (0.1, False)
    assert quantile(h, 0.9) == (0.5, False)


def test_quantile_beyond_the_last_finite_bound():
    h = {"0.1": 10.0, "0.5": 20.0, "+Inf": 100.0}
    assert quantile(h, 0.95) == (0.5, True)


def test_quantile_empty_histogram():
    assert quantile({}, 0.5) == (None, False)
    assert quantile({"1": 0.0}, 0.5) == (None, False)


def test_share_below_exact_bound():
    h = {"0.1": 25.0, "0.5": 80.0, "+Inf": 100.0}
    est, exact, low, high = share_below(h, 0.5)
    assert (est, exact) == (0.8, True) and low == high == 0.8


def test_share_below_interpolated_and_bracketed():
    h = {"0.1": 20.0, "0.5": 80.0, "+Inf": 100.0}
    est, exact, low, high = share_below(h, 0.3)
    assert exact is False
    assert low == 0.2 and high == 0.8 and low < est < high


def test_share_below_without_observations():
    assert share_below({}, 0.5) == (None, None, None, None)


def test_verdict():
    assert verdict(True) == "holds" and verdict(False) == "does not hold" and verdict(None) == "undetermined"
