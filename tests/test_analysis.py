"""observation/analysis.py: quantiles, share below a bound and the verdict text."""
from observation.analysis import quantile, share_below, verdict


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
