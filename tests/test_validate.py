import numpy as np
import pytest

from benchmarks.validate import exact, iterative_support_detection, problem, restarted_isd, sign_test


def test_restarted_isd_stops_at_the_first_certificate():
    X, y, wt = problem("gaussian", 40, 200, 5, 3)
    w, lps = restarted_isd(X, y, np.random.default_rng(0))
    assert lps == 9 and w == pytest.approx(wt, abs=1e-8)


def test_restarted_isd_respects_its_budget():
    X, y, _ = problem("gaussian", 40, 200, 17, 7)
    assert restarted_isd(X, y, np.random.default_rng(0), budget=5) == (None, 0)
    _, lps = restarted_isd(X, y, np.random.default_rng(0), budget=50)
    assert lps <= 50


def test_restarts_recover_what_one_isd_run_misses():
    """The restarts are the point of the baseline: find a problem plain ISD fails and they solve."""
    for seed in range(5000, 5100):
        X, y, wt = problem("gaussian", 40, 200, 11, seed)
        if not exact(iterative_support_detection(X, y, beta=1.5), wt):
            w, lps = restarted_isd(X, y, np.random.default_rng(seed))
            if exact(w, wt):
                assert lps > 9
                return
    pytest.fail("restarted ISD never recovered a problem single-run ISD missed")


def test_sign_test_on_known_counts():
    assert sign_test([1, 1, 0, 1], [0, 1, 0, 0]) == (2, 0, 0.5)
    assert sign_test([1, 0], [1, 0]) == (0, 0, 1.0)
    only_a, only_b, p = sign_test([True] * 18 + [False], [False] * 18 + [True])
    assert (only_a, only_b) == (18, 1) and p == pytest.approx(2 * 20 / 2 ** 19)
