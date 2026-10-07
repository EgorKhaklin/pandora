import numpy as np
import pytest

from pandora import (atlas, basis_pursuit, certify, iterative_support_detection, pandora,
                     reweighted_l1)


def problem(seed, n=40, d=200, k=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    S = rng.choice(d, k, replace=False)
    w[S] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
    return X, X @ w, w


def test_certificate_accepts_a_superset_and_returns_the_truth():
    X, y, wt = problem(0, k=11)
    score = np.abs(wt) + 1e-3 * np.random.default_rng(0).random(200)
    assert certify(X, y, score) == pytest.approx(wt, abs=1e-8)


def test_certificate_refuses_a_set_missing_one_true_column():
    X, y, wt = problem(0, k=11)
    score = np.abs(wt) + 1e-3 * np.random.default_rng(0).random(200)
    score[np.flatnonzero(wt)[0]] = -1.0
    assert certify(X, y, score) is None


def test_restricted_basis_pursuit_stays_on_its_columns():
    X, y, _ = problem(1)
    w, _ = basis_pursuit(X, y, np.arange(100))
    assert np.all(w[100:] == 0) and np.linalg.norm(X @ w - y) < 1e-8 * np.linalg.norm(y)


@pytest.mark.parametrize("n", [20, 40, 60])
def test_pandora_recovers_easy_problems_at_several_sizes(n):
    X, y, wt = problem(2 + n, n=n, k=n // 8)
    w, info = pandora(X, y, rng=0, rounds=30, return_info=True)
    assert info["certified"] and w == pytest.approx(wt, abs=1e-6)


def test_pandora_beyond_l1_on_a_known_case():
    """A k = 11 problem where basis pursuit fails and Pandora certifies the truth."""
    for seed in range(1000, 1100):
        X, y, wt = problem(seed, k=11)
        if np.linalg.norm(basis_pursuit(X, y)[0] - wt) > 1e-2 * np.linalg.norm(wt):
            w = pandora(X, y, rng=seed)
            if w is not None:
                assert w == pytest.approx(wt, abs=1e-6)
                return
    pytest.fail("no problem where Pandora recovers what L1 misses")


def test_baselines_recover_an_easy_problem():
    X, y, wt = problem(3)
    assert iterative_support_detection(X, y) == pytest.approx(wt, abs=1e-6)
    assert reweighted_l1(X, y) == pytest.approx(wt, abs=1e-6)


def test_atlas_recovers_under_small_noise():
    X, y0, wt = problem(4, k=5)
    y = y0 + 0.01 * np.random.default_rng(4).normal(size=len(y0))
    w = atlas(X, y, 0.01, rng=4, rounds=20)
    assert w is not None and np.linalg.norm(w - wt) / np.linalg.norm(wt) < 0.05
