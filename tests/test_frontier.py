import numpy as np
import pytest

from benchmarks.frontier import amp, certified, denoise, irls_lp, sbl
from benchmarks.validate import exact, problem
from pandora.core import certify


@pytest.mark.parametrize("tau,eps", [(0.3, 0.05), (1.0, 0.2), (0.05, 0.1)])
def test_denoiser_matches_direct_integration(tau, eps):
    """Posterior mean and variance against brute-force quadrature over the prior."""
    r = np.array([-2.7, -1.4, -0.3, 0.0, 0.6, 1.2, 1.9, 3.1])
    w = np.linspace(1, 2, 20001)
    lik = lambda v: np.exp(-0.5 * ((r[:, None] - v) / tau) ** 2)
    z0 = (1 - eps) * lik(np.zeros(1))[:, 0]
    zp, zn = lik(w), lik(-w)
    mass = np.trapezoid(zp, w, axis=1) + np.trapezoid(zn, w, axis=1)
    first = np.trapezoid(zp * w, w, axis=1) - np.trapezoid(zn * w, w, axis=1)
    second = np.trapezoid((zp + zn) * w ** 2, w, axis=1)
    Z = z0 + eps / 2 * mass
    mean_q = eps / 2 * first / Z
    var_q = eps / 2 * second / Z - mean_q ** 2
    mean, var = denoise(r, tau, eps)
    assert mean == pytest.approx(mean_q, abs=1e-6)
    assert var == pytest.approx(var_q, abs=1e-6)


def test_amp_recovers_an_easy_problem():
    X, y, wt = problem("gaussian", 40, 200, 5, 11)
    x, _ = amp(X, y, 5 / 200)
    assert exact(certify(X, y, np.abs(x)), wt)


@pytest.mark.parametrize("p", [0.5, 0.1])
def test_irls_lp_recovers_an_easy_problem(p):
    X, y, wt = problem("gaussian", 40, 200, 5, 12)
    assert exact(certify(X, y, np.abs(irls_lp(X, y, p))), wt)


def test_sbl_recovers_an_easy_problem():
    X, y, wt = problem("gaussian", 40, 200, 5, 13)
    assert exact(certify(X, y, np.abs(sbl(X, y))), wt)


def test_certified_wrapper_stops_at_the_first_certificate():
    X, y, wt = problem("gaussian", 40, 200, 5, 14)
    calls = []

    def run(g):
        calls.append(g)
        return wt if len(calls) == 3 else np.ones(200)

    w, used = certified(X, y, run, 10, np.random.default_rng(0))
    assert used == 3 and len(calls) == 3 and exact(w, wt)
