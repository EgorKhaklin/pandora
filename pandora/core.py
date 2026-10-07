"""Pandora: certified sparse recovery from many small basis-pursuit solves.

Given y = X w with X of shape (n, d), n < d, and w sparse:

1. score every column by |w| from basis pursuit (min ||w||_1 s.t. X w = y) on all columns;
2. repeat: a vessel is the n/2 best-scored columns plus n random others. Solve basis
   pursuit restricted to the vessel. It has 1.5 n unknowns instead of d, a much easier
   problem. score <- decay * score + |w_vessel| (decay 0.5: recent vessels count most);
3. every few vessels, certify: least squares on the n - 1 best-scored columns. If it fits y
   exactly, the true support lies inside them (for generic X and w, see README) and least
   squares returns w itself. Stop there.

A returned answer is certified. None means no certificate appeared within the budget.
"""

import numpy as np
from scipy.optimize import linprog


def basis_pursuit(X, y, cols=None, weights=None):
    """min sum_i weights_i |w_i| subject to X w = y, using only `cols` (default: all columns).

    Returns (w over all d columns, dual multipliers of X w = y)."""
    d = X.shape[1]
    cols = np.arange(d) if cols is None else np.asarray(cols)
    Xc = X[:, cols]
    c = np.ones(len(cols)) if weights is None else np.asarray(weights, float)[cols]
    r = linprog(np.concatenate([c, c]), A_eq=np.hstack([Xc, -Xc]), b_eq=y, bounds=(0, None),
                method="highs")
    w = np.zeros(d)
    if r.x is not None:
        w[cols] = r.x[:len(cols)] - r.x[len(cols):]
    duals = r.eqlin.marginals if r.eqlin is not None else np.zeros(len(y))
    return w, duals


def certify(X, y, score, tol=1e-8):
    """Least squares on the n - 1 best-scored columns: the solution if it fits y exactly, else None."""
    n = X.shape[0]
    C = np.argsort(-np.asarray(score))[:n - 1]
    c, *_ = np.linalg.lstsq(X[:, C], y, rcond=None)
    if np.linalg.norm(X[:, C] @ c - y) > tol * np.linalg.norm(y):
        return None
    w = np.zeros(X.shape[1])
    w[C] = c
    w[np.abs(w) < 1e-9 * max(1.0, np.abs(c).max())] = 0.0
    return w


def pandora(X, y, rng=None, rounds=200, top=None, rest=None, decay=0.5, check_every=5,
            return_info=False):
    """Certified sparse recovery. Returns w (or None); with return_info, also a dict of
    {'rounds': vessels used, 'certified': bool, 'score': final column scores}.

    top and rest default to n // 2 and n: vessels of 1.5 n columns, which scale with the
    number of measurements."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    rng = np.random.default_rng(rng)
    top = n // 2 if top is None else top
    rest = n if rest is None else rest
    rest = min(rest, d - top)
    score = np.abs(basis_pursuit(X, y)[0])
    w = certify(X, y, score)
    r = 0
    while w is None and r < rounds:
        r += 1
        T = np.argsort(-score)[:top]
        R = rng.choice(np.setdiff1d(np.arange(d), T), rest, replace=False)
        score = decay * score + np.abs(basis_pursuit(X, y, np.concatenate([T, R]))[0])
        if r % check_every == 0 or r == rounds:
            w = certify(X, y, score)
    if return_info:
        return w, {"rounds": r, "certified": w is not None, "score": score}
    return w


def reweighted_l1(X, y, iters=5, eps=0.1):
    """Candes, Wakin and Boyd (2008): basis pursuit reweighted by 1 / (|w| + eps)."""
    w = basis_pursuit(X, y)[0]
    for _ in range(iters - 1):
        w = basis_pursuit(X, y, weights=1 / (np.abs(w) + eps))[0]
    return w


def iterative_support_detection(X, y, iters=8, beta=2.0):
    """Wang and Yin (2010): basis pursuit leaving the detected support unpenalized,
    support = {i : |w_i| > max|w| / beta^(t+1)} from the previous solution."""
    w = basis_pursuit(X, y)[0]
    for t in range(iters):
        detected = np.abs(w) > np.abs(w).max() / beta ** (t + 1)
        w = basis_pursuit(X, y, weights=np.where(detected, 0.0, 1.0))[0]
    return w
