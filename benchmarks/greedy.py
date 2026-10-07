"""Greedy and thresholding baselines. All of them are given the true sparsity k (Pandora is not)."""

import numpy as np


def _ls(X, y, S):
    w = np.zeros(X.shape[1])
    if len(S):
        w[S] = np.linalg.lstsq(X[:, S], y, rcond=None)[0]
    return w


def omp(X, y, k):
    S, r = [], y.copy()
    for _ in range(k):
        S.append(int(np.argmax(np.abs(X.T @ r))))
        w = _ls(X, y, S)
        r = y - X @ w
    return w


def cosamp(X, y, k, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        r = y - X @ w
        omega = np.argsort(-np.abs(X.T @ r))[:2 * k]
        T = np.union1d(omega, np.flatnonzero(w))
        b = _ls(X, y, T)
        w = np.zeros_like(b)
        keep = np.argsort(-np.abs(b))[:k]
        w[keep] = b[keep]
        if np.linalg.norm(y - X @ w) < 1e-10 * np.linalg.norm(y):
            break
    return w


def subspace_pursuit(X, y, k, iters=50):
    T = np.argsort(-np.abs(X.T @ y))[:k]
    r = y - X @ _ls(X, y, T)
    for _ in range(iters):
        cand = np.union1d(T, np.argsort(-np.abs(X.T @ r))[:k])
        b = _ls(X, y, cand)
        Tn = np.argsort(-np.abs(b))[:k]
        rn = y - X @ _ls(X, y, Tn)
        if np.linalg.norm(rn) >= np.linalg.norm(r):
            break
        T, r = Tn, rn
    return _ls(X, y, T)


def iht(X, y, k, iters=500):
    """Normalized iterative hard thresholding (Blumensath and Davies, 2010), plus a final refit."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        g = X.T @ (y - X @ w)
        S = np.flatnonzero(w) if np.any(w) else np.argsort(-np.abs(g))[:k]
        mu = (g[S] @ g[S]) / max(np.linalg.norm(X[:, S] @ g[S]) ** 2, 1e-30)
        v = w + mu * g
        keep = np.argsort(-np.abs(v))[:k]
        w = np.zeros_like(v)
        w[keep] = v[keep]
    return _ls(X, y, np.flatnonzero(w))
