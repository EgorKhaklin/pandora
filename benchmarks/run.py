"""Benchmarks on fresh problems: the phase diagram and four measurement ensembles.

    python -m benchmarks.run phase       # about 25 minutes
    python -m benchmarks.run ensembles   # about 15 minutes
    python -m benchmarks.figures         # redraws figures/ from results/
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import dct

from pandora import basis_pursuit, iterative_support_detection, pandora

RES = Path(__file__).resolve().parent.parent / "results"


def matrix(kind, n, d, rng):
    if kind == "gaussian":
        return rng.normal(size=(n, d))
    if kind == "bernoulli":
        return rng.choice([-1.0, 1.0], size=(n, d))
    if kind == "partial DCT":
        D = dct(np.eye(d), norm="ortho", axis=0)
        return D[rng.choice(d, n, replace=False)] * np.sqrt(d)
    if kind == "correlated":  # neighbouring columns correlated 0.5
        Z = rng.normal(size=(n, d))
        X = np.empty_like(Z)
        X[:, 0] = Z[:, 0]
        for j in range(1, d):
            X[:, j] = 0.5 * X[:, j - 1] + np.sqrt(0.75) * Z[:, j]
        return X
    raise ValueError(kind)


def problem(kind, n, d, k, seed):
    rng = np.random.default_rng(seed)
    X = matrix(kind, n, d, rng)
    w = np.zeros(d)
    S = rng.choice(d, k, replace=False)
    w[S] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
    return X, X @ w, w


def exact(w, wt):
    return w is not None and np.linalg.norm(w - wt) <= 1e-2 * np.linalg.norm(wt)


def cell(kind, n, d, k, trials, seed0):
    hits = {"L1": 0, "ISD": 0, "Pandora": 0}
    for i in range(trials):
        X, y, wt = problem(kind, n, d, k, seed0 + i)
        hits["L1"] += exact(basis_pursuit(X, y)[0], wt)
        hits["ISD"] += exact(iterative_support_detection(X, y), wt)
        hits["Pandora"] += exact(pandora(X, y, rng=seed0 + i), wt)
    return {k_: int(v) for k_, v in hits.items()}


def main(part):
    RES.mkdir(exist_ok=True)
    rows = []
    if part == "phase":
        for n in (20, 40, 60, 100):
            for ratio in (0.15, 0.25, 0.35, 0.45):
                k = max(1, int(round(ratio * n)))
                t0 = time.time()
                h = cell("gaussian", n, 200, k, 30, 50_000 + 1000 * n + k)
                rows.append({"n": n, "k": k, "trials": 30, **h})
                print(f"n={n:3d} k={k:2d}: {h} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for kind in ("gaussian", "bernoulli", "partial DCT", "correlated"):
            for k in (8, 11, 14):
                t0 = time.time()
                h = cell(kind, 40, 200, k, 40, 60_000 + 100 * k)
                rows.append({"matrix": kind, "n": 40, "k": k, "trials": 40, **h})
                print(f"{kind:12s} k={k:2d}: {h} ({time.time() - t0:.0f}s)", flush=True)
    (RES / f"{part}.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase")
