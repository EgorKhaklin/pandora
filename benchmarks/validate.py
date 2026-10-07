"""Validation batch 1: strong baselines with confidence intervals, ablations, scaling, harder matrices,
and the mechanism measurement. Fresh seeds (70_000+).

    python -m benchmarks.validate baselines|ablations|scaling|matrices|mechanism
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz

from pandora import basis_pursuit, iterative_support_detection, pandora
from pandora.core import certify
from .greedy import cosamp, iht, omp, subspace_pursuit
from .run import matrix as base_matrix

RES = Path(__file__).resolve().parent.parent / "results"


def wilson(h, n, z=1.96):
    p = h / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * max(0, c - half), 1), round(100 * min(1, c + half), 1)]


def matrix(kind, n, d, rng):
    if kind == "toeplitz":
        return toeplitz(rng.normal(size=n), rng.normal(size=d))
    if kind == "ill-conditioned":  # singular values from 1 down to 1e-2
        U, _ = np.linalg.qr(rng.normal(size=(n, n)))
        V, _ = np.linalg.qr(rng.normal(size=(d, n)))
        return (U * np.logspace(0, -2, n)) @ V.T * np.sqrt(d)
    return base_matrix(kind, n, d, rng)


def problem(kind, n, d, k, seed):
    rng = np.random.default_rng(seed)
    X = matrix(kind, n, d, rng)
    w = np.zeros(d)
    S = rng.choice(d, k, replace=False)
    w[S] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
    return X, X @ w, w


def exact(w, wt):
    return w is not None and np.linalg.norm(w - wt) <= 1e-2 * np.linalg.norm(wt)


def run_methods(methods, cells, trials, seed0, label):
    out = []
    for kind, n, d, k in cells:
        t0 = time.time()
        hits = {m: 0 for m in methods}
        for i in range(trials):
            X, y, wt = problem(kind, n, d, k, seed0 + 1000 * k + i)
            for m, f in methods.items():
                hits[m] += exact(f(X, y, k, seed0 + i), wt)
        row = {"matrix": kind, "n": n, "d": d, "k": k, "trials": trials,
               **{m: int(h) for m, h in hits.items()},
               "ci95": {m: wilson(h, trials) for m, h in hits.items()}}
        out.append(row)
        print(f"{label} {kind} n={n} d={d} k={k}: " + " | ".join(
            f"{m} {h}/{trials} {row['ci95'][m]}" for m, h in hits.items()) +
            f" ({time.time() - t0:.0f}s)", flush=True)
    return out


BASE = {"L1": lambda X, y, k, s: basis_pursuit(X, y)[0],
        "ISD": lambda X, y, k, s: iterative_support_detection(X, y),
        "OMP (given k)": lambda X, y, k, s: omp(X, y, k),
        "CoSaMP (given k)": lambda X, y, k, s: cosamp(X, y, k),
        "Subspace Pursuit (given k)": lambda X, y, k, s: subspace_pursuit(X, y, k),
        "IHT (given k)": lambda X, y, k, s: iht(X, y, k),
        "Pandora": lambda X, y, k, s: pandora(X, y, rng=s)}


def pandora_variant(X, y, rng, top, rest, decay=0.5, rounds=200, biased=True, random=True):
    n, d = X.shape
    rng = np.random.default_rng(rng)
    score = np.abs(basis_pursuit(X, y)[0])
    w = certify(X, y, score)
    r = 0
    while w is None and r < rounds:
        r += 1
        order = np.argsort(-score)
        T = order[:top] if biased else np.array([], int)
        pool = np.setdiff1d(np.arange(d), T)
        if random:
            R = rng.choice(pool, top + rest - len(T), replace=False)
        else:  # deterministic: the next-best columns, sliding window
            start = (top + (r - 1) * rest) % max(1, d - top)
            R = order[top:][np.arange(start, start + rest) % (d - top)]
        score = decay * score + np.abs(basis_pursuit(X, y, np.concatenate([T, R]))[0])
        if r % 5 == 0 or r == rounds:
            w = certify(X, y, score)
    return w


def main(part):
    RES.mkdir(exist_ok=True)
    if part == "baselines":
        cells = [("gaussian", 40, 200, k) for k in (8, 10, 12, 14)] + \
                [("gaussian", 100, 500, k) for k in (25, 30, 35)]
        rows = run_methods(BASE, cells, 100, 70_000, "baselines")
    elif part == "ablations":
        n = 40
        m = {"Pandora (default)": lambda X, y, k, s: pandora(X, y, rng=s),
             "no randomness (sliding window)": lambda X, y, k, s: pandora_variant(X, y, s, 20, 40, random=False),
             "no bias (vessels all random)": lambda X, y, k, s: pandora_variant(X, y, s, 20, 40, biased=False),
             "no forgetting (decay 1.0)": lambda X, y, k, s: pandora_variant(X, y, s, 20, 40, decay=1.0)}
        for size in (45, 50, 60, 80, 120, 200):
            m[f"vessel {size} columns"] = (lambda size: lambda X, y, k, s: pandora_variant(
                X, y, s, size // 3, size - size // 3))(size)
        rows = run_methods(m, [("gaussian", n, 200, k) for k in (10, 12)], 60, 80_000, "ablations")
    elif part == "scaling":
        rows = []
        for d in (200, 500, 1000, 2000):
            n = d // 5
            for ratio in (0.25, 0.3):
                k = int(round(ratio * n))
                trials = 30 if d <= 500 else 15
                t0 = time.time()
                hits = {"L1": 0, "ISD": 0, "Pandora": 0}
                lps, secs = [], []
                for i in range(trials):
                    X, y, wt = problem("gaussian", n, d, k, 90_000 + d + i)
                    hits["L1"] += exact(basis_pursuit(X, y)[0], wt)
                    hits["ISD"] += exact(iterative_support_detection(X, y), wt)
                    ts = time.time()
                    w, info = pandora(X, y, rng=i, return_info=True)
                    secs.append(time.time() - ts)
                    lps.append(1 + info["rounds"])
                    hits["Pandora"] += exact(w, wt)
                row = {"d": d, "n": n, "k": k, "trials": trials, **hits,
                       "pandora_lps_mean": float(np.mean(lps)),
                       "pandora_seconds_mean": float(np.mean(secs))}
                rows.append(row)
                print(f"scaling d={d} n={n} k={k}: {hits} of {trials}; Pandora {np.mean(lps):.0f} LPs, "
                      f"{np.mean(secs):.1f}s per problem ({time.time() - t0:.0f}s)", flush=True)
    elif part == "matrices":
        m = {k_: BASE[k_] for k_ in ("L1", "ISD", "OMP (given k)", "Pandora")}
        rows = run_methods(m, [(kd, 40, 200, k) for kd in ("toeplitz", "ill-conditioned")
                               for k in (8, 11, 14)], 40, 95_000, "matrices")
    elif part == "mechanism":  # how often the true support is inside the top n/2 of each ranking
        rows = []
        for k in (8, 10, 12, 14):
            inside = {"L1 ranking": 0, "Pandora ranking at the end": 0}
            hits = 0
            for i in range(60):
                X, y, wt = problem("gaussian", 40, 200, k, 99_000 + 1000 * k + i)
                S = set(np.flatnonzero(wt))
                inside["L1 ranking"] += S <= set(np.argsort(-np.abs(basis_pursuit(X, y)[0]))[:20])
                w, info = pandora(X, y, rng=i, return_info=True)
                inside["Pandora ranking at the end"] += S <= set(np.argsort(-info["score"])[:20])
                hits += exact(w, wt)
            rows.append({"k": k, "trials": 60, **inside, "Pandora recovered": hits})
            print(f"mechanism k={k}: support inside top 20 of: {inside}; Pandora recovered {hits}/60",
                  flush=True)
    (RES / f"validate_{part}.json").write_text(json.dumps(rows, indent=1, default=int) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
