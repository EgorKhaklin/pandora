"""Parallel speedup: B vessels per round from the same scores, solved concurrently, then merged.
Same total vessel budget as serial Pandora. Run on an otherwise idle machine.

    python -m benchmarks.speedup
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pandora import basis_pursuit, pandora
from pandora.core import certify
from .validate import exact, problem

RES = Path(__file__).resolve().parent.parent / "results"


def _vessel(args):
    X, y, cols = args
    return cols, np.abs(basis_pursuit(X, y, cols)[0])


def pandora_batched(X, y, rng, pool, batch=8, rounds=25, decay=0.5):
    n, d = X.shape
    rng = np.random.default_rng(rng)
    top, rest = n // 2, n
    score = np.abs(basis_pursuit(X, y)[0])
    w = certify(X, y, score)
    for r in range(rounds):
        if w is not None:
            break
        T = np.argsort(-score)[:top]
        pool_cols = np.setdiff1d(np.arange(d), T)
        jobs = [(X, y, np.concatenate([T, rng.choice(pool_cols, rest, replace=False)])) for _ in range(batch)]
        merged = np.zeros(d)
        for _, a in pool.map(_vessel, jobs):
            merged += a
        score = decay * score + merged / batch
        w = certify(X, y, score)
    return w


def main():
    probs = [problem("gaussian", 40, 200, 11, 130_000 + i) for i in range(30)]
    t0 = time.time()
    serial = sum(exact(pandora(X, y, rng=i), wt) for i, (X, y, wt) in enumerate(probs))
    t_serial = time.time() - t0
    rows = {"serial": {"recovered": int(serial), "seconds": t_serial}}
    for workers in (2, 4, 8):
        with ProcessPoolExecutor(max_workers=workers) as pool:
            t0 = time.time()
            hit = sum(exact(pandora_batched(X, y, i, pool, batch=workers, rounds=200 // workers), wt)
                      for i, (X, y, wt) in enumerate(probs))
            t = time.time() - t0
        rows[f"{workers} workers"] = {"recovered": int(hit), "seconds": t, "speedup": t_serial / t}
        print(f"{workers} workers: recovered {hit}/30, {t:.1f}s, speedup {t_serial / t:.2f}x (serial {serial}/30, {t_serial:.1f}s)", flush=True)
    rows["cpus"] = os.cpu_count()
    (RES / "speedup.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
