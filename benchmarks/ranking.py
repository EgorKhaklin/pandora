"""Ranking-only benchmark: how often does each ranking's top r contain the entire true support?

    python -m benchmarks.ranking
"""

import json
from pathlib import Path

import numpy as np

from pandora import basis_pursuit, pandora, reweighted_l1
from .greedy import omp
from .validate import problem

RES = Path(__file__).resolve().parent.parent / "results"


def omp_order(X, y, steps):
    """Columns in the order OMP selects them, run past k (to `steps` columns)."""
    S, r = [], y.copy()
    for _ in range(steps):
        c = np.abs(X.T @ r)
        c[S] = -1
        S.append(int(np.argmax(c)))
        coef = np.linalg.lstsq(X[:, S], y, rcond=None)[0]
        r = y - X[:, S] @ coef
        if np.linalg.norm(r) < 1e-12:
            break
    score = np.zeros(X.shape[1])
    score[S] = np.arange(len(S), 0, -1)
    return score


def main():
    rows = []
    for k in (10, 12, 14):
        counts = {}
        true_rank_hist = {}
        for i in range(60):
            X, y, wt = problem("gaussian", 40, 200, k, 110_000 + 1000 * k + i)
            S = set(np.flatnonzero(wt))
            rng = np.random.default_rng(i)
            w, info = pandora(X, y, rng=i, return_info=True, rounds=200)
            scores = {"random": rng.random(200),
                      "correlation |X^T y|": np.abs(X.T @ y),
                      "OMP order": omp_order(X, y, 39),
                      "L1 magnitude": np.abs(basis_pursuit(X, y)[0]),
                      "reweighted L1 magnitude": np.abs(reweighted_l1(X, y)),
                      "Pandora score": info["score"]}
            for name, sc in scores.items():
                order = np.argsort(-sc)
                for r in (20, 39):
                    key = f"{name} | top {r}"
                    counts[key] = counts.get(key, 0) + (S <= set(order[:r]))
                ranks = np.empty(200, int)
                ranks[order] = np.arange(200)
                true_rank_hist.setdefault(name, []).extend(ranks[list(S)].tolist())
        row = {"k": k, "trials": 60, "contains_support": counts,
               "median_rank_of_true_columns": {n: float(np.median(v)) for n, v in true_rank_hist.items()},
               "share_of_true_columns_in_top_10": {n: float(np.mean(np.array(v) < 10)) for n, v in true_rank_hist.items()}}
        rows.append(row)
        print(f"k={k}:", flush=True)
        for name in ("random", "correlation |X^T y|", "OMP order", "L1 magnitude",
                     "reweighted L1 magnitude", "Pandora score"):
            print(f"   {name:26s} top20 {counts[name + ' | top 20']:2d}/60  top39 {counts[name + ' | top 39']:2d}/60"
                  f"  median rank of true columns {row['median_rank_of_true_columns'][name]:5.1f}", flush=True)
    (RES / "ranking.json").write_text(json.dumps(rows, indent=1, default=int) + "\n")


if __name__ == "__main__":
    main()
