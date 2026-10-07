"""How many measurements does each method need? Recovery curves P(exact | m) at fixed sparsity k,
and m50, the measurements needed for 50% recovery (linear interpolation).

    python -m benchmarks.measurements
"""

import json
from pathlib import Path

import numpy as np

from pandora import basis_pursuit, iterative_support_detection, pandora
from .validate import exact, problem

RES = Path(__file__).resolve().parent.parent / "results"


def m50(ms, rates):
    for (m0, r0), (m1, r1) in zip(zip(ms, rates), zip(ms[1:], rates[1:])):
        if r0 < 0.5 <= r1:
            return m0 + (0.5 - r0) * (m1 - m0) / (r1 - r0)
    return None


def main():
    out = []
    for k in (5, 8, 11):
        ms = list(range(max(2 * k, 12), 65, 4))
        curves = {"L1": [], "ISD": [], "Pandora": []}
        for m in ms:
            h = {key: 0 for key in curves}
            for i in range(30):
                X, y, wt = problem("gaussian", m, 200, k, 120_000 + 1000 * k + 10 * m + i)
                h["L1"] += exact(basis_pursuit(X, y)[0], wt)
                h["ISD"] += exact(iterative_support_detection(X, y), wt)
                h["Pandora"] += exact(pandora(X, y, rng=i), wt)
            for key in curves:
                curves[key].append(h[key] / 30)
            print(f"k={k} m={m}: " + " | ".join(f"{key} {h[key]}/30" for key in curves), flush=True)
        row = {"k": k, "d": 200, "m": ms, "trials": 30, "curves": curves,
               "m50": {key: m50(ms, v) for key, v in curves.items()}}
        out.append(row)
        print(f"k={k} m50: {row['m50']}", flush=True)
    (RES / "measurements.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
