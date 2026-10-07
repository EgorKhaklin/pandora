"""Pandora at scale: does a larger vessel budget or a different recipe restore the lead at d = 1000?

    python -m benchmarks.atscale
"""

import json
import time
from pathlib import Path

import numpy as np

from pandora import basis_pursuit, iterative_support_detection, pandora
from .validate import exact, problem

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    n, d, k, trials = 200, 1000, 55, 10
    probs = [problem("gaussian", n, d, k, 170_000 + i) for i in range(trials)]
    rows = []
    l1 = sum(exact(basis_pursuit(X, y)[0], wt) for X, y, wt in probs)
    isd = sum(exact(iterative_support_detection(X, y), wt) for X, y, wt in probs)
    print(f"d={d} n={n} k={k}: L1 {l1}/{trials}, ISD {isd}/{trials}", flush=True)
    configs = [("200 vessels, decay 0.5", dict(rounds=200)),
               ("600 vessels, decay 0.5", dict(rounds=600)),
               ("1000 vessels, decay 0.5", dict(rounds=1000)),
               ("600 vessels, decay 0.3", dict(rounds=600, decay=0.3)),
               ("600 vessels, top n/4", dict(rounds=600, top=n // 4, rest=n + n // 4))]
    for name, kw in configs:
        t0 = time.time()
        hits, used = 0, []
        for i, (X, y, wt) in enumerate(probs):
            w, info = pandora(X, y, rng=i, return_info=True, **kw)
            hits += exact(w, wt)
            used.append(info["rounds"])
        rows.append({"config": name, "recovered": int(hits), "trials": trials,
                     "mean_vessels": float(np.mean(used)), "seconds": time.time() - t0})
        print(f"{name}: {hits}/{trials}, mean vessels {np.mean(used):.0f} ({time.time() - t0:.0f}s)", flush=True)
    (RES / "atscale.json").write_text(json.dumps({"L1": int(l1), "ISD": int(isd), "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
