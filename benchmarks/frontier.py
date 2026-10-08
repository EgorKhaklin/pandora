"""The frontier: Pandora against the strongest known methods for noiseless sparse recovery.

    python -m benchmarks.frontier tune       # settings and restart budgets, on tuning problems
    python -m benchmarks.frontier compare    # fresh problems, paired against Pandora
    python -m benchmarks.frontier se         # large-system thresholds of L1 and Bayes AMP
    python -m benchmarks.frontier threshold  # Pandora vs that line as n grows, budget up to 50 n

Methods, each followed by Pandora's exact-fit certificate:
- AMP: approximate message passing (Donoho, Maleki and Montanari, 2009) with the Bayes-optimal
  denoiser for the true signal law (Krzakala et al., 2012). It is told the sparsity and the
  magnitude law; Pandora is not.
- IRLS-lp: iteratively reweighted least squares for the lp quasi-norm with epsilon continuation
  (Chartrand and Yin, 2008; Daubechies et al., 2010), p = 0.5 and p = 0.1.
- SBL: sparse Bayesian learning, noiseless, with MacKay's updates (Wipf and Rao, 2004).
- ISD restarted under the certificate (benchmarks.validate.restarted_isd).

Budgets: IRLS and SBL restart from random initial weights until the certificate passes. Their
restart caps are fixed in `tune` so that their mean time matches Pandora's on the tuning problems,
which keeps `compare` deterministic.
"""

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr

from pandora import pandora
from pandora.core import certify
from .validate import exact, problem, restarted_isd, sign_test, wilson

RES = Path(__file__).resolve().parent.parent / "results"
WORKERS = int(os.environ.get("PANDORA_WORKERS", "5"))
LO, HI = 1.0, 2.0          # the benchmark law: random signs, magnitudes uniform on [1, 2]


# ---- AMP with the Bayes-optimal denoiser ---------------------------------------------------------

def _log_mass(alpha, beta):
    """log(Phi(beta) - Phi(alpha)) for beta > alpha, without cancellation in either tail."""
    upper = alpha > 0
    a = np.where(upper, -beta, alpha)
    b = np.where(upper, -alpha, beta)
    la, lb = log_ndtr(a), log_ndtr(b)
    return lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))


def _upper_ratios(a, b):
    """phi(a)/Z and phi(b)/Z for 0 <= a < b, Z = Phi(b) - Phi(a), via the scaled erfc: no cancellation
    however far into the tail (phi(a)/Phi_c(a) = 2 / (sqrt(2 pi) erfcx(a / sqrt 2)))."""
    s2 = np.sqrt(2.0)
    g = np.exp(-0.5 * (b - a) * (b + a))                    # phi(b) / phi(a)
    q = erfcx(b / s2) / erfcx(a / s2) * g                   # Phi_c(b) / Phi_c(a)
    pa = 2 / (np.sqrt(2 * np.pi) * erfcx(a / s2)) / (1 - q)
    return pa, pa * g


def _tn_moments(alpha, beta):
    """Mean and variance of a standard normal truncated to [alpha, beta], stable at any tau."""
    alpha, beta = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(beta, float))
    pa, pb = np.empty_like(alpha), np.empty_like(alpha)
    up, lo = alpha > 0, beta < 0
    mid = ~(up | lo)
    pa[up], pb[up] = _upper_ratios(alpha[up], beta[up])
    pb[lo], pa[lo] = _upper_ratios(-beta[lo], -alpha[lo])  # reflect x -> -x
    z = ndtr(beta[mid]) - ndtr(alpha[mid])
    c = 1 / np.sqrt(2 * np.pi)
    pa[mid] = c * np.exp(-0.5 * alpha[mid] ** 2) / z
    pb[mid] = c * np.exp(-0.5 * beta[mid] ** 2) / z
    mean = pa - pb
    with np.errstate(invalid="ignore", over="ignore"):
        var = 1 + alpha * pa - beta * pb - mean ** 2
    return mean, np.clip(np.nan_to_num(var), 0, 1)        # a truncated standard normal has var <= 1


def denoise(r, tau, eps):
    """Posterior mean and variance of W given W + tau Z = r, for W = 0 with probability 1 - eps,
    else a random sign times Uniform[LO, HI]."""
    r = np.asarray(r, float)
    logs, means, varis = [np.log1p(-eps) - 0.5 * (r / tau) ** 2 - np.log(tau)], [np.zeros_like(r)], [np.zeros_like(r)]
    for a, b in ((LO, HI), (-HI, -LO)):
        alpha, beta = (a - r) / tau, (b - r) / tau
        logs.append(np.log(eps / 2) - np.log(b - a) + _log_mass(alpha, beta) + 0.5 * np.log(2 * np.pi))
        m, v = _tn_moments(alpha, beta)
        means.append(r + tau * np.nan_to_num(m))
        varis.append(tau ** 2 * np.nan_to_num(v))
    L = np.array(logs)
    P = np.exp(L - L.max(axis=0))
    P /= P.sum(axis=0)
    M, V = np.array(means), np.array(varis)
    mean = (P * M).sum(axis=0)
    var = np.clip((P * (V + M ** 2)).sum(axis=0) - mean ** 2, 0, None)
    return mean, var


def amp(X, y, eps, iters=300, damping=1.0):
    """Bayes-optimal AMP on A = X / sqrt(n). Returns (x, iterations)."""
    n, d = X.shape
    A, ya = X / np.sqrt(n), y / np.sqrt(n)
    delta = n / d
    x, z = np.zeros(d), ya.copy()
    scale = np.linalg.norm(ya)
    t = 0
    for t in range(1, iters + 1):
        tau = np.linalg.norm(z) / np.sqrt(n)
        if tau <= 1e-11 * scale / np.sqrt(n):      # converged far below the certificate's 1e-8
            break
        mean, var = denoise(x + A.T @ z, tau, eps)
        x_new = damping * mean + (1 - damping) * x
        z_new = ya - A @ x_new + (1 / delta) * z * np.mean(var) / tau ** 2
        if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(z_new))):
            break                                  # keep the last finite iterate
        x, z = x_new, z_new
    return x, t


# ---- IRLS for the lp quasi-norm -------------------------------------------------------------------

def irls_lp(X, y, p=0.5, rng=None, max_iter=600):
    """Chartrand-Yin IRLS: x = Q X^T (X Q X^T)^-1 y with Q = diag((x^2 + eps)^(1 - p/2)), eps from 1
    down to 1e-8, cut tenfold whenever the iterate settles. A generator rng starts from random weights."""
    q = np.ones(X.shape[1]) if rng is None else rng.uniform(0.5, 1.5, X.shape[1])
    x = q * (X.T @ np.linalg.solve((X * q) @ X.T, y))
    e = 1.0
    for _ in range(max_iter):
        q = (x ** 2 + e) ** (1 - p / 2)
        x_new = q * (X.T @ np.linalg.solve((X * q) @ X.T, y))
        if np.linalg.norm(x_new - x) < np.sqrt(e) / 100:
            e /= 10
            if e < 1e-8:
                return x_new
        x = x_new
    return x


# ---- sparse Bayesian learning ---------------------------------------------------------------------

def sbl(X, y, rng=None, max_iter=600):
    """Noiseless SBL with MacKay updates gamma_i <- mu_i^2 / (gamma_i g_i), g = diag(X^T M^-1 X),
    M = X Gamma X^T (a tiny ridge keeps M invertible once fewer than n gammas remain)."""
    n, d = X.shape
    g_ = np.ones(d) if rng is None else rng.uniform(0.5, 1.5, d)
    mu = np.zeros(d)
    for _ in range(max_iter):
        M = (X * g_) @ X.T
        M += 1e-12 * np.trace(M) / n * np.eye(n)
        B = np.linalg.solve(M, X)
        mu = g_ * (B.T @ y)
        gi = np.einsum("ij,ij->j", X, B)
        new = np.where(g_ > 0, mu ** 2 / np.maximum(g_ * gi, 1e-300), 0.0)
        new[new < 1e-12 * max(new.max(), 1e-300)] = 0.0
        if np.max(np.abs(new - g_)) <= 1e-9 * max(new.max(), 1e-300):
            g_ = new
            break
        g_ = new
    return mu


# ---- state evolution: where AMP and L1 stop, as n -> infinity ------------------------------------

_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(161)          # E[f(Z)], Z ~ N(0, 1)
_GH_W = _GH_W / np.sqrt(2 * np.pi)
_GL_U, _GL_W = np.polynomial.legendre.leggauss(48)               # E[f(U)], U ~ Uniform[1, 2]
_GL_U, _GL_W = 1.5 + 0.5 * _GL_U, 0.5 * _GL_W


def mmse(tau, eps):
    """E[(posterior mean - W)^2] for W + tau Z under the benchmark law, by quadrature."""
    zero = np.sum(_GH_W * denoise(tau * _GH_Z, tau, eps)[0] ** 2)
    R = _GL_U[:, None] + tau * _GH_Z[None, :]
    err = (denoise(R.ravel(), tau, eps)[0].reshape(R.shape) - _GL_U[:, None]) ** 2
    nonzero = np.sum(_GL_W[:, None] * _GH_W[None, :] * err)       # the sign is symmetric
    return (1 - eps) * zero + eps * nonzero


def amp_succeeds(delta, rho, iters=4000):
    """State evolution tau^2 <- mmse(tau) / delta from x = 0: does it reach tau = 0?"""
    eps = rho * delta
    t2 = eps * (7 / 3) / delta                                     # E[W^2] = eps E[U^2], E[U^2] = 7/3
    for _ in range(iters):
        t2_new = mmse(np.sqrt(t2), eps) / delta
        if t2_new < 1e-14:
            return True
        if t2_new > t2 * (1 - 1e-9):                               # stuck at a nonzero fixed point
            return False
        t2 = t2_new
    return False


def rho_amp(delta, tol=1e-4):
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if amp_succeeds(delta, mid) else (lo, mid)
    return lo


def rho_l1(delta):
    """L1's phase transition (Donoho, Maleki and Montanari, 2009), equal to Donoho-Tanner's."""
    z = np.linspace(1e-4, 10, 200001)
    m = (1 + z ** 2) * ndtr(-z) - z * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return float(np.max((1 - 2 / delta * m) / (1 + z ** 2 - 2 * m)))


# ---- certified, restarted wrappers -----------------------------------------------------------------

def certified(X, y, run, restarts, rng):
    """Run `run(rng_or_None)` up to `restarts` times (the first from its default start) and return the
    first certified answer; else the last raw output. Returns (w, runs used)."""
    w = None
    for j in range(restarts):
        w = run(None if j == 0 else rng)
        wc = certify(X, y, np.abs(w))
        if wc is not None:
            return wc, j + 1
    return w, restarts


def methods(eps, caps):
    """name -> f(X, y, seed) returning (w, cost); cost is LPs for Pandora/ISD, runs otherwise."""
    return {
        "Pandora": lambda X, y, s: (lambda r: (r[0], 1 + r[1]["rounds"]))(pandora(X, y, rng=s, return_info=True)),
        "ISD restarted": lambda X, y, s: restarted_isd(X, y, np.random.default_rng(10_000 + s)),
        "AMP (told the prior)": lambda X, y, s: (lambda x: (certify(X, y, np.abs(x)) if certify(X, y, np.abs(x)) is not None else x, 1))(
            amp(X, y, eps, damping=caps.get("amp_damping", 1.0))[0]),
        "IRLS p=0.5": lambda X, y, s: certified(X, y, lambda g: irls_lp(X, y, 0.5, g), caps["IRLS p=0.5"],
                                                np.random.default_rng(20_000 + s)),
        "IRLS p=0.1": lambda X, y, s: certified(X, y, lambda g: irls_lp(X, y, 0.1, g), caps["IRLS p=0.1"],
                                                np.random.default_rng(30_000 + s)),
        "SBL": lambda X, y, s: certified(X, y, lambda g: sbl(X, y, g), caps["SBL"], np.random.default_rng(40_000 + s)),
    }


def _one(job):
    n, d, k, seed, i, caps, names = job
    X, y, wt = problem("gaussian", n, d, k, seed)
    out = {}
    for name, f in methods(k / d, caps).items():
        if name not in names:
            continue
        t0 = time.perf_counter()
        w, cost = f(X, y, i)
        out[name] = (bool(exact(w, wt)), float(cost), time.perf_counter() - t0)
    return out


def run_cell(pool, n, d, k, seed0, trials, caps, names):
    jobs = [(n, d, k, seed0 + 1000 * k + i, i, caps, names) for i in range(trials)]
    return pool.map(_one, jobs, chunksize=1)


def tune(pool):
    """AMP damping, and restart caps that match Pandora's mean time, on tuning problems (40 x 200)."""
    caps = {"IRLS p=0.5": 1, "IRLS p=0.1": 1, "SBL": 1}
    report = {}
    cells = [(40, 200, 12), (40, 200, 14)]
    for damping in (1.0, 0.7, 0.5):
        hits = 0
        for n, d, k in cells:
            res = run_cell(pool, n, d, k, 610_000, 40, {**caps, "amp_damping": damping}, {"AMP (told the prior)"})
            hits += sum(r["AMP (told the prior)"][0] for r in res)
        report[f"AMP damping {damping}"] = hits
        print(f"tune AMP damping {damping}: {hits}/80", flush=True)
    damping = max((v, -abs(1 - float(k.split()[-1])), float(k.split()[-1])) for k, v in report.items())[2]
    times = {}
    for n, d, k in cells:
        res = run_cell(pool, n, d, k, 610_000, 40, caps, {"Pandora", "IRLS p=0.5", "IRLS p=0.1", "SBL"})
        for name in ("Pandora", "IRLS p=0.5", "IRLS p=0.1", "SBL"):
            times.setdefault(name, []).extend(r[name][2] for r in res)
    tp = float(np.mean(times["Pandora"]))
    out = {"amp_damping": damping, "pandora_seconds": tp}
    for name in ("IRLS p=0.5", "IRLS p=0.1", "SBL"):
        one = float(np.mean(times[name]))
        out[name] = max(1, int(round(tp / one)))
        print(f"tune {name}: {one:.3f}s per run vs Pandora {tp:.3f}s -> up to {out[name]} runs", flush=True)
    (RES / "frontier_tune.json").write_text(json.dumps({"report": report, "caps": out}, indent=1) + "\n")
    return out


def compare(pool):
    caps = json.loads((RES / "frontier_tune.json").read_text())["caps"]
    names = {"Pandora", "ISD restarted", "AMP (told the prior)", "IRLS p=0.5", "IRLS p=0.1", "SBL"}
    rows = []
    for (n, d, k, trials) in ((40, 200, 10, 100), (40, 200, 12, 100), (40, 200, 14, 100), (40, 200, 16, 100),
                              (100, 500, 30, 50), (100, 500, 35, 50)):
        t0 = time.time()
        res = run_cell(pool, n, d, k, 720_000 + 10 * d, trials, caps, names)
        P = [r["Pandora"][0] for r in res]
        row = {"n": n, "d": d, "k": k, "trials": trials, "methods": {}}
        for name in sorted(names, key=lambda m: m != "Pandora"):
            H = [r[name][0] for r in res]
            only_p, only_m, p = sign_test(P, H)
            row["methods"][name] = {"hits": sum(H), "ci95": wilson(sum(H), trials),
                                    "mean_cost": float(np.mean([r[name][1] for r in res])),
                                    "mean_seconds": float(np.mean([r[name][2] for r in res])),
                                    "only_pandora": only_p, "only_this": only_m, "p": round(p, 4)}
        rows.append(row)
        print(f"compare {n}x{d} k={k} ({time.time() - t0:.0f}s): " + " | ".join(
            f"{m} {v['hits']}/{trials}" + ("" if m == "Pandora" else f" (P-only {v['only_pandora']}, "
                                                                  f"{m.split()[0]}-only {v['only_this']}, p={v['p']})")
            for m, v in row["methods"].items()), flush=True)
    (RES / "frontier_compare.json").write_text(json.dumps(rows, indent=1) + "\n")


def _threshold_job(job):
    n, d, k, seed, i, cap, damping = job
    X, y, wt = problem("gaussian", n, d, k, seed)
    t0 = time.perf_counter()
    w, info = pandora(X, y, rng=i, rounds=cap, return_info=True)
    tp = time.perf_counter() - t0
    hit = bool(exact(w, wt))
    x, _ = amp(X, y, k / d, damping=damping)
    wa = certify(X, y, np.abs(x))
    return {"pandora_hit": hit, "pandora_rounds": int(info["rounds"]), "pandora_seconds": tp,
            "amp_hit": bool(exact(wa if wa is not None else x, wt))}


def threshold(pool):
    """Does a growing budget keep Pandora above Bayes AMP's large-system line (k/n = 0.299 at n/d = 0.2)
    as n grows? One run per problem with a cap of 50 n vessels: a run that certifies by vessel r
    succeeds under every budget >= r, so success at budgets n, 5n, 10n, 25n, 50n is read off one run."""
    damping = json.loads((RES / "frontier_tune.json").read_text())["caps"]["amp_damping"]
    rows = []
    for n, trials in ((40, 40), (80, 30), (160, 20)):
        d = 5 * n
        for rho in (0.28, 0.30, 0.32, 0.34):
            k = int(round(rho * n))
            t0 = time.time()
            res = pool.map(_threshold_job, [(n, d, k, 800_000 + 10 * d + 1000 * k + i, i, 50 * n, damping)
                                            for i in range(trials)], chunksize=1)
            by_budget = {c: sum(r["pandora_hit"] and r["pandora_rounds"] <= c * n for r in res) for c in (1, 5, 10, 25, 50)}
            rounds = sorted(r["pandora_rounds"] for r in res if r["pandora_hit"])
            row = {"n": n, "d": d, "k": k, "k_over_n": round(k / n, 4), "trials": trials,
                   "pandora_by_budget": {f"{c}n": h for c, h in by_budget.items()},
                   "pandora_rounds_of_successes": rounds, "amp": sum(r["amp_hit"] for r in res),
                   "pandora_seconds_mean": float(np.mean([r["pandora_seconds"] for r in res]))}
            rows.append(row)
            print(f"threshold n={n} k={k} (k/n={k / n:.3f}): Pandora by budget " +
                  ", ".join(f"{c}n {h}" for c, h in by_budget.items()) + f" of {trials}; AMP {row['amp']}; "
                  f"success rounds {rounds[:12]}{'...' if len(rounds) > 12 else ''} ({time.time() - t0:.0f}s)", flush=True)
            (RES / "frontier_threshold.json").write_text(json.dumps(rows, indent=1) + "\n")


def main(part):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    RES.mkdir(exist_ok=True)
    with Pool(WORKERS) as pool:
        if part == "se":
            rows = [{"delta": dl, "rho_l1": round(rho_l1(dl), 4), "rho_amp": round(rho_amp(dl), 4)}
                    for dl in (0.1, 0.2, 0.3, 0.5)]
            for r in rows:
                print(f"se delta={r['delta']}: L1 stops at k/n = {r['rho_l1']}, Bayes AMP at k/n = {r['rho_amp']}",
                      flush=True)
            (RES / "frontier_se.json").write_text(json.dumps(rows, indent=1) + "\n")
        elif part == "threshold":
            threshold(pool)
        elif part == "tune":
            tune(pool)
        elif part == "compare":
            compare(pool)
        else:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1])
