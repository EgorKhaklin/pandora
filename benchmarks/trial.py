"""Trial a candidate method against a baseline: tune on fixed problems, confirm on fresh ones.

    python -m benchmarks.trial scratch/vortex.py:charybdis --k 11 14
    python -m benchmarks.trial benchmarks.greedy:omp --baseline isd --kind toeplitz

A method is "path/to/file.py:function", "package.module:function", or a builtin name (pandora,
isd, l1). It is called as f(X, y, rng) and returns w, None, or a tuple whose first item is w.

1. Tuning: both methods run on fixed tuning problems (seeds 80_000+). Variants may be tuned on
   these as often as you like, so a win here is a lead, not a result.
2. Confirmation, only after a tuning win (or --confirm-anyway): both run on a block of seeds no
   earlier trial used (900_000+, picked from the ledger). The verdict comes from this stage alone.
3. Verdict: an exact sign test on the problems exactly one method solved (McNemar). WIN or LOSS
   needs p < 0.05; anything else is NO DIFFERENCE. A stage where the BASELINE hit the time limit
   is INVALID: a truncated baseline would flatter the candidate, so raise --timeout and rerun.

Guards:
- every problem runs in a worker process with a time limit (--timeout, default max(30, 0.3 d)
  seconds); a candidate that hangs is killed, counted as a miss and reported as a timeout, and
  the trial carries on; a worker whose trial dies exits within a second instead of spinning on;
- --workers N runs N problems at once (default 1);
- cached results are keyed on the exact problem set and on every local source file the method
  imports (found by loading it in a fresh interpreter), so neither a different problem set nor
  an edited helper can replay a stale result;
- the process runs at nice 10, and the run is marked "contended" when a Polaris gate holds
  ~/Desktop/polaris/.gate-lock: counts stay valid, times do not;
- every trial appends one row to results/ledger.jsonl.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LEDGER = RES / "ledger.jsonl"
CACHE = RES / ".trial_cache.json"
GATE_LOCK = Path.home() / "Desktop" / "polaris" / ".gate-lock"
TUNE_SEED0 = 80_000
FRESH_SEED0 = 900_000
BLOCK = 10_000
BUILTINS = {"pandora", "isd", "l1"}


def load(spec):
    """The callable named by spec, as f(X, y, rng) -> w | None."""
    import pandora as P
    if spec == "pandora":
        return lambda X, y, rng: P.pandora(X, y, rng=rng)
    if spec == "isd":
        return lambda X, y, rng: P.iterative_support_detection(X, y)
    if spec == "l1":
        return lambda X, y, rng: P.basis_pursuit(X, y)[0]
    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise ValueError(f"method must be file.py:function, module:function or one of {sorted(BUILTINS)}: {spec}")
    if target.endswith(".py"):
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(f"_trial_{path.stem}", path)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[mod_spec.name] = mod  # so source_files sees the method file itself
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, name)


_LIST_SOURCES = """
import json, sys, sysconfig
sys.path.insert(0, sys.argv[2])
from benchmarks import trial
trial.load(sys.argv[1])
skip = tuple(sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib"))
files = {getattr(m, "__file__", None) for m in list(sys.modules.values())}
print(json.dumps(sorted(f for f in files if f and f.endswith(".py") and not f.startswith(skip)
                        and f != trial.__file__)))
"""


def source_files(spec):
    """Every local .py file loading spec imports (pandora itself, the method, its helpers),
    from a fresh interpreter so nothing this process already imported leaks in."""
    out = subprocess.run([sys.executable, "-c", _LIST_SOURCES, spec, str(ROOT)],
                         capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise ValueError(f"cannot load {spec}: {out.stderr.strip()[-500:]}")
    return json.loads(out.stdout)


def source_digest(spec):
    """Hash of the code a result depends on: every local source file the method loads."""
    h = hashlib.sha256(spec.encode())
    for f in source_files(spec):
        h.update(f.encode())
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def cache_key(spec, kind, n, d, k, seeds, digest=None):
    problems = json.dumps([kind, n, d, k, list(seeds)])
    return hashlib.sha256(f"{digest or source_digest(spec)}|{problems}".encode()).hexdigest()[:24]


def _exit_with_parent(parent):
    while os.getppid() == parent:
        time.sleep(1)
    os._exit(1)  # the trial died: never outlive it spinning in a method (occult.py ran 2.5 h this way)


def _worker(conn, parent):
    from benchmarks.validate import exact, problem
    threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()
    methods = {}
    while (job := conn.recv()) is not None:
        spec, kind, n, d, k, seed = job
        try:
            f = methods.get(spec) or methods.setdefault(spec, load(spec))
            X, y, wt = problem(kind, n, d, k, seed)
            t0 = time.perf_counter()
            w = f(X, y, np.random.default_rng(seed))
            w = w[0] if isinstance(w, tuple) else w
            conn.send(("ok", bool(exact(w, wt)), time.perf_counter() - t0))
        except Exception as e:  # a crashing method is a miss, not a crashed trial
            conn.send(("error", False, repr(e)[:200]))


class Runner:
    """One worker process; a problem past its time limit kills it and a fresh one takes over."""

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self._start()

    def _start(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker, args=(child, os.getpid()), daemon=True)
        self.proc.start()

    def run(self, job, timeout):
        try:
            self.conn.send(job)
            if self.conn.poll(timeout):
                return self.conn.recv()
            result = ("timeout", False, timeout)
        except (EOFError, OSError):  # the worker died (a segfault, os._exit in the method)
            result = ("error", False, "worker process died")
        self.proc.kill()
        self.proc.join()
        self._start()
        return result

    def close(self):
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.kill()


def _read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def run_cell(runners, spec, kind, n, d, k, seeds, timeout, cache, digest=None):
    """{'hits': [bool per seed], 'timeouts', 'errors', 'seconds', 'cached'} for one method on one
    cell, its problems spread over the runners."""
    runners = runners if isinstance(runners, list) else [runners]
    key = cache_key(spec, kind, n, d, k, seeds, digest)
    if key in cache:
        return {**cache[key], "cached": True}
    results, todo, lock = [None] * len(seeds), iter(range(len(seeds))), threading.Lock()

    def drain(runner):
        while True:
            with lock:
                i = next(todo, None)
            if i is None:
                return
            results[i] = runner.run((spec, kind, n, d, k, seeds[i]), timeout)

    threads = [threading.Thread(target=drain, args=(r,)) for r in runners]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    hits, timeouts, errors, seconds = [], 0, [], 0.0
    for status, hit, info in results:
        hits.append(hit)
        if status == "timeout":
            timeouts += 1
            seconds += timeout
        elif status == "error":
            errors.append(info)
        else:
            seconds += info
    out = {"hits": hits, "timeouts": timeouts, "errors": errors[:3], "seconds": round(seconds, 1)}
    if not timeouts and not errors:  # only clean runs are reused
        cache[key] = out
        CACHE.write_text(json.dumps(cache))
    return {**out, "cached": False}


def sign_test(cand, base):
    """(verdict, p, cand_only, base_only) from paired hits: exact one-sided binomial on discordant pairs."""
    b = sum(c and not a for c, a in zip(cand, base))
    c = sum(a and not c_ for c_, a in zip(cand, base))
    m = b + c
    tail = lambda x: sum(comb(m, i) for i in range(x, m + 1)) / 2 ** m if m else 1.0
    if b > c and tail(b) < 0.05:
        return "WIN", round(tail(b), 4), b, c
    if c > b and tail(c) < 0.05:
        return "LOSS", round(tail(c), 4), b, c
    return "NO DIFFERENCE", round(tail(max(b, c)), 4), b, c


def used_blocks():
    blocks = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                blocks.update(json.loads(line).get("fresh_blocks", []))
            except ValueError:
                pass
    return blocks


def stage(runners, a, label, cand, base, cells, cache, digests):
    from benchmarks.validate import wilson as _wilson
    wilson = lambda h, t: [float(x) for x in _wilson(h, t)]
    rows, all_c, all_b, base_timeouts = [], [], [], 0
    for (kind, n, d, k), seeds in cells:
        rc = run_cell(runners, cand, kind, n, d, k, seeds, a.timeout, cache, digests[cand])
        rb = run_cell(runners, base, kind, n, d, k, seeds, a.timeout, cache, digests[base])
        v, p, b_only, c_only = sign_test(rc["hits"], rb["hits"])
        base_timeouts += rb["timeouts"]
        if rb["timeouts"]:
            v = "INVALID"
        t = len(seeds)
        hc, hb = sum(rc["hits"]), sum(rb["hits"])
        flag = "".join(f" [{name}: {r['timeouts']} timeouts]" for name, r in (("cand", rc), ("base", rb)) if r["timeouts"])
        flag += "".join(f" [{name}: {len(r['errors'])}+ errors, e.g. {r['errors'][0]}]" for name, r in (("cand", rc), ("base", rb)) if r["errors"])
        print(f"{label:7s} {kind} {n}x{d} k={k:2d}: cand {hc}/{t} {wilson(hc, t)} vs base {hb}/{t} "
              f"| only cand {b_only}, only base {c_only} -> {v} (p={p}) "
              f"[{rc['seconds']}s vs {rb['seconds']}s]{flag}", flush=True)
        rows.append({"kind": kind, "n": n, "d": d, "k": k, "trials": t, "seed0": seeds[0],
                     "cand_hits": hc, "base_hits": hb, "cand_ci": wilson(hc, t), "base_ci": wilson(hb, t),
                     "only_cand": b_only, "only_base": c_only, "verdict": v, "p": p,
                     "cand_seconds": rc["seconds"], "base_seconds": rb["seconds"],
                     "cand_timeouts": rc["timeouts"], "base_timeouts": rb["timeouts"],
                     "cand_errors": len(rc["errors"]), "base_errors": len(rb["errors"])})
        all_c += rc["hits"]
        all_b += rb["hits"]
    if base_timeouts:
        return rows, ("INVALID", None, base_timeouts, 0)
    return rows, sign_test(all_c, all_b)


def git_head():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "pandora"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("method")
    ap.add_argument("--baseline", default="pandora")
    ap.add_argument("--kind", default="gaussian")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--d", type=int, default=200)
    ap.add_argument("--k", type=int, nargs="+", default=[11, 14])
    ap.add_argument("--tune", type=int, default=40, help="tuning problems per cell")
    ap.add_argument("--confirm", type=int, default=100, help="fresh problems per cell")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per problem (default max(30, 0.3 d))")
    ap.add_argument("--workers", type=int, default=1, help="problems run at once")
    ap.add_argument("--confirm-anyway", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    a.timeout = a.timeout or max(30.0, 0.3 * a.d)
    digests = {spec: source_digest(spec) for spec in (a.method, a.baseline)}  # fails fast on a bad spec
    try:
        os.nice(10)
    except OSError:
        pass
    RES.mkdir(exist_ok=True)
    cache = _read_json(CACHE, {})
    contended = GATE_LOCK.exists()
    if contended:
        print("note: a Polaris gate holds the CPU; counts are valid, times are not comparable", flush=True)
    t0 = time.time()
    runners = [Runner() for _ in range(max(1, a.workers))]
    row = {"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": git_head(),
           "method": a.method, "method_source": digests[a.method], "baseline": a.baseline,
           "note": a.note, "timeout": a.timeout, "workers": len(runners), "fresh_blocks": []}
    try:
        tune_cells = [((a.kind, a.n, a.d, k), list(range(TUNE_SEED0 + 1000 * k, TUNE_SEED0 + 1000 * k + a.tune)))
                      for k in a.k]
        row["tuning"], (tv, tp, *_) = stage(runners, a, "tuning", a.method, a.baseline, tune_cells, cache,
                                            digests)
        row["tuning_verdict"] = tv
        if tv == "INVALID":
            row["verdict"] = f"INVALID: the baseline hit the {a.timeout:.0f} s time limit; raise --timeout"
            print(f"VERDICT: {row['verdict']}")
        elif tv == "WIN" or a.confirm_anyway:
            used = used_blocks()
            block = next(b for b in range(10_000) if b not in used)
            row["fresh_blocks"] = [block]
            base = FRESH_SEED0 + BLOCK * block
            fresh_cells = [((a.kind, a.n, a.d, k), list(range(base + 1000 * i, base + 1000 * i + a.confirm)))
                           for i, k in enumerate(a.k)]
            row["confirm"], (fv, fp, b_only, c_only) = stage(runners, a, "confirm", a.method, a.baseline,
                                                             fresh_cells, {}, digests)
            if fv == "INVALID":
                fv = f"INVALID: the baseline hit the {a.timeout:.0f} s time limit; raise --timeout"
            row["verdict"], row["p"] = fv, fp
            print(f"VERDICT (fresh seeds, block {block}): {fv}, p={fp}, only cand {b_only}, only base {c_only}")
        else:
            row["verdict"] = f"NOT CONFIRMED (tuning: {tv}, p={tp})"
            print(f"VERDICT: {row['verdict']}; nothing ran on fresh seeds (--confirm-anyway to force)")
    finally:
        for r in runners:
            r.close()
        row["seconds"] = round(time.time() - t0, 1)
        row["contended"] = contended or GATE_LOCK.exists()
        with LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    main()
