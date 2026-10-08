import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks import trial

SMALL = ("gaussian", 20, 60, 2)  # kind, n, d, k: L1 recovers these


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "RES", tmp_path)
    monkeypatch.setattr(trial, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(trial, "CACHE", tmp_path / ".trial_cache.json")
    monkeypatch.setattr(trial, "GATE_LOCK", tmp_path / "no-gate-lock")
    return tmp_path


def methods_file(tmp_path, body):
    p = tmp_path / "cand.py"
    p.write_text("import numpy as np\nfrom pandora import basis_pursuit\n" + body)
    return p


def test_a_hanging_method_times_out_and_the_trial_carries_on(isolated):
    p = methods_file(isolated, "def hang(X, y, rng):\n    while True:\n        pass\n")
    runner = trial.Runner()
    try:
        r = trial.run_cell(runner, f"{p}:hang", *SMALL, [1, 2], timeout=2, cache={})
        assert r["timeouts"] == 2 and r["hits"] == [False, False]
        after = trial.run_cell(runner, "l1", *SMALL, [1, 2], timeout=30, cache={})
        assert after["timeouts"] == 0 and all(after["hits"])
    finally:
        runner.close()


def test_a_worker_does_not_outlive_its_trial(tmp_path):
    p = methods_file(tmp_path, "def hang(X, y, rng):\n    while True:\n        pass\n")
    script = tmp_path / "dies.py"
    script.write_text(
        "import os, sys, time\nsys.path.insert(0, os.getcwd())\nfrom benchmarks import trial\n"
        "if __name__ == '__main__':\n"
        "    r = trial.Runner()\n"
        "    assert r.run(('l1', 'gaussian', 20, 60, 2, 1), 60)[0] == 'ok'\n"  # the worker is up
        f"    r.conn.send(({str(p)!r} + ':hang', 'gaussian', 20, 60, 2, 1))\n"
        "    time.sleep(2)\n"
        "    print(r.proc.pid, flush=True)\n"
        "    os._exit(0)\n")  # dies without cleanup, as a killed terminal would
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                         cwd=Path(trial.__file__).resolve().parent.parent)
    assert out.returncode == 0 and out.stdout.strip().isdigit(), out.stderr[-2000:]
    worker = int(out.stdout)  # without the guard it spins on (mutation-checked); with it, gone within ~1 s
    deadline = time.time() + 10
    while time.time() < deadline and _alive(worker):
        time.sleep(0.2)
    alive = _alive(worker)
    if alive:
        os.kill(worker, signal.SIGKILL)
    assert not alive, "the worker kept spinning after its trial died"


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_crashing_method_is_a_miss_with_its_error_kept(isolated):
    p = methods_file(isolated, "def boom(X, y, rng):\n    raise RuntimeError('vessel broke')\n")
    runner = trial.Runner()
    try:
        r = trial.run_cell(runner, f"{p}:boom", *SMALL, [1], timeout=30, cache={})
    finally:
        runner.close()
    assert r["hits"] == [False] and "vessel broke" in r["errors"][0]


def test_a_method_that_kills_its_worker_is_a_miss_and_the_trial_carries_on(isolated):
    p = methods_file(isolated, "import os\ndef die(X, y, rng):\n    os._exit(3)\n")
    runner = trial.Runner()
    try:
        r = trial.run_cell(runner, f"{p}:die", *SMALL, [1, 2], timeout=30, cache={})
        after = trial.run_cell(runner, "l1", *SMALL, [1], timeout=30, cache={})
    finally:
        runner.close()
    assert r["hits"] == [False, False] and "died" in r["errors"][0] and after["hits"] == [True]


def test_the_cache_key_changes_with_the_problem_set():
    key = lambda kind, n, d, k, seeds: trial.cache_key("l1", kind, n, d, k, seeds)
    base = key("gaussian", 40, 200, 11, range(10))
    assert base == key("gaussian", 40, 200, 11, range(10))
    assert base != key("gaussian", 40, 200, 14, range(10))   # the Anabasis replay bug: k ignored
    assert base != key("gaussian", 40, 200, 11, range(1, 11))
    assert base != key("toeplitz", 40, 200, 11, range(10))


def test_the_cache_key_changes_with_the_method_source(tmp_path):
    p = methods_file(tmp_path, "def f(X, y, rng):\n    return basis_pursuit(X, y)[0]\n")
    before = trial.cache_key(f"{p}:f", *SMALL, range(3))
    p.write_text(p.read_text() + "# edited\n")
    assert trial.cache_key(f"{p}:f", *SMALL, range(3)) != before


def test_the_cache_key_changes_when_a_helper_the_method_imports_changes(tmp_path):
    helper = tmp_path / "trial_helper_mod.py"
    helper.write_text("SCALE = 1.0\n")
    p = tmp_path / "uses_helper.py"
    p.write_text(f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\nimport trial_helper_mod\n"
                 "from pandora import basis_pursuit\n"
                 "def f(X, y, rng):\n    return basis_pursuit(X, y)[0] * trial_helper_mod.SCALE\n")
    before = trial.cache_key(f"{p}:f", *SMALL, range(3))
    helper.write_text("SCALE = 2.0\n")
    assert trial.cache_key(f"{p}:f", *SMALL, range(3)) != before


def test_a_baseline_that_hits_the_time_limit_makes_the_trial_invalid(isolated):
    p = methods_file(isolated, "import time\ndef slow(X, y, rng):\n    time.sleep(5)\n"
                               "    return basis_pursuit(X, y)[0]\n")
    trial.main(["l1", "--baseline", f"{p}:slow", "--n", "20", "--d", "60", "--k", "2", "--tune", "6",
                "--timeout", "1", "--confirm-anyway"])
    row = json.loads((isolated / "ledger.jsonl").read_text())
    assert row["verdict"].startswith("INVALID") and row["fresh_blocks"] == []
    assert [c["verdict"] for c in row["tuning"]] == ["INVALID"]


def test_the_default_time_limit_grows_with_the_problem(isolated):
    trial.main(["l1", "--baseline", "l1", "--n", "20", "--d", "60", "--k", "2", "--tune", "1"])
    trial.main(["l1", "--baseline", "l1", "--n", "20", "--d", "400", "--k", "2", "--tune", "1"])
    small, large = [json.loads(x)["timeout"] for x in (isolated / "ledger.jsonl").read_text().splitlines()]
    assert small == 30 and large == 120


def test_workers_run_problems_at_once_with_the_same_results(isolated):
    p = methods_file(isolated, "import time\ndef slow(X, y, rng):\n    time.sleep(1)\n"
                               "    return basis_pursuit(X, y)[0]\n")
    runners = [trial.Runner() for _ in range(4)]
    try:
        t0 = time.time()
        par = trial.run_cell(runners, f"{p}:slow", *SMALL, list(range(8)), timeout=30, cache={})
        elapsed = time.time() - t0
        one = trial.run_cell(runners[:1], f"{p}:slow", *SMALL, list(range(8)), timeout=30, cache={})
    finally:
        for r in runners:
            r.close()
    assert par["hits"] == one["hits"] and all(par["hits"])
    assert elapsed < 6, f"4 workers took {elapsed:.1f}s for 8 one-second problems"


def test_a_cached_result_is_reused_only_for_the_same_problems(isolated):
    runner = trial.Runner()
    try:
        cache = {}
        first = trial.run_cell(runner, "l1", *SMALL, [1, 2], timeout=30, cache=cache)
        again = trial.run_cell(runner, "l1", *SMALL, [1, 2], timeout=30, cache=cache)
        other = trial.run_cell(runner, "l1", *SMALL, [3, 4], timeout=30, cache=cache)
    finally:
        runner.close()
    assert not first["cached"] and again["cached"] and not other["cached"]


@pytest.mark.parametrize("cand, base, verdict", [
    ([True] * 12 + [False] * 8, [False] * 12 + [False] * 8, "WIN"),
    ([False] * 12 + [True] * 8, [True] * 12 + [True] * 8, "LOSS"),
    ([True, True, False] + [True] * 17, [False, True, True] + [True] * 17, "NO DIFFERENCE"),
    ([True] * 4 + [False] * 16, [False] * 3 + [True] + [False] * 16, "NO DIFFERENCE"),  # 3 vs 1: p=0.31
    ([True] * 20, [True] * 20, "NO DIFFERENCE"),
])
def test_the_verdict_counts_only_problems_one_method_solved(cand, base, verdict):
    assert trial.sign_test(cand, base)[0] == verdict


def test_each_confirmation_runs_on_a_block_no_earlier_trial_used(isolated):
    args = ["l1", "--baseline", "l1", "--n", "20", "--d", "60", "--k", "2", "--tune", "2",
            "--confirm", "2", "--confirm-anyway"]
    trial.main(args)
    trial.main(args)
    rows = [json.loads(line) for line in (isolated / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["fresh_blocks"] != rows[1]["fresh_blocks"]
    assert rows[1]["confirm"][0]["seed0"] != rows[0]["confirm"][0]["seed0"]


def test_without_a_tuning_win_nothing_runs_on_fresh_seeds(isolated):
    trial.main(["l1", "--baseline", "l1", "--n", "20", "--d", "60", "--k", "2", "--tune", "2"])
    row = json.loads((isolated / "ledger.jsonl").read_text())
    assert row["verdict"].startswith("NOT CONFIRMED") and row["fresh_blocks"] == [] and "confirm" not in row
