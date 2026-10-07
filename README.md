# pandora

### Certified sparse recovery from many small basis-pursuit solves.

[![Tests](https://github.com/EgorKhaklin/pandora/actions/workflows/tests.yml/badge.svg)](https://github.com/EgorKhaklin/pandora/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-2a78d6?style=flat-square)](LICENSE)

You measure `y = X w` with fewer measurements than unknowns, and `w` is sparse. The standard
tool is L1 minimization (basis pursuit), one linear program. Pandora instead opens many small
"vessels": basis pursuit on a few dozen columns at a time. It adds up what each vessel finds,
and stops when the data certify the answer. Every answer it returns has passed an exact-fit
test, so a hit is a recovery, not a guess.

```python
import numpy as np
from pandora import pandora

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 200))                      # 40 measurements, 200 unknowns
w_true = np.zeros(200)
w_true[rng.choice(200, 11, replace=False)] = rng.uniform(1, 2, 11)
w = pandora(X, X @ w_true, rng=0)                    # None if no certificate within the budget
```

## Results

All results use fresh random problems that were never used to choose settings. Recovery counts
as relative error below 0.01. Everything is in [results/](results) and reruns with the commands
at the bottom.

### Against seven methods

![baselines](figures/baselines.png)

100 problems per cell, with 95% confidence intervals (Wilson). OMP, CoSaMP, Subspace Pursuit
and IHT are given the true number of nonzeros. Pandora is not.

| setting | L1 | ISD | CoSaMP | Subspace Pursuit | IHT | OMP | **Pandora** |
|---|---|---|---|---|---|---|---|
| 40 × 200, 10 nonzeros | 47 | 68 | 34 | 28 | 23 | 10 | **90** (83–95) |
| 40 × 200, 12 nonzeros | 13 | 33 (25–43) | 13 | 6 | 3 | 1 | **56** (46–65) |
| 40 × 200, 14 nonzeros | 4 | 8 | 1 | 2 | 0 | 0 | **21** (14–30) |
| 100 × 500, 30 nonzeros | 4 | 13 (8–21) | 1 | 2 | 0 | 0 | **34** (26–44) |

ISD is iterative support detection (Wang and Yin, 2010), the strongest baseline here.
Across 7 cells Pandora is never beaten. Where problems are hard, its interval sits above ISD's.

### Fewer measurements for the same recovery

![measurements](figures/measurements.png)

Fix the sparsity, vary the number of measurements, and find `m50`, the number of measurements
at which half the problems are recovered (200 unknowns, 30 problems per point):

| nonzeros | L1 | ISD | **Pandora** | fewer than L1 | fewer than ISD |
|---|---|---|---|---|---|
| 5 | 23.5 | 22.0 | **18.3** | 22% | 17% |
| 8 | 34.3 | 33.3 | **27.1** | 21% | 19% |
| 11 | 41.2 | 40.0 | **36.8** | 11% | 8% |

### Phase diagram and four kinds of matrix

![phase diagram](figures/phase.png)

![ensembles](figures/ensembles.png)

With Gaussian matrices from 20 to 100 measurements (30 problems per point), and with ±1
entries, rows of a discrete cosine transform and correlated columns (40 problems per cell),
Pandora ties or beats ISD in all 28 cells.

### Larger problems

The same ratio, 1 measurement per 5 unknowns, at growing size (Pandora's budget fixed at 200
vessels; [results/validate_scaling.json](results/validate_scaling.json) when complete):

| unknowns | measurements | nonzeros | L1 | ISD | **Pandora** | Pandora time per problem |
|---|---|---|---|---|---|---|
| 200 | 40 | 12 | 5 | 7 | **19** of 30 | 0.2 s |
| 500 | 100 | 30 | 1 | 5 | **14** of 30 | 2.7 s |
| 1000 | 200 | 50 | 3 | **14** | 13 of 15 | 8.5 s |
| 1000 | 200 | 60 | 0 | 0 | 0 of 15 | 28 s |

The lead holds to 500 unknowns. At 1000 it is gone at this budget: more true columns can sit
outside the top group at once, and the chance that one vessel covers them all falls
exponentially in that number (THEORY.md). Whether a larger budget restores it is being tested.

## How it works

1. Solve basis pursuit on all columns, and score each column by `|w|`.
2. Repeat. A vessel is the `n/2` best-scored columns plus `n` random others. Solve basis pursuit
   on the vessel alone: it has `1.5 n` unknowns instead of `d`, a much easier problem. If the
   vessel's solution has fewer than `n` nonzeros, it is the answer: stop. Otherwise update
   `score ← 0.5 · score + |w_vessel|`.
3. Every 5 vessels, run least squares on the `n − 1` best-scored columns. If it fits `y` exactly,
   stop and return it.

**Why the stops are certificates** ([THEORY.md](THEORY.md)). For generic `X` and `w`, if fewer
than `n` columns fit `y` exactly, they contain the whole true support, and least squares on
them returns the true `w` (Lemma 1). So a vessel that comes back sparse has recovered `w`
(Corollary 2). This is a stopping rule built on a known fact: a sparse vector is generically
the unique sparsest solution once `n > k`.

## Why it works

Recovery comes down to one question: is the whole true support inside the ranking's top
columns? Out of 60 problems, the top 20 columns contain the entire support this often:

| ranking | 10 nonzeros | 12 nonzeros | 14 nonzeros |
|---|---|---|---|
| random | 0 | 0 | 0 |
| correlation \|Xᵀy\| | 0 | 0 | 0 |
| OMP selection order | 10 | 0 | 0 |
| L1 magnitudes | 33 | 12 | 0 |
| reweighted L1 magnitudes | 34 | 12 | 0 |
| **Pandora scores** | **52** | **40** | **11** |

The median true column ranks nearly the same under L1 and Pandora (5, 7 and 15 against 5, 7
and 13). The difference is in the last few true columns. An L1 solution has at most `n` nonzeros, and every column it sets to
zero gets score 0 and cannot be ranked. Pandora's vessels re-solve smaller problems in which
those columns can come back. A vessel that contains the whole support succeeds with the high
probability basis pursuit has at undersampling `n / 1.5n = 2/3`, compared with `n/d` for the
full problem. [THEORY.md](THEORY.md) proves the certificates and the conditional step, gives
the probability that a vessel covers the support, and states what is still open: a bound on
how fast the scores pull true columns up.

**What each part contributes** (60 problems, 40 × 200):

| variant | 10 nonzeros | 12 nonzeros |
|---|---|---|
| Pandora | 51 | 30 |
| vessels with no bias toward the top columns | 34 | 8 |
| no forgetting (`decay = 1`) | 48 | 19 |
| no randomness (a sliding window of next-best columns) | 55 | 33 |
| vessels of 45 / 50 / 60 / 80 / 120 / 200 columns | 38 / 54 / 51 / 45 / 33 / 33 | 14 / 37 / 30 / 18 / 8 / 8 |

The bias toward the current best columns is the engine. Randomness is not needed. Vessel size
has a sweet spot near `1.25 n` to `1.5 n`. Smaller vessels barely exceed `n` and are hard to
solve. Larger ones are the full problem again: a 200-column vessel scores what L1 does. The
cover-versus-success trade-off in [THEORY.md](THEORY.md) predicts this sweet spot.

## Options

- `pandora(X, y, rng, rounds=200, top=n//2, rest=n, decay=0.5)`. The settings were chosen on
  separate tuning problems. Vessels that scale with `n` matter: a fixed 60-column vessel fell
  behind ISD at 20, 60 and 100 measurements. With `decay=0.3`, found by a second tuning search
  over mixed sparsity, the counts on the 100-problem sets at 8 / 11 / 14 / 17 nonzeros were
  99 / 85 / 30 / 4, against 98 / 80 / 27 / 6 for the default. Each difference is within
  sampling noise, so the default stays.
- `atlas(X, y, sigma, rng)` handles noisy measurements of known noise size `sigma`. Vessels are
  solved by lasso, and the certificate becomes a residual test at the noise floor. At noise 0.05,
  11 nonzeros and 40 measurements, it came within 5% of the truth on 25 of 40 problems, against
  9 for lasso followed by least squares. Needs `pip install 'pandora-cs[noisy]'`.
- Baselines are included for comparison: `basis_pursuit`, `iterative_support_detection`,
  `reweighted_l1`. Greedy baselines are in `benchmarks/greedy.py`.

Variants that did not make the cut: solving each vessel with a small Pandora of its own
recovered 82 and 33 of 100 at 11 and 14 nonzeros (default: 80 and 27), at 3.5 to 4 times the
cost. Nesting deeper lost recoveries and cost up to 7 times more.

## Where it came from

Pandora came out of [charon](https://github.com/EgorKhaklin/charon), a study of what
reparameterizing a weight does to gradient descent. charon's
[THEORY.md](https://github.com/EgorKhaklin/charon/blob/main/THEORY.md) proves that a fixed
elementwise reparameterization can at best tie L1 at finding sparse answers. Past that point
the problem is the ranking, and Pandora builds the ranking from many small solves instead
(charon E16 to E18).

**Related work.** Random Lasso (Wang et al., 2011) runs lasso on random column subsets.
Stability selection (Meinshausen and Bühlmann, 2010) scores columns over resamples. Iterative
support detection (Wang and Yin, 2010) and reweighted L1 (Candès, Wakin and Boyd, 2008) refine
a single solve. Pandora's combination, small vessels biased toward the current best columns
with exact-fit stopping certificates, was not found in a literature search.

## Limits

- Synthetic problems only: random matrices, signals with random signs and sizes between 1 and
  2, 30 to 100 problems per cell. Nothing here is tested on real data or a physical measurement
  system.
- The exact certificates need noiseless `y`. `atlas` needs the noise level.
- The measurement savings shrink as signals get less sparse (22% at 5 nonzeros, 11% at 11).
- On a classification task with more samples than features (styx's sparse-input task), Pandora
  picked the same inputs as lasso. It helps when there are fewer measurements than unknowns.
- It costs more linear programs than L1: on 40 × 200 problems, about 30 at 10 nonzeros and 100
  at 12, small ones, against L1's one.
- At a fixed budget of 200 vessels the advantage fades by 1000 unknowns (see Larger problems).
- Independent replication is the next step this needs.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test,bench]'
.venv/bin/python -m pytest -q                         # 9 tests
.venv/bin/python -m benchmarks.run phase              # phase diagram
.venv/bin/python -m benchmarks.run ensembles          # four matrix kinds
.venv/bin/python -m benchmarks.validate baselines     # seven methods, 95% intervals
.venv/bin/python -m benchmarks.validate ablations     # what each part contributes
.venv/bin/python -m benchmarks.measurements           # m50 curves
.venv/bin/python -m benchmarks.ranking                # support inside the top of each ranking
.venv/bin/python -m benchmarks.figures                # figures/ from results/
```

Seeds are fixed in each script, so reruns reproduce the published numbers.

## License

MIT
