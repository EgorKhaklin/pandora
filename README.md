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

Fresh random problems, with recovery counted as relative error below 0.01. Baselines: L1
minimization, and iterative support detection (ISD; Wang and Yin, 2010), a published method
that beats L1 in this setting. Output in [results/](results); rerun with the commands below.

![phase diagram](figures/phase.png)

**Phase diagram** (Gaussian matrices, 200 unknowns, 30 problems per point). Pandora recovers as
many problems as ISD or more in all 16 cells, and the gap is widest just past where L1 fails:

| measurements | nonzeros | L1 | ISD | **Pandora** |
|---|---|---|---|---|
| 20 | 5 | 6 | 6 | **17** |
| 40 | 10 | 17 | 21 | **26** |
| 60 | 21 | 5 | 9 | **21** |
| 100 | 45 | 6 | 6 | **18** |

![ensembles](figures/ensembles.png)

**Four kinds of measurement matrix** (40 measurements, 200 unknowns, 40 problems per cell). The
same holds for ±1 entries, rows of a discrete cosine transform, and columns correlated with
their neighbours. At 14 nonzeros Pandora recovers 9 to 13 problems where ISD recovers 2 to 6.
In all 28 benchmark cells Pandora ties or beats ISD.

## How it works

1. Solve basis pursuit on all columns, and score each column by `|w|`.
2. Repeat. A vessel is the `n/2` best-scored columns plus `n` random others. Solve basis pursuit
   on the vessel alone: it has `1.5 n` unknowns instead of `d`, a much easier problem. Update
   `score ← 0.5 · score + |w_vessel|`.
3. Every 5 vessels, run least squares on the `n − 1` best-scored columns. If it fits `y` exactly,
   stop and return it.

**Why an exact fit is a certificate.** Take a fixed set `C` of at most `n − 1` columns that misses
part of the true support. Then `y` can lie in the span of `X_C` only on a set of measure zero.
The columns outside `C` are in general position with respect to that span, and the nonzero
weights on them are continuous. There are finitely many such `C`, so this holds even though `C`
is chosen after seeing `y`. An exact fit on `n − 1` columns therefore means the true support is
inside them, and least squares returns the true `w`. This is a stopping rule built on a known fact
(a sparse vector is generically the unique sparsest solution once `n > k`). It holds for
generic `X` and `w`, and in floating point it uses a tolerance.

**Cost.** L1 is one linear program. Pandora stops at its first certificate, after up to 201. On
the 40-by-200 benchmarks the average was 65 per problem at 11 nonzeros and 162 at 14.

## Options

- `pandora(X, y, rng, rounds=200, top=n//2, rest=n, decay=0.5)`. `top`, `rest` and `decay` were
  chosen on separate tuning problems, never on the benchmark problems above. Vessels that scale
  with `n` matter: with a fixed 60-column vessel, Pandora fell behind ISD at 20, 60 and 100
  measurements.
- `atlas(X, y, sigma, rng)` handles noisy measurements of known noise size `sigma`. Vessels are
  solved by lasso, and the certificate becomes a residual test at the noise floor. At noise 0.05,
  11 nonzeros and 40 measurements, it came within 5% of the truth on 25 of 40 problems, against
  9 for lasso followed by least squares. Needs `pip install 'pandora-cs[noisy]'`.
- Baselines are included for comparison: `basis_pursuit`, `iterative_support_detection`,
  `reweighted_l1`.

A heavier variant solves each vessel with a small Pandora of its own. It recovered 82 and 33 of
100 problems at 11 and 14 nonzeros, against 80 and 27 for the default, at 3.5 to 4 times the
cost, so it is not included. Nesting a further level added cost and no recoveries.

## Where it came from

Pandora came out of [charon](https://github.com/EgorKhaklin/charon), a study of what
reparameterizing a weight does to gradient descent. charon's
[THEORY.md](https://github.com/EgorKhaklin/charon/blob/main/THEORY.md) proves that a fixed
elementwise reparameterization can at best tie L1 at finding sparse answers. Past that point
the problem is the ranking: the true support is not near the top of any single solve. Pandora
builds the ranking from many small randomized solves instead (charon E16 to E18).

**Related work.** Random Lasso (Wang et al., 2011) runs lasso on random column subsets.
Stability selection (Meinshausen and Bühlmann, 2010) scores columns over resamples. Iterative
support detection (Wang and Yin, 2010) and reweighted L1 (Candès, Wakin and Boyd, 2008) refine
a single solve. Pandora's combination, small vessels biased toward the current best columns
with an exact-fit stopping certificate, was not found in a literature search.

## Limits

- Synthetic problems only: random matrices with 200 unknowns, signals with random signs and
  sizes between 1 and 2, 30 to 40 problems per cell. Nothing here is tested on real data.
- The exact certificate needs noiseless `y`. `atlas` needs the noise level.
- On a classification task with more samples than features (styx's sparse-input task), Pandora
  picked the same inputs as lasso. It helps when there are fewer measurements than unknowns.
- It costs up to 200 times as many linear programs as L1.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test,bench]'
.venv/bin/python -m pytest -q                         # 9 tests
.venv/bin/python -m benchmarks.run phase              # ~25 min, results/phase.json
.venv/bin/python -m benchmarks.run ensembles          # ~15 min, results/ensembles.json
.venv/bin/python -m benchmarks.figures                # figures/ from results/
```

## License

MIT
