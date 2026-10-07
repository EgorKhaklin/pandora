# Why Pandora works

Notation: `y = X w*`, `X` is `n × d` with `n < d`, and `w*` has support `S` with `|S| = k`. A vessel
`V` is a set of columns with `|V| > n`. `BP(V)` is basis pursuit restricted to `V`:
`min ‖w‖₁ s.t. X_V w = y`. "Generic" means the statement holds with probability one when `X` has
a continuous distribution (Gaussian, for example) and the nonzero values of `w*` are drawn from
a continuous distribution.

## 1. The certificates

**Lemma 1 (exact fit on few columns).** Generically, if a set `C` with `|C| ≤ n − 1` has `y` in
the span of `X_C`, then `S ⊆ C`, and least squares on `C` returns `w*`.

*Proof.* `X_C` has full column rank, so least squares on `C` is unique. Suppose `S ⊄ C`. Split
`w*` into its part `a` on `S ∩ C` and its part `b ≠ 0` on `S \ C`. Then `y ∈ span(X_C)` exactly
when `X_{S∖C} b ∈ span(X_C)`. That set of `b` is a proper linear subspace, because `span(X_C)` has
dimension at most `n − 1` and the columns outside `C` are in general position with respect to it.
A continuous `b` avoids it with probability one. There are finitely many sets `C`, so this holds
even for a `C` chosen after seeing `y`. ∎

**Corollary 2 (a sparse vessel is a recovery).** Generically, if `BP(V)` returns a solution `w`
with fewer than `n` nonzeros, then `w = w*`.

*Proof.* `w` fits `y` exactly on `C = supp(w)`, and `|C| ≤ n − 1`. Apply Lemma 1. ∎

Basis pursuit's optimal solutions are generically at vertices with exactly `n` nonzeros unless
they are sparser. So the vessels that come back sparse are exactly the ones that recovered `w*`.

## 2. When a vessel recovers `w*`

**Proposition 3.** If `S ⊆ V` and `w*` is the unique solution of `BP(V)`, then `BP(V)` returns
`w*`, and by Corollary 2 the round that solves `V` ends Pandora with `w*`.

So Pandora succeeds as soon as some vessel (i) **covers** `S` and (ii) **succeeds**: basis
pursuit on that vessel's columns recovers `w*`.

**(ii) Success.** `X_V` is an `n × |V|` matrix. For Gaussian `X`, whether basis pursuit recovers a
`k`-sparse vector is governed by the Donoho–Tanner phase transition at undersampling `n/|V|` and
sparsity `k/n`. With `|V| = 1.5 n` the undersampling is `2/3`, far easier than the full
problem's `n/d` (`0.2` in the benchmarks). Vessels are small, so they succeed at sparsity levels
where `BP` on all `d` columns fails.

**(i) Cover.** Each vessel is `T ∪ R`, where `T` holds the `top` best-scored columns and `R` holds
`rest` columns drawn at random from the remaining `d − top`. If `T` contains all but `j` of the
true columns, the probability that `R` holds the missing `j` is

```
p_j = C(d − top − j, rest − j) / C(d − top, rest)  ≈  (rest / (d − top))^j
```

which is about `0.22^j` at `n = 40`, `d = 200`. Over `R` independent rounds with the same `j`,
the chance that no vessel both covers and succeeds is at most `(1 − p_j · q)^R`, where `q` is
the success probability in (ii).

**The trade-off in vessel size.** A larger `rest` raises `p_j`, the chance to cover. It also
lowers the undersampling `n/|V|`, which lowers `q`. The best vessel size balances the two. In
the ablations (`n = 40`, 10 nonzeros), recovery peaks at 50 to 60 columns, then falls toward L1's
level as the vessel grows to all 200 columns.

## 3. Why the ranking matters: what L1 cannot see

An optimal basic solution of `BP` on all `d` columns has at most `n` nonzeros. Every other
column gets the value 0 and cannot be ranked at all. If L1 sets a true column to zero, no
ranking built from L1's solution puts that column above the zeros. The ranking benchmark shows
this. L1's top 20 and top 39 contain the true support equally often (33 and 33 of 60 at 10
nonzeros), because past its nonzeros the ranking is arbitrary. Pandora's vessels re-solve
smaller problems in which such a column can come back as nonzero. Its score then rises into
`T`, and `j`, the number of true columns outside `T`, falls. The median true column ranks the
same under L1 and Pandora. The whole difference is in the last few true columns.

## 4. What is proved and what is observed

| statement | status |
|---|---|
| Lemma 1, Corollary 2, Proposition 3 | proved, for generic `X` and `w*` |
| a vessel covering `S` succeeds with the Donoho–Tanner probability at undersampling `n/|V|` | known for a vessel fixed in advance and Gaussian `X` (Donoho and Tanner, 2009). Pandora picks vessels using earlier solves, and that adaptivity is not covered |
| the cover probability `p_j` for a fixed `j` | proved (a counting argument) |
| how fast `j` falls over rounds | **observed, not proved**: the score dynamics are the open part |
| recovery ≈ support contained in Pandora's top `n − 1` | observed (benchmarks, `results/validate_mechanism.json`) |

A full guarantee needs a bound on how the scores move true columns into `T`. That is the open
problem.
