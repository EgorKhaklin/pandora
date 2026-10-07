"""Atlas: Pandora for noisy measurements, y = X w + noise of known size sigma.

The vessels are solved by lasso instead of basis pursuit. The certificate becomes a
noise-floor test: take the smallest top-ranked support whose least-squares residual is
no larger than the noise allows. Needs scikit-learn (pip install 'pandora-cs[noisy]').
"""

import numpy as np


def lasso(X, y, cols=None, alpha=1e-3):
    from sklearn.linear_model import Lasso
    d = X.shape[1]
    cols = np.arange(d) if cols is None else np.asarray(cols)
    w = np.zeros(d)
    w[cols] = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000).fit(X[:, cols], y).coef_
    return w


def certify_noisy(X, y, score, sigma, max_s=None, slack=1.5):
    """Smallest top-s support with |residual|^2 <= slack * sigma^2 * (n - s); its least-squares fit."""
    n = X.shape[0]
    max_s = n // 2 if max_s is None else max_s
    order = np.argsort(-score)
    for s in range(1, max_s + 1):
        S = order[:s]
        c, *_ = np.linalg.lstsq(X[:, S], y, rcond=None)
        r = y - X[:, S] @ c
        if r @ r <= slack * sigma**2 * (n - s):
            w = np.zeros(X.shape[1])
            w[S] = c
            return w
    return None


def atlas(X, y, sigma, rng=None, rounds=100, top=None, rest=None, decay=0.7, alpha=1e-3):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    rng = np.random.default_rng(rng)
    top = n // 2 if top is None else top
    rest = min(n if rest is None else rest, d - top)
    score = np.abs(lasso(X, y, alpha=alpha))
    for _ in range(rounds):
        T = np.argsort(-score)[:top]
        R = rng.choice(np.setdiff1d(np.arange(d), T), rest, replace=False)
        score = decay * score + np.abs(lasso(X, y, np.concatenate([T, R]), alpha=alpha))
    return certify_noisy(X, y, score, sigma)
