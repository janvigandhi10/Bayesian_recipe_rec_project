"""Metric helpers shared by the hannah_* notebooks.

Rating metrics (rmse, mae) are re-exported from ``src.evaluation.metrics`` so
there is a single implementation. Interval helpers both return ``(lo, hi)``
tuples. Ranking helpers implement the 1+99 protocol: the rank of the single
positive among [positive] + 99 sampled negatives, with a random tie-break
when an rng is supplied.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .metrics import mae, rmse

__all__ = ["rmse", "mae", "wilson_ci", "mean_ci_normal",
           "rank_of_positive", "hr_at_k", "ndcg_at_k"]


def wilson_ci(k, n, level=0.95):
    """Wilson score interval (lo, hi) for a binomial proportion k/n.

    Returns (0.0, 1.0) when n == 0.
    """
    k = float(k)
    n = float(n)
    if n == 0:
        return (0.0, 1.0)
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def mean_ci_normal(x, level=0.95):
    """Normal-approximation CI (lo, hi) for the mean of x (sample sd, ddof=1).

    n == 0 -> (nan, nan); n == 1 -> degenerate (x0, x0).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    m = float(np.mean(x))
    if n == 1:
        return (m, m)
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return (m - z * se, m + z * se)


def rank_of_positive(pos_score, neg_scores, rng=None):
    """1-indexed rank of the positive among [positive] + negatives (1+99 protocol).

    Ties: with ``rng`` (numpy Generator) the positive is placed uniformly at
    random among the tied scores; without an rng the tie-break is conservative
    (all tied negatives rank ahead of the positive).
    """
    pos_score = float(pos_score)
    neg = np.asarray(neg_scores, dtype=float)
    higher = int(np.sum(neg > pos_score))
    ties = int(np.sum(neg == pos_score))
    if ties:
        if rng is not None:
            higher += int(rng.integers(0, ties + 1))
        else:
            higher += ties
    return higher + 1


def hr_at_k(rank, k=10):
    """Hit rate at k for a 1-indexed rank (scalar or array): 1.0 if rank <= k."""
    r = np.asarray(rank)
    out = (r <= k).astype(float)
    return float(out) if out.ndim == 0 else out


def ndcg_at_k(rank, k=10):
    """NDCG@k for a single positive at a 1-indexed rank (scalar or array).

    With one relevant item, IDCG = 1, so NDCG = 1/log2(rank+1) if rank <= k
    else 0.
    """
    r = np.asarray(rank, dtype=float)
    out = np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0)
    return float(out) if out.ndim == 0 else out
