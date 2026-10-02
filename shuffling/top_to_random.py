r"""Closed-form mixing quantities for the top-to-random shuffle, valid for any N.

Key reduction
-------------
The inverse of "top card -> uniform position" is "uniform card -> top"
(random-to-top). Distance to uniformity (TV or separation) is invariant under
sigma -> sigma^{-1}, so we may study random-to-top. After n random-to-top moves
the deck is: the D_n distinct cards that were touched, in order of last touch,
followed by the untouched cards in their original relative order. Hence

    mu_n(sigma) = (1/N!) * sum_{j >= N - s(sigma)} P(D_n = j) (N - j)!

where s(sigma) is the length of the longest bottom block of sigma whose cards
appear in increasing original order and D_n is the number of distinct values
in n uniform draws from {1..N} (occupancy / coupon-collector law).

This gives the *exact* total-variation curve for N = 52 in milliseconds, and
the exact separation  s(n) = P(D_n <= N - 2).

The report's strong stationary time (bottom card reaches the top, then is
re-inserted) satisfies T = sum_{i=1}^{N} Geom(i/N), i.e. P(T > n) = P(D_n < N),
E[T] = N H_N. It bounds the separation but is not sharp: the sharp time is a
coupon collector for N - 1 coupons (mean N (H_N - 1)).
"""
from __future__ import annotations

from math import lgamma

import numpy as np


def distinct_draws_pmf(n_cards: int, n_draws: int) -> np.ndarray:
    """Law of D_n, the number of distinct values in `n_draws` uniform draws."""
    pmf = np.zeros(n_cards + 1)
    pmf[0] = 1.0
    j = np.arange(n_cards + 1)
    stay, move = j / n_cards, (n_cards - j) / n_cards
    for _ in range(n_draws):
        new = pmf * stay
        new[1:] += pmf[:-1] * move[:-1]
        pmf = new
    return pmf


def distinct_draws_pmf_path(n_cards: int, n_max: int) -> np.ndarray:
    """Array of shape (n_max+1, N+1): row n is the law of D_n."""
    out = np.zeros((n_max + 1, n_cards + 1))
    pmf = np.zeros(n_cards + 1)
    pmf[0] = 1.0
    j = np.arange(n_cards + 1)
    stay, move = j / n_cards, (n_cards - j) / n_cards
    out[0] = pmf
    for n in range(1, n_max + 1):
        new = pmf * stay
        new[1:] += pmf[:-1] * move[:-1]
        pmf = new
        out[n] = pmf
    return out


def sst_tail(n_cards: int, n_max: int) -> np.ndarray:
    """P(T > n), n = 0..n_max, for the strong stationary time of the report."""
    path = distinct_draws_pmf_path(n_cards, n_max)
    return 1.0 - path[:, n_cards]


def sst_mean(n_cards: int) -> float:
    """E[T] = N * H_N (coupon collector)."""
    return n_cards * float(np.sum(1.0 / np.arange(1, n_cards + 1)))


def separation_exact(n_cards: int, n_max: int) -> np.ndarray:
    """Exact separation distance s(n) = P(D_n <= N - 2), n = 0..n_max."""
    path = distinct_draws_pmf_path(n_cards, n_max)
    return path[:, : n_cards - 1].sum(axis=1)


def tv_exact(n_cards: int, n_max: int) -> np.ndarray:
    """Exact total-variation distance to uniform after n = 0..n_max shuffles.

    TV(n) = 1/2 * sum_s w(s) |a_n(s) - 1|, with w(s) = #{sigma: s(sigma)=s}/N!
    and a_n(s) = sum_{j >= N-s} P(D_n = j)(N-j)!.  The products w(s) a_n(s)
    are formed in log space (N can be in the thousands).
    """
    N = n_cards
    path = distinct_draws_pmf_path(N, n_max)
    j = np.arange(N + 1)
    log_fact_rest = np.array([lgamma(N - jj + 1) for jj in j])   # log (N-j)!
    s = np.arange(1, N + 1)
    log_w = -np.array([lgamma(k + 1) for k in s]) + np.log(s / (s + 1.0))  # 1/s! - 1/(s+1)!
    log_w[-1] = -lgamma(N + 1)                                              # s = N: identity block only
    w = np.exp(log_w)
    tv = np.empty(n_max + 1)
    with np.errstate(divide="ignore"):
        log_path = np.log(path)
    for n in range(n_max + 1):
        log_terms = log_path[n] + log_fact_rest
        log_a = np.logaddexp.accumulate(log_terms[::-1])[s]   # log a_n(s)
        tv[n] = 0.5 * float(np.sum(np.abs(np.exp(log_w + log_a) - w)))
    return tv
