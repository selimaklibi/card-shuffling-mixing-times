r"""Gilbert-Shannon-Reeds riffle shuffle: Bayer & Diaconis (1992) exact formula.

After m GSR riffles, a deck arrangement with r rising sequences has probability

    P_m(sigma) = C(2^m + N - r, N) / 2^{mN}.

The number of permutations of N cards with r rising sequences is the Eulerian
number A(N, r - 1), which turns the sum over N! permutations into a sum over
N terms. Everything is computed with exact integer arithmetic.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

import numpy as np


def eulerian_row(n: int) -> list[int]:
    """[A(n, 0), ..., A(n, n-1)]: permutations of n with k descents."""
    row = [1]
    for m in range(2, n + 1):
        new = [0] * m
        for k in range(m):
            a = (k + 1) * row[k] if k < m - 1 else 0
            b = (m - k) * row[k - 1] if k >= 1 else 0
            new[k] = a + b
        row = new
    return row


def rising_sequences(deck) -> int:
    """Number of rising sequences of a deck (cards labelled 0..N-1, top first).

    r = 1 + #{v : card v+1 lies above card v}.
    """
    pos = np.empty(len(deck), dtype=int)
    pos[np.asarray(deck)] = np.arange(len(deck))
    return 1 + int(np.sum(pos[1:] < pos[:-1]))


def riffle_prob(n_cards: int, m: int, r: int) -> Fraction:
    return Fraction(comb(2 ** m + n_cards - r, n_cards), 2 ** (m * n_cards))


def riffle_tv_exact(n_cards: int, m: int) -> float:
    """Exact TV distance to uniform after m riffles (exact rational arithmetic)."""
    n_cards, m = int(n_cards), int(m)
    euler = eulerian_row(n_cards)
    u = Fraction(1, factorial(n_cards))
    tv = sum(euler[r - 1] * abs(riffle_prob(n_cards, m, r) - u) for r in range(1, n_cards + 1))
    return float(tv / 2)


def gsr_riffle(decks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One GSR riffle applied independently to each row of `decks` (vectorised).

    Implemented through the inverse shuffle: give every position a fair bit;
    stable-sorting positions by bit and reading the deck in that order yields
    the inverse riffle, whose inverse is the riffle itself.
    """
    n_decks, n = decks.shape
    bits = rng.integers(0, 2, size=(n_decks, n))
    order = np.argsort(bits, axis=1, kind="stable")  # output positions with bit 0 first
    out = np.empty_like(decks)
    rows = np.arange(n_decks)[:, None]
    out[rows, order] = decks  # top packet -> 0-positions, bottom packet -> 1-positions
    return out
