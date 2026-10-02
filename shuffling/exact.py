"""Brute-force evolution of a shuffle's law on the full symmetric group S_N.

Feasible for N <= 8 (8! = 40 320 states). Used as ground truth to validate
the closed-form results in :mod:`shuffling.top_to_random` and
:mod:`shuffling.riffle`.

Convention: a deck is a tuple ``d`` where ``d[p]`` is the label of the card at
position ``p`` (0 = top). The initial deck is the identity ``(0, 1, ..., N-1)``.
"""
from __future__ import annotations

import itertools
from math import factorial
from typing import Callable, Iterable

import numpy as np

Deck = tuple[int, ...]
# A step maps a deck to an iterable of (next_deck, probability)
Step = Callable[[Deck], Iterable[tuple[Deck, float]]]


def top_to_random_step(deck: Deck) -> Iterable[tuple[Deck, float]]:
    """Take the top card and insert it uniformly into one of the N positions."""
    n = len(deck)
    rest = list(deck[1:])
    for k in range(n):
        new = rest.copy()
        new.insert(k, deck[0])
        yield tuple(new), 1.0 / n


def variant_step(t: int) -> Step:
    """Variant of the report (section 5): at step t (1-indexed) the top card is
    inserted uniformly into one of the last min(t, N) positions."""
    def step(deck: Deck) -> Iterable[tuple[Deck, float]]:
        n = len(deck)
        m = min(t, n)
        rest = list(deck[1:])
        for k in range(n - m, n):
            new = rest.copy()
            new.insert(k, deck[0])
            yield tuple(new), 1.0 / m
    return step


def riffle_step(deck: Deck) -> Iterable[tuple[Deck, float]]:
    """Gilbert-Shannon-Reeds riffle shuffle.

    Equivalent description: draw i.i.d. fair bits b_0..b_{N-1} for the output
    positions; with k = #zeros, the top k cards fill the 0-positions and the
    bottom N-k cards fill the 1-positions, each packet keeping its order.
    Every bit string has probability 2^-N.
    """
    n = len(deck)
    p = 0.5 ** n
    for bits in itertools.product((0, 1), repeat=n):
        k = n - sum(bits)
        a, b = iter(deck[:k]), iter(deck[k:])
        yield tuple(next(b) if x else next(a) for x in bits), p


class ExactChain:
    """Exact law mu_t of the deck after t shuffles, stored as a vector on S_N."""

    def __init__(self, n: int):
        if n > 8:
            raise ValueError("exact enumeration is limited to N <= 8")
        self.n = n
        self.states = list(itertools.permutations(range(n)))
        self.index = {s: i for i, s in enumerate(self.states)}
        self.mu = np.zeros(len(self.states))
        self.mu[self.index[tuple(range(n))]] = 1.0
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _kernel(self, step: Step, key: int | None):
        if key is not None and key in self._cache:
            return self._cache[key]
        rows, cols, vals = [], [], []
        for i, s in enumerate(self.states):
            for nxt, p in step(s):
                rows.append(i)
                cols.append(self.index[nxt])
                vals.append(p)
        ker = (np.array(rows), np.array(cols), np.array(vals))
        if key is not None:
            self._cache[key] = ker
        return ker

    def advance(self, step: Step, key: int | None = None) -> np.ndarray:
        rows, cols, vals = self._kernel(step, key)
        new = np.zeros_like(self.mu)
        np.add.at(new, cols, self.mu[rows] * vals)
        self.mu = new
        return self.mu


def tv_distance(mu: np.ndarray) -> float:
    """Total-variation distance to the uniform law on S_N."""
    return 0.5 * float(np.abs(mu - 1.0 / mu.size).sum())


def separation(mu: np.ndarray) -> float:
    """Separation distance  s = max_sigma (1 - N! mu(sigma))."""
    return float(1.0 - mu.size * mu.min())


def uniform_size(n: int) -> int:
    return factorial(n)
