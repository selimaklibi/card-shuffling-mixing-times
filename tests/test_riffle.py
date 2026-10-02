import itertools

import numpy as np
import pytest
from math import factorial

from shuffling import ExactChain, riffle_step, tv_distance, eulerian_row, rising_sequences, riffle_tv_exact, gsr_riffle
from shuffling.riffle import riffle_prob


def test_eulerian_rows_sum_to_factorial():
    for n in range(1, 15):
        assert sum(eulerian_row(n)) == factorial(n)


def test_rising_sequence_counts_are_eulerian():
    n = 6
    counts = np.bincount([rising_sequences(p) for p in itertools.permutations(range(n))])
    assert list(counts[1:]) == eulerian_row(n)


@pytest.mark.parametrize("n_cards", [3, 4, 5, 6])
def test_bayer_diaconis_formula_matches_brute_force(n_cards):
    chain = ExactChain(n_cards)
    for m in range(1, 6):
        mu = chain.advance(riffle_step, key=0)
        expected = np.array([float(riffle_prob(n_cards, m, rising_sequences(s))) for s in chain.states])
        np.testing.assert_allclose(mu, expected, atol=1e-14)
        assert tv_distance(mu) == pytest.approx(riffle_tv_exact(n_cards, m), abs=1e-12)


def test_bayer_diaconis_table_for_52_cards():
    published = [1.000, 1.000, 1.000, 1.000, 0.924, 0.614, 0.334, 0.167, 0.085, 0.043]
    ours = [round(riffle_tv_exact(52, m), 3) for m in range(1, 11)]
    assert ours == published


def test_vectorised_gsr_matches_exact_law():
    rng = np.random.default_rng(0)
    n, m, reps = 4, 2, 200_000
    decks = np.tile(np.arange(n), (reps, 1))
    for _ in range(m):
        decks = gsr_riffle(decks, rng)
    r = np.array([rising_sequences(d) for d in decks])
    euler = eulerian_row(n)
    for k in range(1, n + 1):
        p = euler[k - 1] * float(riffle_prob(n, m, k))
        assert np.mean(r == k) == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / reps) + 1e-12)
