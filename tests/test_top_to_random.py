import numpy as np
import pytest

from shuffling import (ExactChain, top_to_random_step, variant_step, tv_distance, separation,
                       tv_exact, separation_exact, sst_tail, sst_mean, distinct_draws_pmf)


@pytest.mark.parametrize("n_cards", [3, 4, 5, 6])
def test_closed_form_tv_and_separation_match_brute_force(n_cards):
    n_max = 4 * n_cards
    chain = ExactChain(n_cards)
    tv_cf, sep_cf = tv_exact(n_cards, n_max), separation_exact(n_cards, n_max)
    for n in range(1, n_max + 1):
        mu = chain.advance(top_to_random_step, key=0)
        assert tv_distance(mu) == pytest.approx(tv_cf[n], abs=1e-12)
        assert separation(mu) == pytest.approx(sep_cf[n], abs=1e-12)


@pytest.mark.parametrize("n_cards", [4, 5, 6])
def test_strong_stationary_time_bounds_separation_and_tv(n_cards):
    n_max = 6 * n_cards
    tail, sep, tv = sst_tail(n_cards, n_max), separation_exact(n_cards, n_max), tv_exact(n_cards, n_max)
    assert np.all(tv <= sep + 1e-12)
    assert np.all(sep <= tail + 1e-12)


def test_sst_mean_matches_tail_sum():
    n = 52
    tail = sst_tail(n, 5000)
    assert tail.sum() == pytest.approx(sst_mean(n), rel=1e-9)  # E[T] = sum_n P(T > n)


def test_distinct_draws_pmf_is_a_probability():
    pmf = distinct_draws_pmf(52, 100)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf.min() >= 0


@pytest.mark.parametrize("n_cards", [3, 4, 5, 6, 7])
def test_variant_is_exactly_uniform_after_n_steps(n_cards):
    chain = ExactChain(n_cards)
    tvs = [tv_distance(chain.advance(variant_step(t))) for t in range(1, n_cards + 1)]
    assert tvs[-2] > 0.0           # not yet uniform after N - 1 steps
    assert tvs[-1] == pytest.approx(0.0, abs=1e-12)  # perfectly uniform after N steps
