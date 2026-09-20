"""Checks on the closed-form predictions themselves."""

import numpy as np
import pytest

from viral_roulette import (
    analytical_p_viral, compute_gini, compute_lorenz,
    recovery_time_formula, theoretical_gini_lognormal,
)

BASE_LIKE, BASE_SHARE = 0.1478, 0.0041


def test_p_viral_is_a_probability_and_monotone():
    ps = [analytical_p_viral(q, BASE_LIKE, BASE_SHARE) for q in np.linspace(0.05, 0.99, 25)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:])), "P(viral) must be non-decreasing in q"


def test_trend_alignment_never_hurts():
    """Trend alignment boosts share probability, so it cannot lower P(viral)."""
    for q in [0.3, 0.5, 0.7]:
        off = analytical_p_viral(q, BASE_LIKE, BASE_SHARE, trend_aligned=False)
        on = analytical_p_viral(q, BASE_LIKE, BASE_SHARE, trend_aligned=True)
        assert on >= off - 1e-12


def test_gini_endpoints():
    """Perfect equality gives 0; total concentration approaches 1."""
    assert compute_gini(np.ones(100)) == pytest.approx(0.0, abs=1e-12)
    concentrated = np.zeros(100)
    concentrated[-1] = 1.0
    assert compute_gini(concentrated) == pytest.approx(0.99, abs=0.01)
    assert compute_gini(np.array([])) == 0.0


def test_lorenz_curve_is_anchored_and_below_the_diagonal():
    x, y = compute_lorenz(np.random.default_rng(0).lognormal(size=500))
    assert x[0] == 0 and y[0] == 0
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)
    assert np.all(y <= x + 1e-12)


def test_lognormal_gini_matches_a_sampled_distribution():
    """G = 2*Phi(sigma/sqrt(2)) - 1 should agree with an empirical Gini."""
    sigma = 0.8
    sample = np.random.default_rng(1).lognormal(mean=5.0, sigma=sigma, size=200_000)
    assert theoretical_gini_lognormal(sigma) == pytest.approx(compute_gini(sample), abs=0.01)


def test_recovery_time_matches_the_geometric_decay():
    eta, d0, eps = 0.55, 0.8, 0.05
    k = recovery_time_formula(eta, d0, eps)
    assert k == pytest.approx(3.5, abs=0.1)
    assert d0 * (1 - eta) ** k == pytest.approx(eps, rel=1e-9)
    assert recovery_time_formula(0.0, d0) == np.inf   # no adaptation, never recovers
