"""Closed-form predictions used as a benchmark for the Monte Carlo results.

These are not a substitute for the simulation. They are the thing the
simulation gets checked against -- and where simulation and theory disagree,
the disagreement is itself a result (see the virality experiment).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # numerically stable standard normal CDF

from .algorithm import Algorithm
from .config import N_TEST_USERS, STRATEGY_PARAMS

_W = {
    "completion": Algorithm.W_COMPLETION,
    "share": Algorithm.W_SHARE,
    "like": Algorithm.W_LIKE,
    "skip": Algorithm.W_SKIP,
}


def analytical_p_viral(
    quality: float,
    baseline_like: float,
    baseline_share: float,
    topic_like_mult: float = 1.0,
    topic_share_mult: float = 1.0,
    trend_aligned: bool = False,
    threshold: float = 0.35,
    n_users: int = 100,
) -> float:
    """Normal approximation to :math:`P(E > \\theta)` for a single video.

    Each engagement component is a scaled Binomial, so

    .. math::
       \\mathbb{E}[\\hat p_j] = p_j, \\qquad
       \\mathrm{Var}(\\hat p_j) = \\frac{p_j(1-p_j)}{n},

    and, treating viewers as independent,
    :math:`\\mathrm{Var}(E) = \\sum_j w_j^2 \\mathrm{Var}(\\hat p_j)`. With
    ``n = 100`` a normal approximation to the weighted sum is reasonable, giving

    .. math::
       P(\\text{viral}) \\approx 1 - \\Phi\\!\\left(
           \\frac{\\theta - \\mathbb{E}[E]}{\\sqrt{\\mathrm{Var}(E)}}\\right).

    The independence assumption is exactly what makes this a *lower* bound for
    trend-sensitive strategies: it cannot see the share-probability boost that
    trend alignment supplies.
    """
    q = quality

    p_complete = 1.0 / (1.0 + np.exp(-6.0 * (q - 0.5)))
    p_like = float(np.clip(q * topic_like_mult * baseline_like * 8, 0, 0.95))
    trend_boost = 1.4 if trend_aligned else 1.0
    p_share = float(np.clip(q * topic_share_mult * baseline_share * 6 * trend_boost, 0, 0.95))
    p_skip = float(np.clip(1.0 - q * 0.9, 0.05, 0.95))

    E_E = (
        _W["completion"] * p_complete
        + _W["share"] * p_share
        + _W["like"] * p_like
        - _W["skip"] * p_skip
    )

    Var_E = (
        _W["completion"] ** 2 * p_complete * (1 - p_complete) / n_users
        + _W["share"] ** 2 * p_share * (1 - p_share) / n_users
        + _W["like"] ** 2 * p_like * (1 - p_like) / n_users
        + _W["skip"] ** 2 * p_skip * (1 - p_skip) / n_users
    )

    if Var_E <= 0:
        return 1.0 if E_E > threshold else 0.0

    return float(1.0 - ndtr((threshold - E_E) / np.sqrt(Var_E)))


def analytical_growth_rate(
    strategy: str,
    baseline_like: float,
    baseline_share: float,
    p_trend: float = 0.12,
) -> float:
    """Expected new followers per week for a strategy, from a zero-stock baseline.

    .. math::
       \\mathbb{E}[\\Delta F] \\approx r \\cdot P(\\text{viral} \\mid q)
       \\cdot \\mathbb{E}[\\text{impressions} \\mid \\text{viral}] \\cdot \\rho

    This under-predicts the simulation by two orders of magnitude, and that gap
    is informative rather than a bug: the simulation initialises creators with a
    realistic follower stock, so its final means are governed by the balance
    between growth and churn on that stock, not by growth from zero.
    """
    q_mu = STRATEGY_PARAMS[strategy][0]
    tau = STRATEGY_PARAMS[strategy][3]
    post_rate = STRATEGY_PARAMS[strategy][4]

    p_viral_base = analytical_p_viral(q_mu, baseline_like, baseline_share, trend_aligned=False)
    p_viral_trend = analytical_p_viral(q_mu, baseline_like, baseline_share, trend_aligned=True)

    prob_trend_video = p_trend * tau
    p_viral = (1 - prob_trend_video) * p_viral_base + prob_trend_video * p_viral_trend

    # Approximate E[E | E > theta] by the midpoint of the admissible range.
    avg_E_given_viral = 0.35 + (1.0 - 0.35) / 2
    avg_growth_factor = 1.0 + 8.0 * (avg_E_given_viral - 0.35)
    avg_impressions = N_TEST_USERS * avg_growth_factor

    conversion = 0.005
    return post_rate * p_viral * avg_impressions * conversion


def theoretical_gini_lognormal(sigma_logF: float) -> float:
    """Gini of a log-normal follower distribution.

    Follower growth here is multiplicative, so :math:`\\log F(T)` is a sum of
    weakly dependent increments and is approximately normal by a CLT argument.
    For :math:`F \\sim \\mathrm{LogNormal}(\\mu, \\sigma^2)` the Gini has the
    closed form :math:`G = 2\\Phi(\\sigma/\\sqrt 2) - 1`, i.e. it depends on the
    log-scale dispersion alone.
    """
    return float(2.0 * ndtr(sigma_logF / np.sqrt(2)) - 1.0)


def recovery_time_formula(eta: float, d0: float, epsilon: float = 0.05) -> float:
    """Weeks for a trend-adapted content vector to drift back to baseline.

    Residual mismatch decays geometrically, :math:`d(k) = d_0 (1-\\eta)^k`, so

    .. math:: k^\\star = \\frac{\\log(\\varepsilon / d_0)}{\\log(1 - \\eta)}.

    At :math:`\\eta = 0.55` and :math:`d_0 = 0.8` this is about 3.5 weeks --
    which is why aggressive trend adaptation is not as costly as it looks.
    """
    if eta <= 0 or eta >= 1:
        return np.inf if eta <= 0 else 0.0
    if d0 <= epsilon:
        return 0.0
    return float(np.log(epsilon / d0) / np.log(1.0 - eta))
