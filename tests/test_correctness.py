"""The five correctness tests the model has to pass before any experiment runs.

Each one states an expectation that follows from the model definition, not from
an observed output -- so a failure means the implementation drifted from the
specification, rather than that a number moved.
"""

import numpy as np
import pytest
from scipy import stats

from creator_strategy_sim import Algorithm, Simulation, Video


def _video(quality, trend_aligned=False, like_mult=1.0, share_mult=1.0):
    return Video(
        creator_id=0, topic_idx=0, quality=quality, trend_aligned=trend_aligned,
        topic_multiplier_like=like_mult, topic_multiplier_share=share_mult,
    )


def test_high_quality_goes_viral(algo):
    """A near-perfect, trend-aligned video on a strong topic should clear the
    amplification threshold in the large majority of evaluations."""
    viral = sum(
        algo.evaluate_video(_video(0.99, trend_aligned=True, like_mult=2.0, share_mult=2.0))[2]
        for _ in range(100)
    )
    assert viral > 50, f"expected a majority viral, got {viral}/100"


def test_low_quality_is_suppressed(algo):
    """A near-zero-quality video should essentially never be amplified: the
    skip penalty and low completion rate dominate the score."""
    viral = sum(
        algo.evaluate_video(_video(0.01, like_mult=0.5, share_mult=0.5))[2]
        for _ in range(100)
    )
    assert viral < 10, f"expected near-zero viral, got {viral}/100"


def test_engagement_increases_with_quality(algo):
    """Engagement score must rise monotonically with quality, holding topic and
    trend state fixed. A flat or inverted slope would mean the funnel carries no
    usable quality signal and every later comparison would be meaningless."""
    qualities = np.linspace(0.05, 0.95, 20)
    mean_scores = [
        np.mean([algo.compute_engagement_score(_video(q), 100)[0] for _ in range(30)])
        for q in qualities
    ]
    slope, _, r, _, _ = stats.linregress(qualities, mean_scores)
    assert slope > 0, f"engagement not increasing in quality (slope={slope:.4f})"
    assert r > 0.9, f"relationship too noisy to be a usable signal (R={r:.4f})"


def test_initial_state_is_consistent(stub_calibration):
    """Before any step runs, every creator's history must hold exactly one entry
    equal to its starting follower count."""
    sim = Simulation(strategy="niche", n_creators=10, n_steps=0,
                     calibration=stub_calibration, seed=3)
    sim.run()
    assert all(
        len(c.follower_history) == 1 and c.follower_history[0] == c.followers
        for c in sim.creators
    )


def test_niche_content_vector_never_drifts(stub_calibration):
    """Niche specialists have eta = 0, so their content vector must be bit-stable
    even with a trend live in every single step. This is the invariant that makes
    "niche consistency" a real experimental condition rather than a label."""
    sim = Simulation(strategy="niche", n_creators=5, n_steps=50,
                     calibration=stub_calibration, p_trend=1.0, seed=4)
    before = [c.content_vector.copy() for c in sim.creators]
    sim.run()
    max_drift = max(
        np.max(np.abs(c.content_vector - iv)) for c, iv in zip(sim.creators, before)
    )
    assert max_drift < 1e-9, f"niche content vector drifted by {max_drift:.3e}"


@pytest.mark.parametrize("strategy", ["niche", "trend_chaser", "quality_focused", "random_baseline"])
def test_runs_are_reproducible(stub_calibration, strategy):
    """Two runs with the same seed must be bit-identical, for every strategy.
    Without this, none of the confidence intervals mean anything."""
    a = Simulation(strategy, 8, 15, stub_calibration, seed=11).run()
    b = Simulation(strategy, 8, 15, stub_calibration, seed=11).run()
    np.testing.assert_array_equal(a["final_followers"], b["final_followers"])
