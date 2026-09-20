"""Shared fixtures.

The calibration pipeline reads three files off disk, so it is built once per
test session and reused. Tests that only exercise the funnel use the synthetic
``stub_calibration`` fixture instead and run without any data files at all.
"""

import numpy as np
import pytest

from viral_roulette import build_calibration


@pytest.fixture(scope="session")
def calibration():
    """The real calibration bundle, built from the bundled datasets."""
    return build_calibration()


@pytest.fixture(scope="session")
def stub_calibration():
    """A minimal, data-free calibration bundle for fast structural tests."""
    topics = [f"topic_{i}" for i in range(12)]
    return {
        "profiles": {"real_gini": 0.9086},
        "hashtags": {
            "baseline_like_rate": 0.1478,
            "baseline_share_rate": 0.0041,
            "like_scale": 35.0,
            "share_scale": 21.0,
            "topics": topics,
            "topic_multipliers": {
                t: {"like_multiplier": 1.0, "share_multiplier": 1.0} for t in topics
            },
        },
        "trends": {
            "mu_values": [2.683, 2.061, 2.633, 1.953],
            "sigma_values": [0.349, 0.535, 0.200, 0.123],
        },
        "benchmarks": {
            "tier_dist": {"Nano": 0.887, "Micro": 0.089, "Mid-tier": 0.022,
                          "Macro": 0.002, "Mega": 0.000},
            "tier_ranges": {"Nano": (1_000, 10_000), "Micro": (10_000, 50_000),
                            "Mid-tier": (50_000, 500_000), "Macro": (500_000, 1_000_000),
                            "Mega": (1_000_000, 10_000_000)},
        },
    }


@pytest.fixture
def algo(stub_calibration):
    """An ``Algorithm`` wired to the calibrated baseline rates."""
    from viral_roulette import Algorithm

    cal = stub_calibration["hashtags"]
    return Algorithm(
        baseline_like_rate=cal["baseline_like_rate"],
        baseline_share_rate=cal["baseline_share_rate"],
        like_scale=cal["like_scale"],
        share_scale=cal["share_scale"],
        rng=np.random.default_rng(0),
    )
