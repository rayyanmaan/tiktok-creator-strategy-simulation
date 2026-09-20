"""One Monte Carlo trial: N creators, T weeks, one strategy."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .algorithm import Algorithm
from .config import P_TREND, WEEKLY_CHURN
from .creator import Creator
from .metrics import compute_gini
from .trends import TrendEngine


def sample_initial_followers(
    rng: np.random.Generator,
    tier_dist: Dict[str, float],
    tier_ranges: Dict[str, Tuple[int, int]],
) -> int:
    """Draw a starting follower count from the real creator-tier distribution.

    Tier probabilities come from the industry benchmark report; the count is
    then uniform inside the chosen tier's range. Starting everyone at zero would
    have been simpler but would have thrown away the single most important fact
    about the creator population: almost all of it is nano-tier.
    """
    tiers = list(tier_dist)
    probs = np.array([tier_dist[t] for t in tiers], dtype=float)
    probs = probs / probs.sum()
    chosen_tier = tiers[rng.choice(len(tiers), p=probs)]
    lo, hi = tier_ranges[chosen_tier]
    return int(rng.integers(lo, hi + 1))


class Simulation:
    """A single run of the creator ecosystem under one strategy.

    Parameters
    ----------
    strategy:
        Strategy applied to every creator in this run.
    n_creators, n_steps:
        Population size and number of weekly steps.
    calibration:
        Bundle with ``profiles`` / ``hashtags`` / ``trends`` / ``benchmarks``
        sub-dicts, as produced by :mod:`viral_roulette.calibration`.
    p_trend:
        Probability of a new trend spawning per step.
    seed:
        Seed for this trial. Creators get derived seeds so that two runs with
        the same seed are bit-identical.
    """

    def __init__(
        self,
        strategy: str,
        n_creators: int,
        n_steps: int,
        calibration: Dict,
        p_trend: float = P_TREND,
        seed: int = 0,
    ) -> None:
        self.strategy = strategy
        self.n_creators = n_creators
        self.n_steps = n_steps
        self.p_trend = p_trend
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        cal_h = calibration["hashtags"]
        cal_t = calibration["trends"]
        cal_b = calibration["benchmarks"]

        self.topics = cal_h["topics"]
        n_topics = len(self.topics)

        self.topic_mult_like = np.array([
            cal_h["topic_multipliers"].get(t, {}).get("like_multiplier", 1.0)
            for t in self.topics
        ])
        self.topic_mult_share = np.array([
            cal_h["topic_multipliers"].get(t, {}).get("share_multiplier", 1.0)
            for t in self.topics
        ])

        self.creators: List[Creator] = [
            Creator(
                creator_id=i,
                strategy=strategy,
                initial_followers=sample_initial_followers(
                    self.rng, cal_b["tier_dist"], cal_b["tier_ranges"]
                ),
                topics=self.topics,
                topic_multipliers_like=self.topic_mult_like,
                topic_multipliers_share=self.topic_mult_share,
                rng=np.random.default_rng(seed * 10000 + i),
            )
            for i in range(n_creators)
        ]

        self.algorithm = Algorithm(
            baseline_like_rate=cal_h["baseline_like_rate"],
            baseline_share_rate=cal_h["baseline_share_rate"],
            like_scale=cal_h["like_scale"],
            share_scale=cal_h["share_scale"],
            rng=self.rng,
        )

        self.trend_engine = TrendEngine(
            p_trend=p_trend,
            mu_values=cal_t["mu_values"],
            sigma_values=cal_t["sigma_values"],
            n_topics=n_topics,
            rng=self.rng,
        )

    def run(self) -> Dict:
        """Execute ``n_steps`` weeks and return every metric the experiments need.

        Each week: the trend engine advances, creators publish, the recommender
        scores and amplifies, followers are updated net of churn, and adapting
        creators drift their content vector toward the live trend.
        """
        initial_followers = np.array([c.followers for c in self.creators])

        for t in range(self.n_steps):
            trend_active, trend_topic_idx, trend_intensity = self.trend_engine.step(t)

            for creator in self.creators:
                step_delta_f = 0.0
                step_viral = False

                for video in creator.produce_video(
                    t, trend_active, trend_topic_idx, trend_intensity
                ):
                    _E, impressions, went_viral = self.algorithm.evaluate_video(video)
                    step_delta_f += self.algorithm.compute_follower_delta(impressions)
                    step_viral = step_viral or went_viral

                # Baseline audience decay, applied to the whole follower stock.
                decay = creator.followers * WEEKLY_CHURN
                creator.receive_followers(step_delta_f - decay, t, step_viral)

                if trend_active:
                    creator.update_content_vector(trend_topic_idx, trend_intensity)

        final_followers = np.array([c.followers for c in self.creators])
        viral_counts = np.array([c.viral_events for c in self.creators])
        follower_traj = np.array([c.follower_history for c in self.creators])
        survival_flags = (final_followers > initial_followers).astype(int)

        total_videos = np.array([c.posting_rate * self.n_steps for c in self.creators])
        viral_rates = viral_counts / np.maximum(total_videos, 1)

        return {
            "strategy": self.strategy,
            "final_followers": final_followers,
            "initial_followers": initial_followers,
            "follower_traj": follower_traj,
            "viral_counts": viral_counts,
            "viral_rates": viral_rates,
            "survival_flags": survival_flags,
            "gini": compute_gini(final_followers),
            "trend_history": self.trend_engine.history,
        }
