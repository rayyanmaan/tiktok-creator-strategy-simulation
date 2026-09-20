"""The two-stage recommendation funnel.

Stage 1 shows every video to a small random cohort regardless of who made it.
Stage 2 measures engagement in that cohort and either amplifies the video or
drops it. This is the whole feedback loop -- deliberately minimal, so that what
the experiments measure is the loop itself and not a pile of ranking heuristics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .creator import Video


class Algorithm:
    """Threshold-amplification recommender.

    Parameters
    ----------
    baseline_like_rate, baseline_share_rate:
        Cross-topic average engagement rates measured from the hashtag dataset.
    like_scale, share_scale:
        Scaling factors solved for during calibration so that an average video
        (``q = 0.5``) scores just below the amplification threshold.
    expansion_threshold:
        Engagement score above which a video is amplified.
    growth_alpha:
        Slope of the impression expansion above the threshold.
    conversion_rate:
        Fraction of impressions that convert into new followers.
    n_test_users:
        Mean size of the Stage-1 cohort (Poisson-distributed per video).
    """

    # Score weights: completion > share > like, skips subtract.
    # These are architectural choices motivated by watch-time priority, not
    # fitted parameters, and are held fixed across every experiment.
    W_COMPLETION = 0.40
    W_SHARE = 0.25
    W_LIKE = 0.20
    W_SKIP = 0.15

    def __init__(
        self,
        baseline_like_rate: float,
        baseline_share_rate: float,
        like_scale: float = 40.0,
        share_scale: float = 24.0,
        expansion_threshold: float = 0.35,
        growth_alpha: float = 8.0,
        conversion_rate: float = 0.02,
        n_test_users: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.baseline_like_rate = baseline_like_rate
        self.baseline_share_rate = baseline_share_rate
        self.expansion_threshold = expansion_threshold
        self.growth_alpha = growth_alpha
        self.conversion_rate = conversion_rate
        self.n_test_users = n_test_users
        self.rng = rng if rng is not None else np.random.default_rng()
        self.like_scale = like_scale
        self.share_scale = share_scale

    # -- stage 1 ------------------------------------------------------------

    def _sample_user_responses(self, video: Video, n_users: int) -> Dict[str, float]:
        """Sample how ``n_users`` viewers react, returning empirical rates.

        Each of the four behaviours is an independent Bernoulli draw per viewer
        whose probability depends on video quality and the topic multipliers.
        """
        q = video.quality

        # Completion rises smoothly with quality.
        p_complete = 1.0 / (1.0 + np.exp(-6.0 * (q - 0.5)))

        p_like = float(np.clip(
            q * video.topic_multiplier_like * self.baseline_like_rate * self.like_scale,
            0.0, 0.95,
        ))

        # Sharing is rarer than liking, and a live trend makes it easier.
        trend_boost = 1.4 if video.trend_aligned else 1.0
        p_share = float(np.clip(
            q * video.topic_multiplier_share * self.baseline_share_rate
            * self.share_scale * trend_boost,
            0.0, 0.95,
        ))

        p_skip = float(np.clip(1.0 - q * 0.9, 0.05, 0.95))

        return {
            "completion_rate": float(self.rng.binomial(n_users, p_complete)) / n_users,
            "like_rate": float(self.rng.binomial(n_users, p_like)) / n_users,
            "share_rate": float(self.rng.binomial(n_users, p_share)) / n_users,
            "skip_rate": float(self.rng.binomial(n_users, p_skip)) / n_users,
        }

    def compute_engagement_score(self, video: Video, n_users: int) -> Tuple[float, Dict]:
        """Engagement score ``E`` of a video in a cohort of ``n_users``.

        .. math::
           E = 0.40\\,\\hat p_{\\text{complete}} + 0.25\\,\\hat p_{\\text{share}}
             + 0.20\\,\\hat p_{\\text{like}} - 0.15\\,\\hat p_{\\text{skip}}

        ``E`` is a random variable: the hats are empirical rates in a finite
        cohort, which is exactly where the exposure luck enters the model.
        """
        rates = self._sample_user_responses(video, n_users)
        E = (
            self.W_COMPLETION * rates["completion_rate"]
            + self.W_SHARE * rates["share_rate"]
            + self.W_LIKE * rates["like_rate"]
            - self.W_SKIP * rates["skip_rate"]
        )
        return float(E), rates

    # -- stage 2 ------------------------------------------------------------

    def expand_or_suppress(self, initial_impressions: int, E: float) -> Tuple[int, bool]:
        """Amplify above the threshold, otherwise stop at the test cohort.

        .. math:: \\text{impressions} = \\text{initial}\\,(1 + \\alpha (E - \\theta))
        """
        if E <= self.expansion_threshold:
            return initial_impressions, False

        growth_factor = 1.0 + self.growth_alpha * (E - self.expansion_threshold)
        return int(initial_impressions * growth_factor), True

    def compute_follower_delta(self, total_impressions: int) -> float:
        """Convert impressions into new followers at a fixed conversion rate."""
        return total_impressions * self.conversion_rate

    def evaluate_video(self, video: Video) -> Tuple[float, int, bool]:
        """Run one video through the whole funnel.

        Returns ``(engagement_score, total_impressions, went_viral)``.
        """
        # The Poisson draw is the irreducible exposure luck: how many people
        # happened to be served the video in its first hour.
        initial_impressions = max(int(self.rng.poisson(self.n_test_users)), 1)

        E, _ = self.compute_engagement_score(video, initial_impressions)
        total_impressions, went_viral = self.expand_or_suppress(initial_impressions, E)
        return E, total_impressions, went_viral
