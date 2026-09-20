"""The creator agent.

A :class:`Creator` owns a fixed strategy, a fixed production quality, and a
content vector over topics. Quality is drawn once at initialisation and never
updated inside a run -- that is a deliberate modelling choice so that strategy
effects are not confounded with learning effects.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List

import numpy as np

from .config import STRATEGY_PARAMS

#: One published video. Carries everything the recommender needs to score it.
Video = namedtuple("Video", [
    "creator_id",
    "topic_idx",               # index into the topic list
    "quality",                 # per-video draw around the creator's baseline
    "trend_aligned",           # True if posted on the live trend topic
    "topic_multiplier_like",   # from the hashtag calibration
    "topic_multiplier_share",  # from the hashtag calibration
])


class Creator:
    """A content creator following one fixed strategy.

    Parameters
    ----------
    creator_id:
        Unique identifier within a run.
    strategy:
        One of the keys of :data:`~creator_strategy_sim.config.STRATEGY_PARAMS`.
    initial_followers:
        Starting follower count, sampled from the real tier distribution.
    topics:
        Available content topics, taken from the hashtag dataset.
    topic_multipliers_like, topic_multipliers_share:
        Per-topic engagement multipliers, aligned with ``topics``.
    rng:
        Seeded generator; one per creator so runs are reproducible.
    """

    def __init__(
        self,
        creator_id: int,
        strategy: str,
        initial_followers: int,
        topics: List[str],
        topic_multipliers_like: np.ndarray,
        topic_multipliers_share: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        if strategy not in STRATEGY_PARAMS:
            raise ValueError(
                f'Unknown strategy "{strategy}". Valid options: {list(STRATEGY_PARAMS)}'
            )

        self.creator_id = creator_id
        self.strategy = strategy
        self.followers = float(initial_followers)
        self.topics = topics
        self.n_topics = len(topics)
        self.topic_mult_like = topic_multipliers_like
        self.topic_mult_share = topic_multipliers_share
        self.rng = rng

        q_mu, q_sig, eta, tau, post_rate = STRATEGY_PARAMS[strategy]
        self.adaptation_rate = eta
        self.trend_sensitivity = tau
        self.posting_rate = post_rate

        # One quality draw per creator per run, clipped to a probability-like scale.
        self.quality = float(np.clip(rng.normal(q_mu, q_sig), 0.05, 1.0))

        # Content vector: a distribution over topics. Niche specialists put all
        # their mass on a single topic and never move it.
        if strategy == "niche":
            preferred_topic = int(rng.integers(0, self.n_topics))
            self.content_vector = np.zeros(self.n_topics)
            self.content_vector[preferred_topic] = 1.0
            self.preferred_topic = preferred_topic
        else:
            raw = rng.dirichlet(np.ones(self.n_topics))
            self.content_vector = raw
            self.preferred_topic = int(np.argmax(raw))

        self.follower_history: List[float] = [self.followers]
        self.viral_events: int = 0
        self.viral_timesteps: List[int] = []

    # -- behaviour ----------------------------------------------------------

    def choose_topic(self, trend_active: bool, trend_topic_idx: int) -> int:
        """Pick this step's topic.

        Niche specialists always post their one topic. Everyone else samples
        from their content vector, except that a live trend is adopted with
        probability ``trend_sensitivity``.
        """
        if self.strategy == "niche":
            return self.preferred_topic

        if trend_active and self.rng.random() < self.trend_sensitivity:
            return trend_topic_idx

        return int(self.rng.choice(self.n_topics, p=self.content_vector))

    def produce_video(
        self,
        timestep: int,
        trend_active: bool,
        trend_topic_idx: int,
        trend_intensity: float,
    ) -> List[Video]:
        """Publish ``posting_rate`` videos for the current step."""
        videos = []
        for _ in range(self.posting_rate):
            topic_idx = self.choose_topic(trend_active, trend_topic_idx)
            trend_aligned = trend_active and (topic_idx == trend_topic_idx)

            # Small per-video noise around the creator's baseline quality.
            video_quality = float(np.clip(self.quality + self.rng.normal(0, 0.05), 0.01, 1.0))

            # Riding a live trend raises effective quality in proportion to intensity.
            if trend_aligned:
                video_quality = float(
                    np.clip(video_quality * (1.0 + 0.8 * trend_intensity), 0.01, 1.0)
                )

            videos.append(Video(
                creator_id=self.creator_id,
                topic_idx=topic_idx,
                quality=video_quality,
                trend_aligned=trend_aligned,
                topic_multiplier_like=self.topic_mult_like[topic_idx],
                topic_multiplier_share=self.topic_mult_share[topic_idx],
            ))
        return videos

    def update_content_vector(self, trend_topic_idx: int, trend_intensity: float) -> None:
        """Drift the content distribution toward the trend topic.

        .. math:: C(t+1) = (1 - \\eta I(t))\\,C(t) + \\eta I(t)\\,e_{\\text{trend}}

        Niche specialists have :math:`\\eta = 0` and are a no-op here, which is
        asserted by a correctness test.
        """
        if self.adaptation_rate == 0 or self.strategy == "niche":
            return

        trend_vec = np.zeros(self.n_topics)
        trend_vec[trend_topic_idx] = 1.0

        effective_eta = self.adaptation_rate * trend_intensity
        self.content_vector = (
            (1.0 - effective_eta) * self.content_vector + effective_eta * trend_vec
        )
        total = self.content_vector.sum()
        if total > 0:
            self.content_vector /= total

    def receive_followers(self, delta_f: float, timestep: int, was_viral: bool) -> None:
        """Apply this step's net follower change and record any viral event."""
        self.followers = max(0.0, self.followers + delta_f)
        self.follower_history.append(self.followers)
        if was_viral:
            self.viral_events += 1
            self.viral_timesteps.append(timestep)
