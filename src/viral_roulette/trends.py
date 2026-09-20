"""Trend arrival and decay.

Trend lifecycles are not invented: their shapes are log-normal curves fitted to
four real Google Trends series (Sea Shanty, Corn Kid, Brat Summer, Demure). The
engine samples one fitted ``(mu, sigma)`` pair whenever a trend spawns, so the
simulation sees a realistic mix of flash-viral and slow-burn lifecycles.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


class TrendEngine:
    """Spawns at most one live trend at a time and tracks its intensity.

    Parameters
    ----------
    p_trend:
        Probability that a new trend spawns in a step where none is live.
    mu_values, sigma_values:
        The four fitted log-normal parameter pairs.
    n_topics:
        Number of content topics; a trend claims one at random.
    """

    MAX_DURATION = 52   # cap a trend at one simulated year
    DEATH_INTENSITY = 0.05

    def __init__(
        self,
        p_trend: float,
        mu_values: List[float],
        sigma_values: List[float],
        n_topics: int,
        rng: np.random.Generator,
    ) -> None:
        self.p_trend = p_trend
        self.mu_values = mu_values
        self.sigma_values = sigma_values
        self.n_topics = n_topics
        self.rng = rng

        self._active = False
        self._topic_idx = 0
        self._step_in_trend = 0
        self._mu = 0.0
        self._sigma = 1.0
        self._lifecycle: np.ndarray = np.array([])

        self.history: List[Dict] = []

    def _spawn_trend(self) -> None:
        """Draw a lifecycle shape from the empirical parameter pairs."""
        idx = int(self.rng.integers(0, len(self.mu_values)))
        self._mu = self.mu_values[idx]
        self._sigma = self.sigma_values[idx]

        t = np.arange(1, self.MAX_DURATION + 1, dtype=float)
        raw = stats.lognorm.pdf(t, s=self._sigma, scale=np.exp(self._mu))
        self._lifecycle = raw / raw.max() if raw.max() > 0 else raw

        self._topic_idx = int(self.rng.integers(0, self.n_topics))
        self._step_in_trend = 0
        self._active = True

    def step(self, current_timestep: int) -> Tuple[bool, int, float]:
        """Advance one week.

        Returns ``(is_active, topic_idx, intensity)`` where ``intensity`` is in
        ``[0, 1]`` and is 0 when no trend is live.
        """
        if not self._active and self.rng.random() < self.p_trend:
            self._spawn_trend()

        if self._active:
            if self._step_in_trend < len(self._lifecycle):
                intensity = float(self._lifecycle[self._step_in_trend])
            else:
                intensity = 0.0

            if intensity < self.DEATH_INTENSITY or self._step_in_trend >= self.MAX_DURATION:
                self._active = False
                intensity = 0.0

            self._step_in_trend += 1
        else:
            intensity = 0.0

        self.history.append({
            "timestep": current_timestep,
            "active": self._active,
            "topic_idx": self._topic_idx if self._active else -1,
            "intensity": intensity,
        })
        return self._active, self._topic_idx, intensity
