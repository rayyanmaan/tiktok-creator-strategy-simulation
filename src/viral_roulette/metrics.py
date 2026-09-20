"""Inequality metrics shared by the calibration step and the experiments."""

from __future__ import annotations

import numpy as np


def compute_gini(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array of non-negative values.

    Uses the sorted-array formula

    .. math:: G = \\frac{2 \\sum_i i\\,x_{(i)}}{n \\sum_i x_i} - \\frac{n+1}{n}

    where :math:`x_{(i)}` are the values in ascending order.

    Returns a float in ``[0, 1]``: 0 is perfect equality, 1 is maximal
    concentration. Returns 0.0 for an empty or all-zero input.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.dot(index, arr)) / (n * arr.sum()) - (n + 1) / n


def compute_lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve of a 1-D array of non-negative values.

    Returns ``(cumulative_population_share, cumulative_value_share)``, both of
    length ``len(values) + 1`` and both anchored at ``(0, 0)``.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    cum_share = np.concatenate([[0], np.arange(1, len(arr) + 1) / len(arr)])
    cum_val = np.concatenate([[0], np.cumsum(arr) / arr.sum()])
    return cum_share, cum_val


def mean_ci(samples: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Normal-approximation confidence interval for the mean of ``samples``.

    Returns ``(estimate, lower, upper)``.
    """
    from scipy import stats

    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    est = float(arr.mean())
    if n < 2:
        return est, est, est
    half = float(stats.norm.ppf(0.5 + confidence / 2) * arr.std(ddof=1) / np.sqrt(n))
    return est, est - half, est + half
