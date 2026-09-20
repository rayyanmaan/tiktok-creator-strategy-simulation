"""creator_strategy_sim -- a calibrated Monte Carlo model of creator strategy and
emergent inequality on a short-video recommendation platform.

Distributed as ``tiktok-creator-strategy-simulation``; imported as
``creator_strategy_sim``.

Quick start
-----------
>>> from creator_strategy_sim import build_calibration, run_monte_carlo
>>> cal = build_calibration()
>>> res = run_monte_carlo("trend_chaser", cal, n_runs=50)
>>> res["viral_rate"].mean()          # doctest: +SKIP
0.0495
"""

from .algorithm import Algorithm
from .calibration import build_calibration, data_paths
from .config import (
    N_CREATORS, N_MC_RUNS, N_STEPS, STRATEGIES, STRATEGY_COLORS,
    STRATEGY_LABELS, STRATEGY_PARAMS, apply_plot_style,
)
from .creator import Creator, Video
from .metrics import compute_gini, compute_lorenz, mean_ci
from .montecarlo import run_monte_carlo
from .simulation import Simulation
from .theory import (
    analytical_growth_rate, analytical_p_viral,
    recovery_time_formula, theoretical_gini_lognormal,
)
from .trends import TrendEngine

__version__ = "1.0.0"

__all__ = [
    "Algorithm", "Creator", "Video", "TrendEngine", "Simulation",
    "run_monte_carlo", "build_calibration", "data_paths",
    "compute_gini", "compute_lorenz", "mean_ci",
    "analytical_p_viral", "analytical_growth_rate",
    "theoretical_gini_lognormal", "recovery_time_formula",
    "STRATEGIES", "STRATEGY_PARAMS", "STRATEGY_LABELS", "STRATEGY_COLORS",
    "N_CREATORS", "N_STEPS", "N_MC_RUNS", "apply_plot_style",
]
