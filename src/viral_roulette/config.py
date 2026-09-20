"""Global configuration: seeds, strategy parameters, run sizes, plot styling.

Every number in this module is either (a) an explicit modelling choice that is
argued for in the report, or (b) a value calibrated from real data in
:mod:`viral_roulette.calibration`. Nothing is a silent magic constant.
"""

from __future__ import annotations

# --- Reproducibility --------------------------------------------------------
GLOBAL_SEED = 42

# --- Simulation sizing ------------------------------------------------------
N_CREATORS = 30      # creator agents per simulation run
N_STEPS = 60         # time steps per run (1 step = 1 week)
N_MC_RUNS = 1000     # Monte Carlo trials per strategy
N_SENS_RUNS = 30     # trials per cell of the sensitivity grid
N_TEST_USERS = 100   # mean size of the Stage-1 exposure cohort

# --- Strategies -------------------------------------------------------------
# (quality_mean, quality_sd, adaptation_rate eta, trend_sensitivity tau, posts/step)
#
#   quality_mean      baseline production quality of the creator
#   quality_sd        within-strategy dispersion (consistency vs volatility)
#   adaptation_rate   how fast the content vector drifts toward a live trend
#   trend_sensitivity probability of posting on the trend topic when one is live
#   posting_rate      videos published per week
STRATEGY_PARAMS: dict[str, tuple[float, float, float, float, int]] = {
    "niche":           (0.58, 0.07, 0.00, 0.00, 2),
    "trend_chaser":    (0.30, 0.18, 0.55, 0.90, 3),
    "quality_focused": (0.88, 0.03, 0.08, 0.08, 1),
    "random_baseline": (0.45, 0.22, 0.28, 0.38, 2),
}

STRATEGIES = list(STRATEGY_PARAMS)

STRATEGY_LABELS = {
    "niche": "Niche Specialist",
    "trend_chaser": "Trend Chaser",
    "quality_focused": "Quality-Focused",
    "random_baseline": "Random Baseline",
}

# Colour-blind-safe palette; one fixed colour per strategy across every figure.
STRATEGY_COLORS = {
    "niche": "#2E86AB",            # blue
    "trend_chaser": "#E84855",     # red
    "quality_focused": "#3BB273",  # green
    "random_baseline": "#F4A261",  # orange
}

# --- Recommendation funnel defaults ----------------------------------------
EXPANSION_THRESHOLD = 0.35   # engagement score E above which a video is amplified
GROWTH_ALPHA = 8.0           # amplification slope above the threshold
CONVERSION_RATE = 0.02       # impressions -> new followers
WEEKLY_CHURN = 0.003         # 0.3% of the follower stock lost per week
P_TREND = 0.12               # probability a new trend spawns in a given week


def apply_plot_style() -> None:
    """Apply the shared matplotlib style used by every figure in the notebook."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
    })
