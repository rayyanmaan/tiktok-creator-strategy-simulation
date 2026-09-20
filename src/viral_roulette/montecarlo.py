"""The single entry point every experiment goes through.

Having one runner rather than one script per experiment is what keeps the four
strategy conditions genuinely comparable: seeds, run counts and output shapes
cannot silently diverge between conditions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import (
    GLOBAL_SEED, N_CREATORS, N_MC_RUNS, N_STEPS, P_TREND,
    STRATEGY_LABELS, STRATEGY_PARAMS,
)
from .simulation import Simulation


def run_monte_carlo(
    strategy: str,
    calibration: Dict,
    n_runs: int = N_MC_RUNS,
    n_steps: int = N_STEPS,
    n_creators: int = N_CREATORS,
    p_trend: float = P_TREND,
    base_seed: int = GLOBAL_SEED,
    verbose: bool = True,
) -> Dict:
    """Run ``n_runs`` independent trials of one strategy.

    Run ``k`` uses seed ``base_seed + k``, so the whole sweep is reproducible
    and any single run can be replayed in isolation for debugging.

    Returns
    -------
    dict
        Arrays of length ``n_runs`` -- ``final_followers`` (run-level mean),
        ``median_final_f``, ``survival_rate``, ``gini``, ``viral_rate`` -- plus
        median / p25 / p75 follower trajectories across runs.
    """
    if strategy not in STRATEGY_PARAMS:
        raise ValueError(f'Unknown strategy "{strategy}".')

    final_followers = np.zeros(n_runs)
    median_final_f = np.zeros(n_runs)
    survival_rate = np.zeros(n_runs)
    gini_vals = np.zeros(n_runs)
    viral_rate = np.zeros(n_runs)
    trajectories = []

    iterator = range(n_runs)
    if verbose:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc=STRATEGY_LABELS[strategy])
        except ImportError:
            pass

    for run_idx in iterator:
        result = Simulation(
            strategy=strategy,
            n_creators=n_creators,
            n_steps=n_steps,
            calibration=calibration,
            p_trend=p_trend,
            seed=base_seed + run_idx,
        ).run()

        ff = result["final_followers"]
        final_followers[run_idx] = np.mean(ff)
        median_final_f[run_idx] = np.median(ff)
        survival_rate[run_idx] = np.mean(result["survival_flags"])
        gini_vals[run_idx] = result["gini"]
        viral_rate[run_idx] = np.mean(result["viral_rates"])
        trajectories.append(np.mean(result["follower_traj"], axis=0))

    traj_array = np.asarray(trajectories)

    return {
        "strategy": strategy,
        "final_followers": final_followers,
        "median_final_f": median_final_f,
        "survival_rate": survival_rate,
        "gini": gini_vals,
        "viral_rate": viral_rate,
        "traj_medians": np.median(traj_array, axis=0),
        "traj_p25": np.percentile(traj_array, 25, axis=0),
        "traj_p75": np.percentile(traj_array, 75, axis=0),
    }
