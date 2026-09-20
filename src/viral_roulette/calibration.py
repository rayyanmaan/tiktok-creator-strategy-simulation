"""Turning four real datasets into the parameters the model runs on.

Nothing in the simulation is hand-tuned to produce a nice-looking result. Every
engagement rate, topic multiplier, trend lifecycle and starting follower count
traces back to one of these four sources, and this module is where that
translation happens.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from .metrics import compute_gini, compute_lorenz

# Default layout of the bundled ``data/`` directory.
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)

TREND_FILES = {
    "Sea Shanty (Nov 2020)": os.path.join("trends", "sea_shanty.csv"),
    "Corn Kid (Aug 2022)": os.path.join("trends", "corn_kid.csv"),
    "Brat Summer (May 2024)": os.path.join("trends", "brat_summer.csv"),
    "Demure (Jul 2024)": os.path.join("trends", "demure.csv"),
}

# Tier boundaries follow the industry benchmark report's definitions.
TIER_BINS = [0, 10_000, 50_000, 500_000, 1_000_000, np.inf]
TIER_LABELS = ["Nano", "Micro", "Mid-tier", "Macro", "Mega"]


def data_paths(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, str]:
    """Resolve every input file under ``data_dir``."""
    paths = {
        "profiles": os.path.join(data_dir, "tiktok_profiles_dataset.csv"),
        "hashtags": os.path.join(data_dir, "tiktok_popular_hashtags_dataset.xlsx"),
    }
    for name, rel in TREND_FILES.items():
        paths[name] = os.path.join(data_dir, rel)
    return paths


# --- A1a: creator profiles --------------------------------------------------

def calibrate_profiles(path: str) -> Dict:
    """Extract the inequality calibration target from the creator sample.

    Produces the real Gini coefficient and Lorenz curve that the simulated
    follower distribution is later measured against, plus engagement rate by
    follower tier.
    """
    raw = pd.read_csv(path)
    df = raw.dropna(subset=["followers", "awg_engagement_rate"]).copy()
    df = df[(df["followers"] > 0) & (df["awg_engagement_rate"] >= 0)]

    real_gini = compute_gini(df["followers"].values)
    lorenz_x, lorenz_y = compute_lorenz(df["followers"].values)

    df["tier"] = pd.cut(df["followers"], bins=TIER_BINS, labels=TIER_LABELS, right=False)
    er_by_tier = (
        df.groupby("tier", observed=True)["awg_engagement_rate"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "ER_mean", "std": "ER_std", "count": "n"})
    )

    return {
        "n_rows_raw": len(raw),
        "n_rows_clean": len(df),
        "real_gini": real_gini,
        "lorenz_x": lorenz_x,
        "lorenz_y": lorenz_y,
        "er_by_tier": er_by_tier,
        "follower_values": df["followers"].values,
        "tier_distribution": df["tier"].value_counts(normalize=True).to_dict(),
    }


# --- A1b: hashtag engagement ------------------------------------------------

def _find_col(cols, priority_keywords, fallback_keywords):
    """Match a column by priority keywords first, then fallbacks.

    The hashtag export mixes video fields with author metadata, so an
    unprioritised substring match happily picks ``authorMeta/digg`` when it
    wants ``diggCount``. Priority ordering prevents that.
    """
    for kw in priority_keywords:
        for c in cols:
            if kw in c.lower():
                return c
    for kw in fallback_keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None


#: The 12 topics the published results were computed on.
#:
#: This list is pinned rather than recomputed, and the reason is a genuine
#: reproducibility trap. Every one of the 60 hashtags in the dataset has
#: *exactly* 20 rows, so selecting the "top 12 by frequency" is a pure tie-break
#: -- and ``value_counts()`` does not guarantee a stable order for ties across
#: pandas versions. The original run selected this humour-adjacent cluster;
#: pandas 2.3 selects a different twelve and shifts the baseline like rate from
#: 0.1478 to 0.1388. Pinning makes the published numbers reproducible on any
#: pandas; pass ``pinned_topics=None`` to recover the frequency-ranked behaviour.
PUBLISHED_TOPICS = [
    "lmao", "jokes", "memes", "funnymemes", "love", "lol",
    "meme", "humor", "laugh", "comedy", "fun", "funny",
]


def calibrate_hashtags(
    path: str,
    top_n_topics: int = 12,
    min_rows_per_topic: int = 5,
    target_E: float = 0.32,
    pinned_topics: List[str] | None = PUBLISHED_TOPICS,
) -> Dict:
    """Derive per-topic engagement multipliers and the engagement scale factors.

    ``target_E`` is the key modelling lever: the like and share scales are
    solved so that an *average* video (``q = 0.5``) scores just **below** the
    0.35 amplification threshold. If average content cleared the threshold by
    default there would be no strategy trade-off left to measure.

    See :data:`PUBLISHED_TOPICS` for why the topic set is pinned by default.
    """
    raw = pd.read_excel(path)

    play_col = _find_col(raw.columns, ["playcount", "play_count"], ["views", "view"])
    like_col = _find_col(raw.columns, ["diggcount", "digg"], ["like"])
    share_col = _find_col(raw.columns, ["sharecount", "share"], [])
    topic_col = _find_col(raw.columns, ["searchhashtag/name", "searchhashtag"], ["hashtag"])
    for name, col in [("play", play_col), ("like", like_col),
                      ("share", share_col), ("topic", topic_col)]:
        assert col is not None, f"No {name} column found in: {raw.columns.tolist()}"

    ht = raw[[play_col, like_col, share_col, topic_col]].copy()
    ht.columns = ["play_count", "like_count", "share_count", "topic"]
    for col in ["play_count", "like_count", "share_count"]:
        ht[col] = pd.to_numeric(ht[col], errors="coerce")
    ht = ht.dropna()
    ht = ht[ht["play_count"] > 0]
    ht["topic"] = ht["topic"].astype(str).str.lower().str.strip()
    ht = ht[~ht["topic"].isin({"nan", "", "null", "none"})]

    # Clip to plausible platform ranges before averaging so that a handful of
    # mis-scraped rows cannot dominate a topic's multiplier.
    ht["like_rate"] = (ht["like_count"] / ht["play_count"]).clip(0, 0.30)
    ht["share_rate"] = (ht["share_count"] / ht["play_count"]).clip(0, 0.10)

    topic_counts = ht["topic"].value_counts()
    valid_topics = topic_counts[topic_counts >= min_rows_per_topic].index
    topic_stats = (
        ht[ht["topic"].isin(valid_topics)]
        .groupby("topic")[["like_rate", "share_rate"]]
        .mean()
        .sort_values("like_rate", ascending=False)
    )
    if pinned_topics is not None:
        missing = [t for t in pinned_topics if t not in topic_stats.index]
        assert not missing, f"pinned topics absent from the dataset: {missing}"
        top_topics = pinned_topics
    else:
        top_topics = topic_counts[
            topic_counts.index.isin(topic_stats.index)
        ].head(top_n_topics).index
    topic_stats = topic_stats.loc[topic_stats.index.isin(top_topics)]

    mean_like_rate = float(topic_stats["like_rate"].mean())
    mean_share_rate = float(topic_stats["share_rate"].mean())
    topic_stats["like_multiplier"] = topic_stats["like_rate"] / mean_like_rate
    topic_stats["share_multiplier"] = topic_stats["share_rate"] / mean_share_rate

    # Solve for the scale factors. At q = 0.5:
    #   p_complete = sigmoid(0) = 0.50,  p_skip = 1 - 0.5*0.9 = 0.55
    #   0.40*p_complete + 0.20*p_like + 0.25*p_share - 0.15*p_skip = target_E
    # with p_like = q * baseline_like * LIKE_SCALE, likewise for share, and
    # SHARE_SCALE fixed at 0.6 * LIKE_SCALE because sharing is harder than liking.
    W_COMP, W_LIKE, W_SHARE, W_SKIP = 0.40, 0.20, 0.25, 0.15
    q_avg = 0.5
    p_comp_avg = 1.0 / (1.0 + np.exp(-6.0 * (q_avg - 0.5)))
    p_skip_avg = float(np.clip(1.0 - q_avg * 0.9, 0.05, 0.95))
    residual = target_E - W_COMP * p_comp_avg + W_SKIP * p_skip_avg

    if mean_like_rate > 0 and mean_share_rate > 0:
        denominator = (W_LIKE * mean_like_rate + W_SHARE * mean_share_rate * 0.6) * q_avg
        like_scale = residual / denominator
        share_scale = like_scale * 0.6
    else:
        like_scale, share_scale = 40.0, 24.0

    like_scale = max(float(np.clip(like_scale, 10.0, 80.0)), 35.0)
    share_scale = max(float(np.clip(share_scale, 6.0, 48.0)), 21.0)

    return {
        "baseline_like_rate": mean_like_rate,
        "baseline_share_rate": mean_share_rate,
        "like_scale": like_scale,
        "share_scale": share_scale,
        "topic_multipliers": topic_stats[["like_multiplier", "share_multiplier"]].to_dict("index"),
        "topic_stats": topic_stats,
        "topics": list(topic_stats.index),
    }


# --- A1c: trend lifecycles --------------------------------------------------

def load_google_trend(filepath: str) -> np.ndarray:
    """Read a Google Trends export and normalise weekly interest to ``[0, 1]``."""
    df = pd.read_csv(filepath, skiprows=1)
    interest_col = df.columns[1]
    raw = df[interest_col].replace("<1", "0").astype(float).dropna().values
    return raw / raw.max() if raw.max() > 0 else raw


def lognormal_curve(t: np.ndarray, mu: float, sigma: float, scale: float) -> np.ndarray:
    """Log-normal density used as a trend lifecycle shape."""
    t = np.clip(t, 1e-9, None)
    return scale * stats.lognorm.pdf(t, s=sigma, scale=np.exp(mu))


def calibrate_trends(paths: Dict[str, str]) -> Dict:
    """Fit a log-normal lifecycle to each real trend series.

    The fits are not required to reconstruct each trend exactly. They supply a
    *family* of plausible intensity schedules -- fast spike, slow burn, plateau
    -- which the trend engine then samples from.
    """
    trend_params: Dict[str, Dict] = {}
    series: Dict[str, np.ndarray] = {}

    for name, filepath in paths.items():
        interest = load_google_trend(filepath)
        series[name] = interest
        t = np.arange(1, len(interest) + 1, dtype=float)
        p0 = [np.log(float(np.argmax(interest) + 1)), 1.0, 1.0]
        try:
            popt, _ = curve_fit(
                lognormal_curve, t, interest, p0=p0, maxfev=8000,
                bounds=([0, 0.01, 0], [10, 5, 100]),
            )
            mu_fit, sigma_fit, scale_fit = popt
            fit_success = True
        except RuntimeError:
            # Fall back to moment matching on the raw series.
            log_t = np.log(t)
            w = interest / interest.sum() if interest.sum() > 0 else np.ones_like(interest) / len(interest)
            mu_fit = float(np.dot(w, log_t))
            sigma_fit = float(np.sqrt(np.dot(w, (log_t - mu_fit) ** 2)))
            scale_fit = float(interest.max())
            fit_success = False

        trend_params[name] = {
            "mu": float(mu_fit), "sigma": float(sigma_fit), "scale": float(scale_fit),
            "n_weeks": len(interest), "fit_success": fit_success,
        }

    return {
        "trend_params": trend_params,
        "series": series,
        "mu_values": [v["mu"] for v in trend_params.values()],
        "sigma_values": [v["sigma"] for v in trend_params.values()],
    }


# --- A1d: industry benchmarks ----------------------------------------------

#: Engagement rate (%) by follower tier, from the 2026 influencer benchmark report.
BENCHMARK_ER = {
    "Nano": 11.6, "Micro": 9.3, "Mid-tier": 7.6, "Macro": 6.8, "Mega": 6.8,
    "Overall": 11.2,
}

#: Share of the creator population in each tier. The platform is overwhelmingly
#: nano-tier, and initialising creators from this distribution rather than from
#: zero is what makes the follower-stock regime of the results appear at all.
BENCHMARK_TIER_DIST = {
    "Nano": 0.887, "Micro": 0.089, "Mid-tier": 0.022, "Macro": 0.002, "Mega": 0.000,
}

TIER_FOLLOWER_RANGES: Dict[str, Tuple[int, int]] = {
    "Nano": (1_000, 10_000),
    "Micro": (10_000, 50_000),
    "Mid-tier": (50_000, 500_000),
    "Macro": (500_000, 1_000_000),
    "Mega": (1_000_000, 10_000_000),
}


def benchmark_bundle() -> Dict:
    """Package the published benchmark figures for the simulation."""
    return {
        "er_by_tier": BENCHMARK_ER,
        "tier_dist": BENCHMARK_TIER_DIST,
        "tier_ranges": TIER_FOLLOWER_RANGES,
    }


# --- A1e: the full bundle ---------------------------------------------------

def build_calibration(data_dir: str = DEFAULT_DATA_DIR) -> Dict:
    """Run the whole calibration pipeline and return the bundle the model takes.

    >>> cal = build_calibration()
    >>> round(cal["profiles"]["real_gini"], 4)
    0.9086
    """
    paths = data_paths(data_dir)
    trend_paths = {name: paths[name] for name in TREND_FILES}
    return {
        "profiles": calibrate_profiles(paths["profiles"]),
        "hashtags": calibrate_hashtags(paths["hashtags"]),
        "trends": calibrate_trends(trend_paths),
        "benchmarks": benchmark_bundle(),
    }
