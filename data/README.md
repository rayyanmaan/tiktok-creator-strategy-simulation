# Data provenance

Four sources. Every parameter the simulation runs on is derived from one of
them; nothing in the model is a free-floating constant. Each entry below states
what the file is, where it came from, what is extracted from it, and which
appendix of the notebook does the extraction.

| File | Rows | Source | What it calibrates | Appendix |
|---|---:|---|---|---|
| `tiktok_profiles_dataset.csv` | 1,000 | Bright Data, TikTok user profiles dataset | Real follower Gini (0.9086), Lorenz curve, engagement rate by tier | A1a |
| `tiktok_popular_hashtags_dataset.xlsx` | 1,200 | Kaggle, *TikTok popular hashtags* (CC0) | Per-topic like/share multipliers, baseline engagement rates, funnel scale factors | A1b |
| `trends/sea_shanty.csv` | 25 wks | Google Trends, worldwide weekly interest | Log-normal lifecycle fit (slow build, sustained peak) | A1c |
| `trends/corn_kid.csv` | 22 wks | Google Trends | Log-normal lifecycle fit (fast symmetric spike) | A1c |
| `trends/brat_summer.csv` | 26 wks | Google Trends | Log-normal lifecycle fit (plateau, slow decay) | A1c |
| `trends/demure.csv` | 22 wks | Google Trends | Log-normal lifecycle fit (fast spike, rapid crash) | A1c |
| *(no file)* | — | HypeAuditor, *State of Influencer Marketing 2026* | Creator-tier population shares and engagement benchmarks by tier | A1d |

The HypeAuditor figures are published summary statistics rather than a
downloadable dataset, so they are transcribed as constants in
[`src/creator_strategy_sim/calibration.py`](../src/creator_strategy_sim/calibration.py)
(`BENCHMARK_ER`, `BENCHMARK_TIER_DIST`) with the source named at the definition.

## Columns actually used

**`tiktok_profiles_dataset.csv`** — `followers`, `awg_engagement_rate`. Rows are
kept only where both are present, `followers > 0` and `awg_engagement_rate >= 0`.
All 1,000 rows survive cleaning.

**`tiktok_popular_hashtags_dataset.xlsx`** — `playCount`, `diggCount`,
`shareCount`, `searchHashtag/name`. The file also carries `authorMeta/*`
columns, so column matching is priority-ordered: an unprioritised substring
search for `"digg"` happily selects `authorMeta/digg` instead of `diggCount`.

**`trends/*.csv`** — standard Google Trends exports: a two-row header followed
by `date, interest` where interest is an integer 0–100 (`<1` is coerced to 0).
Each series is normalised to `[0, 1]` before fitting.

## Licensing

These files are redistributed here so that the notebook runs end to end without
manual downloads. The hashtag dataset is CC0. The remaining files are included
for reproducibility of the analysis only; rights remain with their original
publishers, and if you intend to use any of them beyond reproducing this
analysis, obtain them from the original source under that source's terms.
