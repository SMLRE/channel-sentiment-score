"""
Channel Sentiment Score: 0-100, higher = healthier relationship.

Each signal is turned into a 0-100 sub-score by percentile rank *within the
customer's path*, so new customers are compared with new customers. The
sub-scores are then blended with path-specific weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = {"channel": "success_rate_shrunk", "balance": "balance_trend", "txn": "txn_velocity"}

# Production weights before validation (channel / balance / txn)
PRODUCTION_WEIGHTS = {
    "DIGITAL_ETB": {"channel": 0.60, "balance": 0.20, "txn": 0.20},
    "DIGITAL_NTB": {"channel": 0.80, "balance": 0.00, "txn": 0.20},
    "NON_DIGITAL": {"channel": 0.00, "balance": 0.50, "txn": 0.50},
}

BANDS = ["Negative", "At-Risk", "Neutral", "Positive"]
DEFAULT_CUTS = [30, 50, 70]  # score < 30 Negative, 30-50 At-Risk, 50-70 Neutral, >= 70 Positive


def sub_scores(df: pd.DataFrame, channel_col: str = "success_rate_shrunk") -> pd.DataFrame:
    out = df.copy()
    cols = {**SIGNALS, "channel": channel_col}
    for name, col in cols.items():
        out[f"s_{name}"] = out.groupby("customer_path")[col].rank(pct=True) * 100
    return out


def composite(df: pd.DataFrame, weights: dict[str, dict[str, float]]) -> pd.Series:
    score = pd.Series(np.nan, index=df.index)
    for path, w in weights.items():
        m = df.customer_path == path
        parts = [w[s] * df.loc[m, f"s_{s}"].fillna(50) for s in w if w[s] > 0]
        score[m] = sum(parts) / sum(v for v in w.values() if v > 0)
    return score.round(1)


def band(score: pd.Series, cuts=DEFAULT_CUTS) -> pd.Series:
    return pd.cut(score, [-np.inf, *cuts, np.inf], labels=BANDS, right=False)


def quantile_matched_cuts(score: pd.Series, target_mix: dict[str, float]) -> list[float]:
    """Recalibrate band boundaries so the band mix matches a target distribution.

    After the weights change, the score distribution moves, and fixed cut-offs
    would suddenly put far more (or fewer) customers into 'Negative'.
    Quantile matching keeps the operational volumes stable.
    """
    cum = np.cumsum([target_mix[b] for b in BANDS])[:-1]
    return [float(np.quantile(score.dropna(), q)) for q in cum]
