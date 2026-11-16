"""
Outcome-based weight validation.

For each path:
  * grid-search the weight simplex and measure the Gini of the composite score
    against the dormancy outcome
  * compare with a *logistic ceiling*: a cross-validated logistic regression on
    the same sub-scores, i.e. the best any linear blend could realistically do
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

PATH_SIGNALS = {
    "DIGITAL_ETB": ["channel", "balance", "txn"],
    "DIGITAL_NTB": ["channel", "txn"],
    "NON_DIGITAL": ["balance", "txn"],
}


def gini(y, risk) -> float:
    return 2 * roc_auc_score(y, risk) - 1


def weight_grid(signals: list[str], step: float = 0.05):
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 2)
    for combo in itertools.product(ticks, repeat=len(signals)):
        if abs(sum(combo) - 1) < 1e-9:
            yield dict(zip(signals, combo))


def score_gini(d: pd.DataFrame, w: dict[str, float]) -> float:
    s = sum(w[k] * d[f"s_{k}"].fillna(50) for k in w)
    return gini(d.dormant_flag, -s)  # low score = high dormancy risk


def logistic_ceiling(d: pd.DataFrame, signals: list[str]) -> float:
    X = d[[f"s_{s}" for s in signals]].fillna(50).to_numpy()
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, d.dormant_flag, cv=5, method="predict_proba")[:, 1]
    return gini(d.dormant_flag, p)


def validate_path(df: pd.DataFrame, path: str, production: dict[str, float], step: float = 0.05) -> dict:
    d = df[df.customer_path == path]
    sigs = PATH_SIGNALS[path]
    grid = pd.DataFrame([{**w, "gini": score_gini(d, w)} for w in weight_grid(sigs, step)])
    best = grid.sort_values("gini", ascending=False).iloc[0]
    prod_g = score_gini(d, {s: production[s] for s in sigs})
    ceil = logistic_ceiling(d, sigs)
    return {
        "path": path,
        "customers": int(len(d)),
        "dormancy_rate": round(float(d.dormant_flag.mean()), 4),
        "production_weights": {s: production[s] for s in sigs},
        "production_gini": round(prod_g, 4),
        "best_weights": {s: float(best[s]) for s in sigs},
        "best_gini": round(float(best.gini), 4),
        "logistic_ceiling_gini": round(ceil, 4),
        "production_gap_to_ceiling_pct": round(100 * (ceil - prod_g) / ceil, 1),
        "grid": grid,
    }
