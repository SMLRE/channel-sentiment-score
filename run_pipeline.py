"""
End-to-end Channel Sentiment Score pipeline.

    1. Generate synthetic data (if missing)
    2. Build the three signals in SQL (DuckDB)
    3. Score customers with the production weights
    4. Validate the weights against real dormancy outcomes, per path
    5. Adopt validated weights, then recalibrate the bands with quantile matching
    6. Write metrics, figures and the scored file to reports/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.score import PRODUCTION_WEIGHTS, band, composite, quantile_matched_cuts, sub_scores, BANDS
from src.signals import build_signals, connect
from src.validate import gini, score_gini, validate_path

ROOT = Path(__file__).parent
REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
ACCENT, GREY = "#E8A54B", "#8A8A8A"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

# Recommended weights: the validated optimum, rounded towards a stable plateau
RECOMMENDED = {
    "DIGITAL_ETB": {"channel": 0.60, "balance": 0.20, "txn": 0.20},
    "DIGITAL_NTB": {"channel": 0.25, "balance": 0.00, "txn": 0.75},
    "NON_DIGITAL": {"channel": 0.00, "balance": 0.50, "txn": 0.50},
}
TARGET_MIX = {"Negative": 0.15, "At-Risk": 0.25, "Neutral": 0.35, "Positive": 0.25}


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data/raw/txn_monthly.parquet").exists():
        subprocess.run([sys.executable, str(ROOT / "data/generate_synthetic.py")], check=True)

    con = connect()
    df = sub_scores(build_signals(con))
    print(f"{len(df):,} customers active at t0 | dormancy rate {df.dormant_flag.mean():.1%}")

    # ---- 1. Why shrinkage + volume weighting matter
    sig_cmp = {}
    for label, col in [("naive mean of channel rates", "success_rate_naive"),
                       ("volume-weighted", "success_rate_raw"),
                       ("volume-weighted + shrinkage", "success_rate_shrunk")]:
        d = sub_scores(df, channel_col=col)
        sig_cmp[label] = {p: round(score_gini(d[d.customer_path == p], {"channel": 1.0}), 4)
                          for p in ["DIGITAL_ETB", "DIGITAL_NTB"]}
    print("Channel-signal Gini on its own:", json.dumps(sig_cmp, indent=1))

    # ---- 2. Weight validation
    results = {p: validate_path(df, p, PRODUCTION_WEIGHTS[p]) for p in PRODUCTION_WEIGHTS}
    for p, r in results.items():
        r["recommended_weights"] = {s: RECOMMENDED[p][s] for s in r["production_weights"]}
        r["recommended_gini"] = round(score_gini(df[df.customer_path == p], r["recommended_weights"]), 4)
        print(f"{p:12s} prod {r['production_gini']:.3f} | best {r['best_gini']:.3f} {r['best_weights']} "
              f"| ceiling {r['logistic_ceiling_gini']:.3f} | recommended {r['recommended_gini']:.3f}")

    # ETB weight surface
    g = results["DIGITAL_ETB"]["grid"]
    piv = g.pivot_table(index="channel", columns="balance", values="gini")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(piv.values, origin="lower", cmap="magma", aspect="auto",
                   extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    ax.scatter([0.2], [0.6], marker="o", s=120, facecolors="none", edgecolors="#00B4D8", lw=2.5, label="60/20/20 (production)")
    ax.set_xlabel("balance weight"); ax.set_ylabel("channel weight  (txn = remainder)")
    ax.set_title("Established digital: Gini across the weight simplex"); ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Gini vs dormancy"); fig.tight_layout(); fig.savefig(FIG / "etb_weight_surface.png"); plt.close(fig)

    # NTB curve
    g = results["DIGITAL_NTB"]["grid"].sort_values("channel")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(g.channel, g.gini, color=ACCENT, lw=2)
    ax.axhline(results["DIGITAL_NTB"]["logistic_ceiling_gini"], ls="--", color=GREY, label="logistic ceiling")
    ax.axvline(0.80, color="#D9534F", lw=1); ax.text(0.81, g.gini.min(), "production 80/20", color="#D9534F", fontsize=9)
    ax.axvline(0.25, color="#3C8D5A", lw=1); ax.text(0.26, g.gini.max(), "recommended 25/75", color="#3C8D5A", fontsize=9, va="top")
    ax.set_xlabel("channel weight (transaction weight = 1 - channel)"); ax.set_ylabel("Gini vs dormancy")
    ax.set_title("New digital customers: the weight was on the wrong signal"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "ntb_weight_curve.png"); plt.close(fig)

    # ---- 3. Score with recommended weights + band recalibration
    df["css_production"] = composite(df, PRODUCTION_WEIGHTS)
    df["css"] = composite(df, RECOMMENDED)
    old_bands = band(df.css, )
    cuts = quantile_matched_cuts(df.css, TARGET_MIX)
    df["css_band"] = band(df.css, cuts)
    mix_before = old_bands.value_counts(normalize=True).reindex(BANDS).round(3).to_dict()
    mix_after = df.css_band.value_counts(normalize=True).reindex(BANDS).round(3).to_dict()
    by_band = df.groupby("css_band", observed=True).dormant_flag.agg(["size", "mean"]).reindex(BANDS)
    print("bands (after recalibration):\n", by_band)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(BANDS, by_band["mean"] * 100, color=["#D9534F", ACCENT, "#B9B4A8", "#3C8D5A"])
    for i, v in enumerate(by_band["mean"] * 100):
        ax.text(i, v + 0.6, f"{v:.0f}%", ha="center")
    ax.set_ylabel("dormant within ~11 months (%)"); ax.set_title("Dormancy rate by sentiment band")
    fig.tight_layout(); fig.savefig(FIG / "dormancy_by_band.png"); plt.close(fig)

    auc_all = roc_auc_score(df.dormant_flag, -df.css)
    auc_prod = roc_auc_score(df.dormant_flag, -df.css_production)

    df[["customer_id", "customer_path", "s_channel", "s_balance", "s_txn", "css", "css_band"]].round(1).to_csv(
        REPORTS / "scored_customers.csv", index=False)

    metrics = {
        "customers_scored": int(len(df)),
        "dormancy_rate": round(float(df.dormant_flag.mean()), 4),
        "auc_all_paths_production_weights": round(auc_prod, 4),
        "auc_all_paths_recommended_weights": round(auc_all, 4),
        "gini_all_paths_recommended_weights": round(2 * auc_all - 1, 4),
        "channel_signal_gini_by_method": sig_cmp,
        "weight_validation": {p: {k: v for k, v in r.items() if k != "grid"} for p, r in results.items()},
        "band_cuts_recalibrated": [round(c, 1) for c in cuts],
        "band_mix_fixed_cuts": mix_before,
        "band_mix_recalibrated": mix_after,
        "dormancy_rate_by_band": by_band["mean"].round(4).to_dict(),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"AUC all paths: production {auc_prod:.3f} -> recommended {auc_all:.3f}")
    return metrics


if __name__ == "__main__":
    main()
