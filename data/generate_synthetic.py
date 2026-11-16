"""
Synthetic data generator for the Channel Sentiment Score.

Simulates a retail customer base over 23 months with three customer paths:

    DIGITAL_ETB   established digital customers (opened > 12 months before t0)
    DIGITAL_NTB   new digital customers (opened in the 6 months before t0)
    NON_DIGITAL   branch / ATM-only customers (no digital channel activity)

Each customer has three hidden drivers of disengagement:
    C  channel friction    -> shows up as failed digital transactions
    B  balance drift       -> shows up as a falling balance
    T  activity decline    -> shows up as falling transaction velocity

Dormancy after t0 is driven by a path-specific mix of these drivers, so the
"right" weights differ by path. That is what the weight-validation step has to
discover. New customers have few digital attempts, so their channel signal is
noisy, which is why shrinkage matters.

Tables written to data/raw/:
    customers.parquet
    channel_monthly.parquet   attempts / successes per customer x month x channel
    balance_monthly.parquet   average balance per customer x month (can be < 0)
    txn_monthly.parquet       customer-initiated transaction count per month
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_MONTHS = 23          # months 0..22
T0 = 12                # scoring date = start of month 12
CHANNELS = ["MOBILE", "INTERNET", "USSD", "CARD"]

# how strongly each hidden driver pushes towards dormancy, by path
DRIVER_MIX = {
    "DIGITAL_ETB": dict(C=1.20, B=0.42, T=0.42),
    "DIGITAL_NTB": dict(C=0.40, B=0.00, T=1.40),
    "NON_DIGITAL": dict(C=0.00, B=0.85, T=0.85),
}


def generate(n: int = 20000, seed: int = 11):
    rng = np.random.default_rng(seed)
    ids = np.array([f"K{1_000_000 + i}" for i in range(n)])
    path = rng.choice(list(DRIVER_MIX), n, p=[0.55, 0.15, 0.30])
    open_month = np.where(path == "DIGITAL_NTB", rng.integers(T0 - 6, T0 - 1, n), rng.integers(-60, T0 - 13, n))

    C, B, T = rng.normal(size=(3, n))
    mix = pd.DataFrame([DRIVER_MIX[p] for p in path])
    D = mix["C"].to_numpy() * C + mix["B"].to_numpy() * B + mix["T"].to_numpy() * T + rng.normal(0, 0.9, n)

    # dormancy month after t0 (or never within horizon)
    hazard = 0.012 * np.exp(0.95 * D)
    dorm_after = rng.geometric(np.clip(hazard, 1e-4, 0.9))
    dorm_month = np.where(T0 + dorm_after < N_MONTHS, T0 + dorm_after, 10_000)

    months = np.arange(N_MONTHS)
    M = months[None, :]
    alive = M >= open_month[:, None]
    active = alive & (M < dorm_month[:, None])
    rel = np.clip(M - (T0 - 12), 0, None)  # months into the observation window

    # ---------------- transactions: velocity decays with T ----------------
    lam = rng.gamma(2.5, 6, n)
    # new customers' early engagement moves faster, so their velocity reacts more sharply
    k_T = np.where(path == "DIGITAL_NTB", 0.16, 0.045)[:, None]
    decay = np.exp(-k_T * T[:, None] * rel) * np.where(M >= T0, np.exp(-k_T * T[:, None] * 12), 1)
    txn = rng.poisson(lam[:, None] * decay * active)

    # ---------------- balances: drift with B, some overdrafts ----------------
    base_bal = rng.lognormal(11.2, 1.1, n)
    od = rng.random(n) < 0.06
    drift = np.exp(-0.05 * B[:, None] * rel + rng.normal(0, 0.12, (n, N_MONTHS)))
    bal = base_bal[:, None] * drift
    bal = np.where(od[:, None], bal - 1.6 * base_bal[:, None] * (B[:, None] > 0.5) * (rel / 12), bal)
    bal = np.where(alive, bal, np.nan).round(2)

    # ---------------- channels: success rate falls with C ----------------
    digital = path != "NON_DIGITAL"
    attempt_rate = np.where(path == "DIGITAL_NTB", rng.gamma(1.2, 2.0, n), rng.gamma(2.5, 6.0, n))
    p_succ = 1 / (1 + np.exp(-(2.6 - 0.85 * C)))

    rows = []
    ch_share = rng.dirichlet([3, 1.5, 1.5, 2], n)
    for k, ch in enumerate(CHANNELS):
        att = rng.poisson(attempt_rate[:, None] * ch_share[:, [k]] * decay * active * digital[:, None])
        p = np.clip(p_succ[:, None] + rng.normal(0, 0.03, att.shape) + (0.04 if ch == "CARD" else 0), 0.01, 0.995)
        suc = rng.binomial(att, p)
        df = pd.DataFrame({
            "customer_id": np.repeat(ids, N_MONTHS),
            "month_idx": np.tile(months, n),
            "channel": ch,
            "attempts": att.ravel(),
            "successes": suc.ravel(),
        })
        rows.append(df[df.attempts > 0])
    channel = pd.concat(rows, ignore_index=True)

    def panel(values, name):
        df = pd.DataFrame({"customer_id": np.repeat(ids, N_MONTHS), "month_idx": np.tile(months, n), name: values.ravel()})
        return df.dropna()

    txn_df = panel(np.where(alive, txn, np.nan), "txn_count")
    txn_df["txn_count"] = txn_df.txn_count.astype(int)
    bal_df = panel(bal, "avg_balance")

    customers = pd.DataFrame({
        "customer_id": ids,
        "open_month_idx": open_month,
        "is_digital": digital.astype(int),
        "customer_path": path,  # convenience; also derivable from open date + digital activity
    })
    return customers, channel, bal_df, txn_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cust, ch, bal, txn = generate(a.customers, a.seed)
    cust.to_parquet(a.out / "customers.parquet", index=False)
    ch.to_parquet(a.out / "channel_monthly.parquet", index=False)
    bal.to_parquet(a.out / "balance_monthly.parquet", index=False)
    txn.to_parquet(a.out / "txn_monthly.parquet", index=False)
    print(f"customers {len(cust):,} | channel rows {len(ch):,} | balance rows {len(bal):,} | txn rows {len(txn):,}")


if __name__ == "__main__":
    main()
