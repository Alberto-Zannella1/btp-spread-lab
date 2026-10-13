"""Is the BTP-Bund spread from our curve real market, or model noise?

Usage (from the project folder):  python -m signals.check_spread

The curve-based spread (table btp_bund) depends on the NSS fit. Here we build
a second spread with NO model at all: the average yield of the real BTPs
maturing close to 2 (or 10) years, minus the Bund rate at the same maturity.

If the two spreads move together day by day, the moves are real market.
If the curve spread jumps while the model-free one does not, it is noise.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"
BUCKETS = {2: (1.5, 2.5), 10: (9.0, 11.0)}  # tenor -> range of years to maturity


def model_free_spread(tenor: int) -> pd.DataFrame:
    low, high = BUCKETS[tenor]
    with sqlite3.connect(DB_FILE) as conn:
        a = pd.read_sql("SELECT a.date, a.ytm, b.maturity FROM analytics a JOIN bonds b ON b.isin = a.isin", conn)
        curve = pd.read_sql(f"SELECT date, spread_{tenor}y_bp AS curve_spread, bund_{tenor}y AS bund FROM btp_bund", conn)
    a["years"] = (pd.to_datetime(a["maturity"]) - pd.to_datetime(a["date"])).dt.days / 365.25
    bucket = a[(a["years"] >= low) & (a["years"] <= high)]
    # Bond yields are semiannual, Bund zero rates continuous: convert to compare like with like
    bucket = bucket.assign(ytm_cont=2 * np.log(1 + bucket["ytm"] / 200) * 100)
    btp = bucket.groupby("date").agg(btp_avg=("ytm_cont", "mean"), n_bonds=("ytm_cont", "size")).reset_index()
    out = curve.merge(btp, on="date").dropna()
    out["free_spread"] = (out["btp_avg"] - out["bund"]) * 100
    return out


if __name__ == "__main__":
    for tenor in BUCKETS:
        df = model_free_spread(tenor)
        if df.empty:
            print(f"{tenor}y: no data")
            continue
        changes = df[["curve_spread", "free_spread"]].diff().dropna()
        print(f"\n===== {tenor}-year spread, {len(df)} days, about {df['n_bonds'].median():.0f} BTPs in the bucket =====")
        print(f"Correlation of daily changes, curve vs model-free: {changes.corr().iloc[0, 1]:.2f}")
        print(f"Std of daily changes (bp): curve {changes['curve_spread'].std():.1f}, "
              f"model-free {changes['free_spread'].std():.1f}")
        print("Last 10 days (bp):")
        print(df[["date", "curve_spread", "free_spread", "n_bonds"]].tail(10).round(1).to_string(index=False))
