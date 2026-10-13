"""Compute yield, duration, DV01 and convexity for every BTP on every day.

Usage (from the project folder):  python -m pricing.run_pricing

Reads bonds and prices from data/btp.db and writes the results to a new
table, 'analytics', in the same database. One row per bond per day.
"""

from pathlib import Path
import sqlite3

import pandas as pd

from pricing.bond_math import make_bond, risk_measures, settlement_date

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"


def load_inputs() -> pd.DataFrame:
    query = """
        SELECT p.isin, p.date, p.price, b.description, b.coupon, b.issue_date, b.maturity
        FROM prices p JOIN bonds b ON b.isin = p.isin
        ORDER BY p.isin, p.date
    """
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(query, conn)
    for col in ["date", "issue_date", "maturity"]:
        df[col] = pd.to_datetime(df[col]).dt.date
    return df


def compute(inputs: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    rows, skipped = [], []
    for isin, group in inputs.groupby("isin"):
        first = group.iloc[0]
        bond = make_bond(first["coupon"], first["issue_date"], first["maturity"])  # build once per bond
        for r in group.itertuples():
            # Trades before the bond exists (grey market) settle before issue: no accrued yet
            if settlement_date(r.date) <= r.issue_date:
                skipped.append((isin, r.date, "settles before issue date"))
                continue
            try:
                rows.append({"isin": isin, "date": r.date, **risk_measures(bond, r.price, r.date)})
            except Exception as error:
                skipped.append((isin, r.date, str(error)[:80]))
    return pd.DataFrame(rows), skipped


def save(analytics: pd.DataFrame) -> None:
    out = analytics.copy()
    for col in ["date", "settle_date"]:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    with sqlite3.connect(DB_FILE) as conn:
        out.to_sql("analytics", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics ON analytics (isin, date)")


if __name__ == "__main__":
    inputs = load_inputs()
    print(f"Input: {len(inputs)} prices for {inputs['isin'].nunique()} bonds")

    analytics, skipped = compute(inputs)
    save(analytics)
    print(f"Saved {len(analytics)} rows to table 'analytics' in {DB_FILE}")

    if skipped:
        print(f"\nSkipped {len(skipped)} rows:")
        for isin, d, reason in skipped[:10]:
            print(f"  {isin} {d}  {reason}")

    # Sanity check: yields should sit in a sensible range for BTPs
    out_of_range = analytics[(analytics["ytm"] < 0) | (analytics["ytm"] > 8)]
    print(f"\nYields outside 0-8%: {len(out_of_range)} rows")

    last_day = analytics["date"].max()
    latest = analytics[analytics["date"] == last_day].merge(
        inputs[["isin", "description", "maturity"]].drop_duplicates(), on="isin"
    ).sort_values("maturity")
    print(f"\nLatest day ({last_day}), from shortest to longest maturity:")
    print(latest[["description", "clean", "accrued", "ytm", "mod_duration", "dv01"]]
          .round({"clean": 2, "accrued": 3, "ytm": 3, "mod_duration": 2, "dv01": 0})
          .to_string(index=False))
