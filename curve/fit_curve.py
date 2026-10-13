"""Fit the BTP curve every day and measure how rich or cheap each bond is.

Usage (from the project folder):  python -m curve.fit_curve

For each day:
1. take every BTP's cash flows still to be paid and its market dirty price;
2. fit a Nelson-Siegel-Svensson curve through all of them (curve/nss.py);
3. reprice each bond on the curve and compute its 'model yield';
4. residual = market yield - model yield, in basis points.
   Positive -> the bond yields MORE than the curve says: it is CHEAP.
   Negative -> the bond yields LESS than the curve says: it is RICH.

Writes three tables to data/btp.db:
- curve_params   the 6 NSS parameters and the fit error (RMSE) for each day
- curve_fit      market yield, model yield and residual for each bond and day
- btp_bund       BTP and Bund zero rates at 2/5/10/30 years and their spread
"""

from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from curve.nss import PARAM_NAMES, fit, model_dirty_prices, starting_points, zero_rate
from pricing.bond_math import make_bond, to_py, ytm_ql

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"
MIN_BONDS = 8            # do not fit a day with fewer bonds than this
SPREAD_TENORS = [2, 5, 10, 30]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(DB_FILE) as conn:
        analytics = pd.read_sql("SELECT isin, date, settle_date, clean, accrued, dirty, ytm, mod_duration FROM analytics", conn)
        bonds = pd.read_sql("SELECT isin, coupon, issue_date, maturity FROM bonds", conn)
        rates = pd.read_sql("SELECT * FROM reference_rates", conn)
    return analytics, bonds, rates


def cash_flow_table(bonds: pd.DataFrame) -> dict:
    """Payment dates and amounts of every bond, built once with QuantLib."""
    table = {}
    for b in bonds.itertuples():
        bond = make_bond(b.coupon, date.fromisoformat(b.issue_date), date.fromisoformat(b.maturity))
        flows = [(to_py(cf.date()), cf.amount()) for cf in bond.cashflows()]
        table[b.isin] = (bond, flows)
    return table


def day_arrays(day: pd.DataFrame, flows: dict):
    """Flatten one day's cash flows into arrays: time in years, amount, bond number."""
    settle = date.fromisoformat(day["settle_date"].iloc[0])
    times, amounts, index = [], [], []
    for i, isin in enumerate(day["isin"]):
        for pay_date, amount in flows[isin][1]:
            if pay_date > settle:
                times.append((pay_date - settle).days / 365.0)  # ACT/365, standard for zero curves
                amounts.append(amount)
                index.append(i)
    return np.array(times), np.array(amounts), np.array(index)


def fit_all(analytics: pd.DataFrame, flows: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    params_rows, fit_rows = [], []
    previous = None
    for day_str, day in analytics.groupby("date"):
        if len(day) < MIN_BONDS:
            continue
        day = day.reset_index(drop=True)
        times, amounts, index = day_arrays(day, flows)
        market = day["dirty"].to_numpy()
        price_per_bp = (day["mod_duration"] * day["dirty"] * 1e-4).to_numpy()

        # Try several starting points and keep the best fit
        fits = [fit(times, amounts, index, market, price_per_bp, x0) for x0 in starting_points(previous)]
        params = min(fits, key=lambda f: f[1])[0]
        previous = params

        # Reprice each bond on the curve and turn the model price into a model yield
        model_dirty = model_dirty_prices(params, times, amounts, index, len(day))
        trade_date = date.fromisoformat(day_str)
        for i, r in day.iterrows():
            model_clean = model_dirty[i] - r["accrued"]
            model_ytm = ytm_ql(flows[r["isin"]][0], model_clean, trade_date)
            fit_rows.append({"isin": r["isin"], "date": day_str, "ytm": r["ytm"],
                             "model_ytm": model_ytm, "residual_bp": (r["ytm"] - model_ytm) * 100})

        residuals = np.array([row["residual_bp"] for row in fit_rows[-len(day):]])
        params_rows.append({"date": day_str, **dict(zip(PARAM_NAMES, params)),
                            "n_bonds": len(day), "rmse_bp": np.sqrt(np.mean(residuals ** 2))})
    return pd.DataFrame(params_rows), pd.DataFrame(fit_rows)


def btp_bund_spread(curve_params: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """BTP zero rates from our curve vs Bund zero rates from the Bundesbank.

    Both are NSS zero rates with continuous compounding, so the difference is
    a clean like-for-like spread.
    """
    out = curve_params[["date"]].copy()
    params = curve_params[PARAM_NAMES].to_numpy()
    for tenor in SPREAD_TENORS:
        out[f"btp_{tenor}y"] = [float(zero_rate(tenor, p)) for p in params]
    out = out.merge(rates, on="date", how="left")
    for tenor in SPREAD_TENORS:
        out[f"spread_{tenor}y_bp"] = (out[f"btp_{tenor}y"] - out[f"bund_{tenor}y"]) * 100
    return out


def save(tables: dict) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_curve_fit ON curve_fit (isin, date)")


if __name__ == "__main__":
    analytics, bonds, rates = load_inputs()
    flows = cash_flow_table(bonds)
    print(f"Fitting the curve on {analytics['date'].nunique()} days...")

    curve_params, curve_fit = fit_all(analytics, flows)
    spread = btp_bund_spread(curve_params, rates)
    save({"curve_params": curve_params, "curve_fit": curve_fit, "btp_bund": spread})
    print(f"Saved curve_params ({len(curve_params)} days), curve_fit ({len(curve_fit)} rows), btp_bund")

    print("\nFit quality, RMSE in bp (how far bonds sit from the curve on average):")
    print(curve_params["rmse_bp"].describe().round(2).to_string())

    last = curve_fit["date"].max()
    latest = curve_fit[curve_fit["date"] == last].merge(
        pd.read_sql("SELECT isin, description FROM bonds", sqlite3.connect(DB_FILE)), on="isin")
    cols = ["description", "ytm", "model_ytm", "residual_bp"]
    rounding = {"ytm": 3, "model_ytm": 3, "residual_bp": 1}
    print(f"\n{last} - CHEAPEST bonds (yield above the curve):")
    print(latest.nlargest(5, "residual_bp")[cols].round(rounding).to_string(index=False))
    print(f"\n{last} - RICHEST bonds (yield below the curve):")
    print(latest.nsmallest(5, "residual_bp")[cols].round(rounding).to_string(index=False))

    print("\nBTP-Bund spread, last 5 days (bp):")
    spread_cols = ["date"] + [f"spread_{t}y_bp" for t in SPREAD_TENORS]
    print(spread[spread_cols].tail(5).round(1).to_string(index=False))
