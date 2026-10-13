"""DV01-neutral switches: buy a cheap BTP, sell a rich neighbour with the same risk.

Usage (from the project folder):  python -m signals.switches

Buying a cheap bond on its own is also a bet on the direction of rates: if all
yields rise 20 bp you lose, however cheap the bond was. A relative value
trader removes that bet with a SWITCH:
- buy bond A (cheap) and sell bond B (rich), with similar maturities;
- size the two legs so they have the SAME DV01: nominal_B = nominal_A * DV01_A / DV01_B.
If the whole curve moves in parallel, the two legs gain and lose the same
amount. The switch makes money only if the distortion between A and B closes.

For every pair of neighbouring bonds we follow the spread of their residuals:
    pair residual = residual_A - residual_B   (bp)
and judge it with the same tools as single bonds: 5-day smoothing, 60-day
z-score, weekly half-life. A good switch today has:
- |z| >= 2 on the smoothed pair residual (the gap is unusually wide);
- weekly half-life between 2 and 30 trading days (it closes, but not instantly);
- carry + roll-down that does not eat the expected gain.

Expected gain (bp) = how far the pair residual is from its 60-day mean
(if it went back there) + one month of net carry and roll-down.
In euro: expected gain x DV01 of the leg.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from curve.plot_curve import short_name
from signals.zscore import MAX_HALF_LIFE, MIN_HALF_LIFE, SMOOTH, WINDOW, Z_ENTRY, rolling_z, weekly_half_life

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"
BUY_NOMINAL = 10_000_000   # size of the long leg, euro nominal
MIN_HISTORY = 120          # trading days of common history needed for a pair


def neighbours(years_a: float, years_b: float) -> bool:
    """Two bonds are comparable if their maturities are close: 1.5 years, or 20% of the maturity for long bonds."""
    return abs(years_a - years_b) <= max(1.5, 0.2 * min(years_a, years_b))


def load():
    with sqlite3.connect(DB_FILE) as conn:
        res = pd.read_sql("SELECT isin, date, residual_bp FROM curve_fit", conn)
        bonds = pd.read_sql("SELECT isin, description, maturity FROM bonds", conn)
        last = pd.read_sql("SELECT MAX(date) AS d FROM curve_fit", conn)["d"].iloc[0]
        today = pd.read_sql("""
            SELECT a.isin, a.dv01, c.total_bp AS carry_roll_bp
            FROM analytics a JOIN carry c ON c.isin = a.isin AND c.date = a.date
            WHERE a.date = ?""", conn, params=(last,))
    residuals = res.pivot(index="date", columns="isin", values="residual_bp").sort_index()
    return residuals, bonds, today, last


def scan(residuals: pd.DataFrame, bonds: pd.DataFrame, today: pd.DataFrame, last: str) -> pd.DataFrame:
    info = bonds.merge(today, on="isin")
    info["years"] = (pd.to_datetime(info["maturity"]) - pd.Timestamp(last)).dt.days / 365.25
    info = info.set_index("isin")
    isins = [i for i in info.index if i in residuals.columns and pd.notna(residuals.loc[last, i])]

    rows = []
    for k, a in enumerate(isins):
        for b in isins[k + 1:]:
            if not neighbours(info.at[a, "years"], info.at[b, "years"]):
                continue
            pair = (residuals[a] - residuals[b]).dropna()
            if len(pair) < MIN_HISTORY:
                continue
            smooth = pair.rolling(SMOOTH).mean()
            z = rolling_z(smooth).iloc[-1]
            if not np.isfinite(z):
                continue
            mean60 = smooth.rolling(WINDOW).mean().iloc[-1]
            # Orient the trade: buy the bond that is cheap relative to the other
            buy, sell, gap, zz = (a, b, smooth.iloc[-1] - mean60, z) if z > 0 else (b, a, mean60 - smooth.iloc[-1], -z)
            rows.append({"buy": buy, "sell": sell, "z": zz, "gap_bp": gap,
                         "half_life": weekly_half_life(pair),
                         "carry_roll_bp": info.at[buy, "carry_roll_bp"] - info.at[sell, "carry_roll_bp"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["expected_bp"] = out["gap_bp"] + out["carry_roll_bp"]
    dv01_buy = out["buy"].map(info["dv01"])
    dv01_sell = out["sell"].map(info["dv01"])
    out["buy_nominal_mn"] = BUY_NOMINAL / 1e6
    out["sell_nominal_mn"] = BUY_NOMINAL * dv01_buy / dv01_sell / 1e6  # same DV01 on both legs
    out["dv01_eur"] = BUY_NOMINAL / 1e6 * dv01_buy
    out["expected_eur"] = out["expected_bp"] * out["dv01_eur"]
    out["buy_name"] = out["buy"].map(info["description"]).map(short_name)
    out["sell_name"] = out["sell"].map(info["description"]).map(short_name)
    return out


if __name__ == "__main__":
    residuals, bonds, today, last = load()
    pairs = scan(residuals, bonds, today, last)
    if pairs.empty:
        print("No pairs with enough history.")
        raise SystemExit

    with sqlite3.connect(DB_FILE) as conn:
        pairs.assign(date=last).to_sql("switches", conn, if_exists="replace", index=False)
    print(f"{last}: scanned {len(pairs)} pairs of neighbouring BTPs (saved to table 'switches')")

    good = pairs[(pairs["z"] >= Z_ENTRY) & pairs["half_life"].between(MIN_HALF_LIFE, MAX_HALF_LIFE)
                 & (pairs["expected_bp"] > 0)].sort_values("expected_bp", ascending=False)  # bp, not euro: euro favour long bonds
    print(f"Switches with z >= {Z_ENTRY}, half-life {MIN_HALF_LIFE}-{MAX_HALF_LIFE} days, positive expected gain: {len(good)}")

    good = good.rename(columns={"half_life": "hl_days", "carry_roll_bp": "carry_bp", "sell_nominal_mn": "sell_mn"})
    cols = ["buy_name", "sell_name", "z", "hl_days", "gap_bp", "carry_bp", "expected_bp", "sell_mn", "expected_eur"]
    rounding = {"z": 2, "hl_days": 1, "gap_bp": 1, "carry_bp": 1, "expected_bp": 1, "sell_mn": 2, "expected_eur": 0}
    pd.set_option("display.width", 250)
    print(f"\nTop 10 by expected bp (buy {BUY_NOMINAL / 1e6:.0f}mn of the first bond, sell the DV01-equivalent of the second):")
    print(good[cols].head(10).round(rounding).to_string(index=False) if len(good) else "  none today")
