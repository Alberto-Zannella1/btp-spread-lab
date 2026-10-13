"""Turn rich/cheap residuals into trading signals.

Usage (from the project folder):  python -m signals.zscore

Why we need this step: a residual of -8 bp does not mean 'sell'. Some bonds
are ALWAYS rich or cheap for structural reasons (low-coupon bonds trade rich,
new long bonds trade cheap...). The signal is not the level of the residual
but how far today's residual is from that bond's OWN normal level.

For each bond, every day:
- mean and std of its residual over the last 60 trading days (about 3 months);
- z-score = (residual today - mean) / std.
  z = +2 -> the bond is 2 standard deviations cheaper than usual: candidate to BUY.
  z = -2 -> 2 standard deviations richer than usual: candidate to SELL.

For each bond, once:
- half-life of the residual, from an AR(1) model: how many days it takes, on
  average, for a deviation to shrink by half. A signal is only useful if the
  bond goes back to normal quickly; with a half-life of 6 months the trade
  would take too long and cost too much to hold.

Writes two tables to data/btp.db: 'signals' (bond x day) and 'bond_stats' (one row per bond).
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"
WINDOW = 60          # trading days in the rolling window
MIN_OBS = 40         # need at least this many days before computing a z-score
Z_ENTRY = 2.0        # |z| above this is a signal
MAX_HALF_LIFE = 20   # trading days: slower than this is not worth trading


def add_zscores(fit: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, std and z-score of the residual, bond by bond.

    Each window only uses data up to that day, never later: what a trader
    would actually have known on that morning.
    """
    df = fit.sort_values(["isin", "date"]).copy()
    rolling = df.groupby("isin")["residual_bp"].rolling(WINDOW, min_periods=MIN_OBS)
    df["res_mean"] = rolling.mean().reset_index(level=0, drop=True)
    df["res_std"] = rolling.std().reset_index(level=0, drop=True)
    df["zscore"] = (df["residual_bp"] - df["res_mean"]) / df["res_std"]
    return df


def half_life(residuals: pd.Series) -> tuple[float, float]:
    """Half-life (in trading days) of a mean-reverting series, from an AR(1) fit.

    We regress the daily change on yesterday's level:
        change_t = a + b * residual_(t-1) + noise
    b < 0 means the series is pulled back towards its mean (-a/b) every day.
    Each day the gap to the mean shrinks by the factor (1 + b), so after h days
    it is (1 + b)^h of what it was. Setting that to 1/2:
        h = -ln(2) / ln(1 + b)
    If b >= 0 there is no pull back: the half-life is infinite.
    Returns (half-life, b).
    """
    x = residuals.dropna().to_numpy()
    if len(x) < MIN_OBS:
        return np.nan, np.nan
    level, change = x[:-1], np.diff(x)
    b = np.polyfit(level, change, 1)[0]  # slope of the straight line through the points
    if b >= 0 or b <= -1:
        return np.inf, b
    return -np.log(2) / np.log(1 + b), b


def bond_stats(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for isin, g in signals.groupby("isin"):
        hl, b = half_life(g["residual_bp"])
        last = g.iloc[-1]
        rows.append({"isin": isin, "n_days": len(g), "res_avg": g["residual_bp"].mean(),
                     "res_std": g["residual_bp"].std(), "half_life_days": hl, "ar1_b": b,
                     "last_date": last["date"], "last_residual": last["residual_bp"], "last_z": last["zscore"]})
    return pd.DataFrame(rows)


def load_fit() -> pd.DataFrame:
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql("SELECT isin, date, residual_bp FROM curve_fit", conn)


def save(signals: pd.DataFrame, stats: pd.DataFrame) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        signals.to_sql("signals", conn, if_exists="replace", index=False)
        stats.to_sql("bond_stats", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals ON signals (isin, date)")


if __name__ == "__main__":
    signals = add_zscores(load_fit())
    stats = bond_stats(signals)
    save(signals, stats)
    print(f"Saved signals ({len(signals)} rows) and bond_stats ({len(stats)} bonds)")

    with sqlite3.connect(DB_FILE) as conn:
        names = pd.read_sql("SELECT isin, description, maturity FROM bonds", conn)
    stats = stats.merge(names, on="isin")

    finite = stats["half_life_days"].replace(np.inf, np.nan)
    print(f"\nHalf-life of residuals (trading days): median {finite.median():.1f}, "
          f"{(finite <= MAX_HALF_LIFE).sum()} of {len(stats)} bonds below {MAX_HALF_LIFE} days")

    last_day = signals["date"].max()
    tradeable = stats[(stats["last_date"] == last_day) & (stats["half_life_days"] <= MAX_HALF_LIFE)]
    cols = ["description", "last_residual", "res_avg", "last_z", "half_life_days"]
    rounding = {"last_residual": 1, "res_avg": 1, "last_z": 2, "half_life_days": 1}

    print(f"\n{last_day} - CHEAP vs own history (z >= {Z_ENTRY}, half-life <= {MAX_HALF_LIFE}d): candidates to BUY")
    buys = tradeable[tradeable["last_z"] >= Z_ENTRY].sort_values("last_z", ascending=False)
    print(buys[cols].round(rounding).to_string(index=False) if len(buys) else "  none today")

    print(f"\n{last_day} - RICH vs own history (z <= -{Z_ENTRY}, half-life <= {MAX_HALF_LIFE}d): candidates to SELL")
    sells = tradeable[tradeable["last_z"] <= -Z_ENTRY].sort_values("last_z")
    print(sells[cols].round(rounding).to_string(index=False) if len(sells) else "  none today")

    print("\nAll bonds today, sorted by z-score:")
    today = stats[stats["last_date"] == last_day].sort_values("last_z", ascending=False)
    print(today[cols].round(rounding).to_string(index=False))
