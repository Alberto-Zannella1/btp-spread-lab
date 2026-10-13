"""Turn rich/cheap residuals into trading signals.

Usage (from the project folder):  python -m signals.zscore

Why we need this step: a residual of -8 bp does not mean 'sell'. Some bonds
are ALWAYS rich or cheap for structural reasons (low-coupon bonds trade rich,
new long bonds trade cheap...). The signal is not the level of the residual
but how far today's residual is from that bond's OWN normal level.

Two versions of everything:
- raw:      the daily residual, as it comes out of the curve fit;
- smoothed: the average of the last 5 days. Daily Euronext prices are VWAPs,
            noisy for bonds with few trades; averaging over a week removes most
            of that noise (see the study book, chapter 13).

For each bond, every day:
- z-score = (residual - mean of the last 60 days) / std of the last 60 days,
  on the raw and on the smoothed residual.
  z = +2 -> 2 standard deviations cheaper than usual: candidate to BUY.
  z = -2 -> 2 standard deviations richer than usual: candidate to SELL.

For each bond, once:
- half-life of the residual from an AR(1) model, estimated two ways:
  on daily data (fooled by noise: too short) and on weekly averages sampled
  once a week (the one we trust).

Writes two tables to data/btp.db: 'signals' (bond x day) and 'bond_stats' (one row per bond).
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"
WINDOW = 60          # trading days in the rolling window
MIN_OBS = 40         # need at least this many days before computing a z-score
SMOOTH = 5           # trading days averaged in the smoothed residual (one week)
MIN_WEEKS = 20       # weekly observations needed for the weekly half-life
Z_ENTRY = 2.0        # |z| above this is a signal
MIN_HALF_LIFE = 2    # trading days: faster than this is probably noise
MAX_HALF_LIFE = 30   # trading days: slower than this is not worth trading


def rolling_z(series: pd.Series) -> pd.Series:
    """z-score of each value against the previous WINDOW days (today included, never the future)."""
    roll = series.rolling(WINDOW, min_periods=MIN_OBS)
    return (series - roll.mean()) / roll.std()


def add_zscores(fit: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, std and z-score of the residual, bond by bond, raw and smoothed.

    Each window only uses data up to that day, never later: what a trader
    would actually have known on that morning.
    """
    df = fit.sort_values(["isin", "date"]).copy()
    by_bond = df.groupby("isin")["residual_bp"]
    rolling = by_bond.rolling(WINDOW, min_periods=MIN_OBS)
    df["res_mean"] = rolling.mean().reset_index(level=0, drop=True)
    df["res_std"] = rolling.std().reset_index(level=0, drop=True)
    df["zscore"] = (df["residual_bp"] - df["res_mean"]) / df["res_std"]
    df["res_smooth"] = by_bond.transform(lambda s: s.rolling(SMOOTH).mean())
    df["zscore_smooth"] = df.groupby("isin")["res_smooth"].transform(rolling_z)
    return df


def half_life(series: pd.Series, min_obs: int = MIN_OBS) -> tuple[float, float]:
    """Half-life (in steps of the series) of a mean-reverting series, from an AR(1) fit.

    We regress the change on the previous level:
        change_t = a + b * level_(t-1) + noise
    b < 0 means the series is pulled back towards its mean (-a/b) at every step.
    Each step the gap to the mean shrinks by the factor (1 + b), so after h steps
    it is (1 + b)^h of what it was. Setting that to 1/2:
        h = -ln(2) / ln(1 + b)
    If b >= 0 there is no pull back: the half-life is infinite.
    Returns (half-life, b).
    """
    x = series.dropna().to_numpy()
    if len(x) < min_obs:
        return np.nan, np.nan
    level, change = x[:-1], np.diff(x)
    b = np.polyfit(level, change, 1)[0]  # slope of the straight line through the points
    if b >= 0 or b <= -1:
        return np.inf, b
    return -np.log(2) / np.log(1 + b), b


def weekly_half_life(series: pd.Series) -> float:
    """Half-life in trading days, estimated on weekly averages sampled once a week.

    Averaging over 5 days removes the daily price noise; taking one point per
    week (not the overlapping daily moving average) avoids creating fake
    persistence. The result is in weeks, so we multiply by 5 to get days.
    """
    weekly = series.rolling(SMOOTH).mean().iloc[::-1].iloc[::SMOOTH].iloc[::-1]  # one point every 5 days, ending today
    hl_weeks, _ = half_life(weekly, min_obs=MIN_WEEKS)
    return hl_weeks * SMOOTH


def bond_stats(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for isin, g in signals.groupby("isin"):
        hl_daily, b = half_life(g["residual_bp"])
        last = g.iloc[-1]
        rows.append({"isin": isin, "n_days": len(g), "res_avg": g["residual_bp"].mean(),
                     "res_std": g["residual_bp"].std(), "half_life_daily": hl_daily, "ar1_b": b,
                     "half_life_weekly": weekly_half_life(g["residual_bp"]),
                     "last_date": last["date"], "last_residual": last["residual_bp"],
                     "last_smooth": last["res_smooth"], "last_z": last["zscore"], "last_z_smooth": last["zscore_smooth"]})
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

    daily = stats["half_life_daily"].replace(np.inf, np.nan)
    weekly = stats["half_life_weekly"].replace(np.inf, np.nan)
    print(f"\nMedian half-life of residuals (trading days): daily data {daily.median():.1f}, "
          f"weekly averages {weekly.median():.1f}")
    print(f"Bonds with weekly half-life between {MIN_HALF_LIFE} and {MAX_HALF_LIFE} days: "
          f"{weekly.between(MIN_HALF_LIFE, MAX_HALF_LIFE).sum()} of {len(stats)}")

    last_day = signals["date"].max()
    tradeable = stats[(stats["last_date"] == last_day) & stats["half_life_weekly"].between(MIN_HALF_LIFE, MAX_HALF_LIFE)]
    cols = ["description", "last_residual", "last_smooth", "last_z", "last_z_smooth", "half_life_weekly"]
    rounding = {"last_residual": 1, "last_smooth": 1, "last_z": 2, "last_z_smooth": 2, "half_life_weekly": 1}

    print(f"\n{last_day} - CHEAP vs own history (smoothed z >= {Z_ENTRY}): candidates to BUY")
    buys = tradeable[tradeable["last_z_smooth"] >= Z_ENTRY].sort_values("last_z_smooth", ascending=False)
    print(buys[cols].round(rounding).to_string(index=False) if len(buys) else "  none today")

    print(f"\n{last_day} - RICH vs own history (smoothed z <= -{Z_ENTRY}): candidates to SELL")
    sells = tradeable[tradeable["last_z_smooth"] <= -Z_ENTRY].sort_values("last_z_smooth")
    print(sells[cols].round(rounding).to_string(index=False) if len(sells) else "  none today")
