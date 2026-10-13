"""Charts of the BTP curve.

Usage (from the project folder):
    python -m curve.plot_curve              latest day
    python -m curve.plot_curve 2026-03-02   a specific day

Saves two PNG files in output/:
- btp_curve_<date>.png   market yields vs the fitted par curve, cheapest/richest bonds labelled
- btp_bund_spread.png    BTP-Bund spread history at 2/5/10/30 years
"""

from pathlib import Path
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")  # draw to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from curve.nss import PARAM_NAMES, par_yield

ROOT = Path(__file__).parent.parent
DB_FILE = ROOT / "data" / "btp.db"
OUT_DIR = ROOT / "output"
N_LABELS = 3  # how many cheap and rich bonds to label


def short_name(description: str) -> str:
    """'BTP 3,85% 1.8.2024 - 1.2.2035' -> 'BTP 3.85% Feb35'"""
    parts = description.split()
    coupon = parts[1].replace(",", ".")
    day, month, year = parts[-1].split(".")
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"BTP {coupon} {months[int(month) - 1]}{year[2:]}"


def plot_curve(day: str) -> Path:
    with sqlite3.connect(DB_FILE) as conn:
        fit = pd.read_sql("""
            SELECT f.*, b.description, b.maturity FROM curve_fit f JOIN bonds b ON b.isin = f.isin
            WHERE f.date = ?""", conn, params=(day,))
        params = pd.read_sql("SELECT * FROM curve_params WHERE date = ?", conn, params=(day,))
    if fit.empty:
        sys.exit(f"No curve for {day}. Use a trading day present in the data.")

    fit["years"] = (pd.to_datetime(fit["maturity"]) - pd.Timestamp(day)).dt.days / 365.25
    p = params[PARAM_NAMES].iloc[0].to_numpy()
    grid = np.linspace(1, fit["years"].max(), 200)
    curve = [par_yield(t, p) for t in grid]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(grid, curve, color="#1f4e79", lw=2, label="Fitted NSS par curve")
    sc = ax.scatter(fit["years"], fit["ytm"], c=fit["residual_bp"], cmap="RdYlGn", vmin=-8, vmax=8,
                    edgecolor="black", s=45, zorder=3)
    fig.colorbar(sc, ax=ax, label="Residual (bp): green = cheap, red = rich")

    labelled = pd.concat([fit.nlargest(N_LABELS, "residual_bp"), fit.nsmallest(N_LABELS, "residual_bp")])
    for r in labelled.itertuples():
        ax.annotate(f"{short_name(r.description)} ({r.residual_bp:+.1f})", (r.years, r.ytm),
                    xytext=(6, -12 if r.residual_bp < 0 else 8), textcoords="offset points", fontsize=8)

    rmse = params["rmse_bp"].iloc[0]
    ax.set_title(f"BTP curve on {day}  -  {len(fit)} bonds, fit RMSE {rmse:.1f} bp")
    ax.set_xlabel("Years to maturity")
    ax.set_ylabel("Yield to maturity (%, semiannual)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    out = OUT_DIR / f"btp_curve_{day}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spread() -> Path:
    with sqlite3.connect(DB_FILE) as conn:
        spread = pd.read_sql("SELECT * FROM btp_bund", conn, parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(11, 5))
    for tenor in (2, 5, 10, 30):
        ax.plot(spread["date"], spread[f"spread_{tenor}y_bp"], label=f"{tenor}y", lw=1.3)
    ax.set_title("BTP-Bund zero-coupon spread (bp)")
    ax.set_ylabel("bp")
    ax.grid(alpha=0.3)
    ax.legend()

    out = OUT_DIR / "btp_bund_spread.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        day = sys.argv[1]
    else:
        with sqlite3.connect(DB_FILE) as conn:
            day = pd.read_sql("SELECT MAX(date) AS d FROM curve_fit", conn)["d"].iloc[0]
    print(f"Saved {plot_curve(day)}")
    print(f"Saved {plot_spread()}")
