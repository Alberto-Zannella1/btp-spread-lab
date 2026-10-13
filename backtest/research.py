"""Phase 6b: why does the switch strategy lose, and can we do better honestly?

Usage (from the project folder):  python -m backtest.research

1. DIAGNOSIS on the switch strategy, before costs:
   - P&L split into residual (the bet), curve (unwanted risk) and carry;
   - cost of the one-day execution delay (lag 0 vs lag 1, lag 0 is NOT tradable).
2. BUTTERFLIES: buy the bond, sell its two neighbours, weighted to be neutral
   to both level and slope of the curve. Same diagnosis.
3. HONEST SELECTION. We try 6 configurations (switch or butterfly, entry z of
   2, 2.5 or 3), all with 0.25 bp cost per leg. We pick the best one looking ONLY
   at the first half of the period (in-sample), then report how it did in the
   second half (out-of-sample), which played no role in the choice.
   Choosing on the whole period and reporting the result would be overfitting:
   with enough tries, something always looks good on the past by chance.

Writes the chart output/research_pnl.png and the table 'research_results'.
"""

from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest.engine import Params, metrics, run
from backtest.run_backtest import DB_FILE, load
from signals.switches import neighbours

ROOT = Path(__file__).parent.parent
BASE_COST = 0.25
Z_GRID = [2.0, 2.5, 3.0]


def build_butterflies(bonds: pd.DataFrame, isins: list, last_date: str) -> list:
    """For each bond: its nearest shorter and longer neighbours, weighted to hedge level and slope."""
    years = {b.isin: (pd.Timestamp(b.maturity) - pd.Timestamp(last_date)).days / 365.25
             for b in bonds.itertuples() if b.isin in isins}
    ordered = sorted(years, key=years.get)
    flies = []
    for k in range(1, len(ordered) - 1):
        s, a, l = ordered[k - 1], ordered[k], ordered[k + 1]
        ts, ta, tl = years[s], years[a], years[l]
        if tl - ts < 0.25 or not (neighbours(ts, ta) and neighbours(ta, tl)):
            continue
        w_long = (ta - ts) / (tl - ts)   # so that wS*T_S + wL*T_L = T_A
        w_short = 1 - w_long             # so that wS + wL = 1
        flies.append(((a, 1.0), (s, -w_short), (l, -w_long)))
    return flies


def decompose(trades: pd.DataFrame) -> dict:
    """Average P&L per trade (bp), split into its sources."""
    return {k: trades[k].mean() for k in ["pnl_bp", "resid_bp", "curve_bp", "carry_bp", "cost_bp"]}


def split_metrics(trades, daily, mid):
    active = daily.loc[daily.ne(0).idxmax():] if daily.ne(0).any() else daily
    first, second = active.loc[:mid], active.loc[mid:].iloc[1:]
    if trades.empty:
        return metrics(trades, first), metrics(trades, second)
    return (metrics(trades[trades["entry"] <= mid], first), metrics(trades[trades["entry"] > mid], second))


def line(name, m):
    return (f"{name:<26} trades {m['trades']:>4}  hit {m['hit_rate_%']:5.1f}%  avg {m['avg_pnl_bp']:+5.2f} bp  "
            f"P&L/year {m['pnl_per_year_eur']:>10,.0f} €  Sharpe {m['sharpe']:5.2f}")


if __name__ == "__main__":
    data, pairs, bonds = load()
    dates = data["residual"].index
    flies = build_butterflies(bonds, list(data["residual"].columns), dates[-1])
    print(f"{len(dates)} days, {len(pairs)} switches, {len(flies)} butterflies\n")

    # ---------------------------------------------------------------- 1-2. diagnosis
    print("DIAGNOSIS before costs - average P&L per trade (bp)")
    print(f"{'':<38}{'total':>7}{'residual':>10}{'curve':>8}{'carry':>8}")
    for name, items in [("switches", pairs), ("butterflies", flies)]:
        for lag in (1, 0):
            tr, _ = run(data, items, Params(cost_bp=0.0, execution_lag=lag))
            d = decompose(tr)
            label = f"{name}, " + ("trade next day" if lag else "same day (NOT tradable)")
            print(f"{label:<38}{d['pnl_bp']:+7.2f}{d['resid_bp']:+10.2f}{d['curve_bp']:+8.2f}{d['carry_bp']:+8.2f}")

    # ---------------------------------------------------------------- 3. honest selection
    active_dates = dates[100:]  # z-scores and weekly half-lives need about 100 days of history
    mid = active_dates[len(active_dates) // 2]
    print(f"\nSELECTION with cost {BASE_COST} bp per leg. In-sample up to {mid}, out-of-sample after.")
    rows, curves = [], {}
    for name, items in [("switch", pairs), ("butterfly", flies)]:
        for z in Z_GRID:
            tr, daily = run(data, items, Params(cost_bp=BASE_COST, z_entry=z))
            ins, outs = split_metrics(tr, daily, mid)
            label = f"{name} z>={z}"
            curves[label] = daily
            rows.append({"config": label, "in_trades": ins["trades"], "in_avg_bp": ins["avg_pnl_bp"],
                         "in_sharpe": ins["sharpe"], "out_trades": outs["trades"],
                         "out_avg_bp": outs["avg_pnl_bp"], "out_sharpe": outs["sharpe"],
                         "out_pnl_year_eur": outs["pnl_per_year_eur"]})
    res = pd.DataFrame(rows)
    print(res.round({"in_avg_bp": 2, "in_sharpe": 2, "out_avg_bp": 2, "out_sharpe": 2, "out_pnl_year_eur": 0})
          .to_string(index=False))

    best = res.loc[res["in_sharpe"].idxmax(), "config"]
    chosen = res[res["config"] == best].iloc[0]
    print(f"\nChosen on the in-sample period only: {best} (in-sample Sharpe {chosen['in_sharpe']:.2f})")
    print(f"Its out-of-sample result: Sharpe {chosen['out_sharpe']:.2f}, "
          f"{chosen['out_avg_bp']:+.2f} bp per trade, {chosen['out_pnl_year_eur']:,.0f} € per year")

    with sqlite3.connect(DB_FILE) as conn:
        res.to_sql("research_results", conn, if_exists="replace", index=False)

    (ROOT / "output").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in ["switch z>=2.0", "butterfly z>=2.0", best]:
        ax.plot(pd.to_datetime(curves[label].index), curves[label].cumsum() / 1000, lw=1.6,
                label=label + ("  (chosen in-sample)" if label == best else ""))
    ax.axvline(pd.Timestamp(mid), color="black", ls="--", lw=1)
    ax.text(pd.Timestamp(mid), ax.get_ylim()[1] * 0.92, "  out-of-sample →", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Cumulative P&L after costs ({BASE_COST} bp per leg), thousand euro, 10mn main leg")
    ax.set_ylabel("thousand €")
    ax.grid(alpha=0.3)
    ax.legend()
    out = ROOT / "output" / "research_pnl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
