"""Run the switch backtest and the honesty checks.

Usage (from the project folder):  python -m backtest.run_backtest

1. Base case: half bid-ask of 0.25 bp per leg (1 bp for a full round trip of a switch).
2. Cost sensitivity: 0, 0.25, 0.5 and 1 bp per leg. A strategy that only works
   with zero costs does not work.
3. Flipped signal: trade against every signal, before costs. If the signal is
   real, this must lose about as much as the strategy gains.
4. First half vs second half of the period: a real effect should show up in both.

Writes tables 'backtest_trades' and 'backtest_daily' and the chart output/backtest_pnl.png.
"""

from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest.engine import Params, metrics, run
from curve.plot_curve import short_name
from signals.switches import neighbours

ROOT = Path(__file__).parent.parent
DB_FILE = ROOT / "data" / "btp.db"
COSTS = [0.0, 0.25, 0.5, 1.0]


def load() -> tuple[dict, list, pd.DataFrame]:
    with sqlite3.connect(DB_FILE) as conn:
        fit = pd.read_sql("SELECT isin, date, residual_bp FROM curve_fit", conn)
        an = pd.read_sql("SELECT isin, date, ytm, dv01 FROM analytics", conn)
        carry = pd.read_sql("SELECT isin, date, carry_bp FROM carry", conn)
        bonds = pd.read_sql("SELECT isin, description, maturity FROM bonds", conn)
    data = {
        "residual": fit.pivot(index="date", columns="isin", values="residual_bp"),
        "ytm": an.pivot(index="date", columns="isin", values="ytm"),
        "dv01": an.pivot(index="date", columns="isin", values="dv01"),
        "carry": carry.pivot(index="date", columns="isin", values="carry_bp"),
    }
    dates = data["residual"].index
    data = {k: v.reindex(index=dates, columns=data["residual"].columns) for k, v in data.items()}

    last = pd.Timestamp(dates[-1])
    years = {b.isin: (pd.Timestamp(b.maturity) - last).days / 365.25 for b in bonds.itertuples()}
    isins = list(data["residual"].columns)
    pairs = [(a, b) for k, a in enumerate(isins) for b in isins[k + 1:] if neighbours(years[a], years[b])]
    return data, pairs, bonds


def show(name: str, m: dict) -> None:
    print(f"{name:<28} trades {m['trades']:>4}  hit {m['hit_rate_%']:5.1f}%  avg {m['avg_pnl_bp']:+5.2f} bp  "
          f"days {m['avg_days']:4.1f}  P&L/year {m['pnl_per_year_eur']:>10,.0f} €  "
          f"Sharpe {m['sharpe']:5.2f}  max DD {m['max_drawdown_eur']:>9,.0f} €")


if __name__ == "__main__":
    data, pairs, bonds = load()
    print(f"Backtest on {len(data['residual'])} days, {len(pairs)} pairs of neighbouring BTPs")
    print("Signal at the close of day t, trade at the price of day t+1. Long leg 10mn nominal, DV01-neutral.\n")

    results = {}
    for cost in COSTS:
        trades, daily = run(data, pairs, Params(cost_bp=cost))
        results[cost] = (trades, daily)
        show(f"cost {cost:.2f} bp per leg", metrics(trades, daily))

    flip_trades, flip_daily = run(data, pairs, Params(cost_bp=0.0), flip=True)
    show("FLIPPED signal, no costs", metrics(flip_trades, flip_daily))

    base_trades, base_daily = results[0.25]
    active = base_daily.loc[base_daily.ne(0).idxmax():]
    mid = active.index[len(active) // 2]
    print(f"\nBase case (0.25 bp) split at {mid}:")
    for label, part in [("first half", active.loc[:mid]), ("second half", active.loc[mid:].iloc[1:])]:
        tr = base_trades[(base_trades["exit"] <= mid) if label == "first half" else (base_trades["exit"] > mid)]
        show(label, metrics(tr, part))

    print("\nHow trades ended (base case):")
    print(base_trades.groupby("reason")["pnl_bp"].agg(["count", "mean"]).round(2).to_string())

    names = bonds.set_index("isin")["description"].map(short_name)
    view = base_trades.assign(buy=base_trades["buy"].map(names), sell=base_trades["sell"].map(names))
    cols = ["buy", "sell", "entry", "exit", "days", "entry_z", "pnl_bp", "pnl_eur", "reason"]
    rounding = {"entry_z": 2, "pnl_bp": 2, "pnl_eur": 0}
    print("\nBest 5 trades:")
    print(view.nlargest(5, "pnl_eur")[cols].round(rounding).to_string(index=False))
    print("\nWorst 5 trades:")
    print(view.nsmallest(5, "pnl_eur")[cols].round(rounding).to_string(index=False))

    with sqlite3.connect(DB_FILE) as conn:
        base_trades.astype({"entry": str, "exit": str}).to_sql("backtest_trades", conn, if_exists="replace", index=False)
        base_daily.rename("pnl_eur").rename_axis("date").reset_index().to_sql("backtest_daily", conn, if_exists="replace", index=False)

    (ROOT / "output").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for cost, (_, daily) in results.items():
        ax.plot(pd.to_datetime(daily.index), daily.cumsum() / 1000, lw=1.6, label=f"cost {cost:.2f} bp per leg")
    ax.plot(pd.to_datetime(flip_daily.index), flip_daily.cumsum() / 1000, lw=1.2, ls="--", color="grey",
            label="flipped signal, no costs")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("BTP switch strategy - cumulative P&L (thousand euro, 10mn long leg, DV01-neutral)")
    ax.set_ylabel("thousand €")
    ax.grid(alpha=0.3)
    ax.legend()
    out = ROOT / "output" / "backtest_pnl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
