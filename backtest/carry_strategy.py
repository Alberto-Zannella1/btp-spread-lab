"""Strategy 2: carry and roll-down (monthly, DV01-neutral).

Usage (from the project folder):  python -m backtest.carry_strategy

The idea is different from the rich/cheap strategies: we do not try to
predict a distortion. We get PAID to hold a risk. Bonds with high carry +
roll-down earn more per month if the curve does not move; we hold them
against bonds with low carry + roll-down.

Rules, fixed BEFORE looking at the results:
- at the close of the last trading day of each month, rank all BTPs by
      score = one-month (carry + roll-down) in bp / one-month yield volatility in bp
  (volatility = std of daily yield changes over the last 60 days x sqrt(21));
- go long the top third and short the bottom third, the next trading day;
- every bond in a side gets the same DV01; each side has a DV01 of 10,000 euro
  per bp (about 10mn nominal of a 10-year BTP), so the portfolio is DV01-neutral;
- hold for one month; at the next rebalance pay 0.25 bp per unit of DV01 traded.

Daily P&L of a bond with DV01 weight w (euro per bp, positive = long):
    w x ( -change in yield in bp  +  carry of the day in bp )
Roll-down is not added separately: if the curve does not move, the bond's
own yield falls as it ages, and that already shows up in the yield change.

Checks: cost sensitivity, flipped strategy, first half vs second half.
"""

from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import metrics

ROOT = Path(__file__).parent.parent
DB_FILE = ROOT / "data" / "btp.db"
SIDE_DV01 = 10_000.0   # euro per bp on each side
VOL_WINDOW = 60
MIN_BONDS = 15         # need at least this many bonds to form the thirds
COSTS = [0.0, 0.25, 0.5, 1.0]


def load() -> dict:
    with sqlite3.connect(DB_FILE) as conn:
        an = pd.read_sql("SELECT isin, date, ytm FROM analytics", conn)
        ca = pd.read_sql("SELECT isin, date, carry_bp, total_bp FROM carry", conn)
        bonds = pd.read_sql("SELECT isin, maturity FROM bonds", conn)
    ytm = an.pivot(index="date", columns="isin", values="ytm").sort_index()
    data = {
        "ytm": ytm,
        "carry": ca.pivot(index="date", columns="isin", values="carry_bp").reindex_like(ytm),
        "total": ca.pivot(index="date", columns="isin", values="total_bp").reindex_like(ytm),
        "maturity": bonds.set_index("isin")["maturity"],
    }
    return data


def month_ends(dates: pd.Index) -> list:
    """Positions (in the date index) of the last trading day of each month."""
    months = pd.to_datetime(dates).to_period("M")
    return [i for i in range(len(dates) - 1) if months[i] != months[i + 1]]


def run(data: dict, cost_bp: float, flip: bool = False, min_bonds: int = MIN_BONDS) -> tuple[pd.Series, pd.DataFrame]:
    ytm, carry, total = data["ytm"], data["carry"], data["total"]
    dates = ytm.index
    vol = (ytm.diff() * 100).rolling(VOL_WINDOW, min_periods=40).std() * np.sqrt(21)
    score = total / vol

    weights = pd.Series(0.0, index=ytm.columns)
    daily = pd.Series(0.0, index=dates)
    book = []
    rebalance_on = {i + 1 for i in month_ends(dates)}  # decide at month end, trade next day

    for i in range(1, len(dates)):
        t, y = dates[i], dates[i - 1]
        # P&L of yesterday's book
        dy = (ytm.loc[t] - ytm.loc[y]) * 100
        pnl = (weights * (-dy + carry.loc[t] / 21)).fillna(0).sum()
        daily.iat[i] += pnl

        if i in rebalance_on:
            s = score.loc[y].dropna()                  # signal known at yesterday's close
            s = s[ytm.loc[t, s.index].notna()]
            if len(s) >= min_bonds:
                n = len(s) // 3
                ranked = s.sort_values(ascending=False)
                longs, shorts = ranked.index[:n], ranked.index[-n:]
                if flip:
                    longs, shorts = shorts, longs
                new = pd.Series(0.0, index=ytm.columns)
                new[longs] = SIDE_DV01 / n
                new[shorts] = -SIDE_DV01 / n
                daily.iat[i] -= cost_bp * (new - weights).abs().sum()   # pay to trade the difference
                yrs = (pd.to_datetime(data["maturity"]) - pd.Timestamp(t)).dt.days / 365.25
                book.append({"date": t, "n_per_side": n,
                             "long_years": yrs[longs].mean(), "short_years": yrs[shorts].mean(),
                             "score_gap": s[longs].mean() - s[shorts].mean()})
                weights = new
    return daily, pd.DataFrame(book)


def show(name: str, daily: pd.Series) -> dict:
    m = metrics(pd.DataFrame(columns=["pnl_bp", "days"]), daily)
    bp = daily.sum() / SIDE_DV01
    print(f"{name:<28} P&L/year {m['pnl_per_year_eur']:>9,.0f} €  total {bp:+6.1f} bp of side DV01  "
          f"Sharpe {m['sharpe']:5.2f}  max DD {m['max_drawdown_eur']:>9,.0f} €")
    return m


if __name__ == "__main__":
    data = load()
    print(f"Carry strategy: {len(data['ytm'])} days, monthly rebalance, DV01 {SIDE_DV01:,.0f} € per bp on each side\n")

    curves = {}
    for cost in COSTS:
        daily, book = run(data, cost)
        curves[f"cost {cost:.2f} bp"] = daily
        show(f"cost {cost:.2f} bp per unit DV01", daily)
    flipped, _ = run(data, 0.0, flip=True)
    curves["flipped, no costs"] = flipped
    show("FLIPPED, no costs", flipped)

    base, book = run(data, 0.25)
    active = base.loc[base.ne(0).idxmax():]
    mid = active.index[len(active) // 2]
    print(f"\nBase case (0.25 bp) split at {mid}:")
    show("first half", active.loc[:mid])
    show("second half", active.loc[mid:].iloc[1:])

    if book.empty:
        raise SystemExit("No rebalance: not enough bonds with a score.")
    print("\nWhat the strategy holds (average over rebalances):")
    print(f"  bonds per side {book['n_per_side'].mean():.0f}, long side maturity {book['long_years'].mean():.1f} years, "
          f"short side maturity {book['short_years'].mean():.1f} years")

    monthly = base.groupby(pd.to_datetime(base.index).to_period("M")).sum()
    monthly = monthly[monthly != 0]
    print(f"  positive months: {(monthly > 0).sum()} of {len(monthly)}")

    with sqlite3.connect(DB_FILE) as conn:
        base.rename("pnl_eur").rename_axis("date").reset_index().to_sql("carry_strategy_daily", conn,
                                                                        if_exists="replace", index=False)
    (ROOT / "output").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, daily in curves.items():
        ax.plot(pd.to_datetime(daily.index), daily.cumsum() / 1000, lw=1.2 if "flip" in label else 1.6,
                ls="--" if "flip" in label else "-", color="grey" if "flip" in label else None, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Carry and roll-down strategy - cumulative P&L (thousand euro, 10k € DV01 per side)")
    ax.set_ylabel("thousand €")
    ax.grid(alpha=0.3)
    ax.legend()
    out = ROOT / "output" / "carry_strategy_pnl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
