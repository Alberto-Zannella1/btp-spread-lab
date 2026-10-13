"""Strategy 3: the auction cycle.

Usage (from the project folder):  python -m backtest.auction_strategy

The idea (documented for US Treasuries by Lou, Yan and Zhang, 2013, and for euro
area bonds by Beetsma and others, 2016): when the Treasury reopens a bond in an
auction, dealers must absorb a lot of new supply. They make room for it by
selling the bond in the days before, so it CHEAPENS relative to its
neighbours; after the auction, as the new supply is placed with investors,
it RECOVERS. Unlike the rich/cheap signals, the dates are known in advance:
no need to guess when the distortion appears.

How we measure it, for every ordinary REOPENING (tranche 2 or later) of a BTP
in our universe:
- a butterfly around the auctioned bond A: A against its nearest shorter and
  longer neighbours, weighted to be neutral to level and slope of the curve
  (bonds auctioned the same day are not used as neighbours);
- fly (bp) = yield_A - wS x yield_S - wL x yield_L. Fly going UP = A cheapening.

Rules, fixed BEFORE looking at the results:
- PRE-auction trade:  sell the fly at the price of day T-3, buy it back at day T (auction day);
- POST-auction trade: buy the fly at day T, sell it at day T+3;
- cost 0.25 bp per unit of DV01 per side: a butterfly trades 2 units of DV01
  (1 for A, 1 split across the wings), so 1 bp per round trip.
Auction dates are announced in advance, so trading on them uses no future information.

Outputs: event study chart output/auction_event_study.png, table 'auction_events'.
"""

from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signals.switches import neighbours

ROOT = Path(__file__).parent.parent
DB_FILE = ROOT / "data" / "btp.db"
WINDOW = 10        # trading days before and after the auction in the event study
PRE, POST = 3, 3   # holding periods of the two trades
COST_BP = 0.25
NOMINAL_MN = 10    # size of the auctioned leg


def load():
    with sqlite3.connect(DB_FILE) as conn:
        an = pd.read_sql("SELECT isin, date, ytm, dv01 FROM analytics", conn)
        bonds = pd.read_sql("SELECT isin, description, maturity FROM bonds", conn)
        auctions = pd.read_sql("SELECT date, isin, tranche, kind, allotted_mln FROM auctions WHERE kind = 'O'", conn)
    ytm = an.pivot(index="date", columns="isin", values="ytm").sort_index()
    dv01 = an.pivot(index="date", columns="isin", values="dv01").reindex_like(ytm)
    return ytm, dv01, bonds, auctions


def fly_for_event(ytm, maturity, date, isin, same_day):
    """Nearest shorter and longer neighbours of the auctioned bond on that date, with their weights."""
    t = pd.Timestamp(date)
    alive = ytm.loc[date].dropna().index
    years = {i: (maturity[i] - t).days / 365.25 for i in alive if i not in same_day or i == isin}
    ta = years[isin]
    shorter = [i for i in years if years[i] < ta and neighbours(years[i], ta)]
    longer = [i for i in years if years[i] > ta and neighbours(years[i], ta)]
    if not shorter or not longer:
        return None
    s = max(shorter, key=years.get)
    l = min(longer, key=years.get)
    w_long = (ta - years[s]) / (years[l] - years[s])
    return s, l, 1 - w_long, w_long


def study(ytm, dv01, bonds, auctions):
    maturity = pd.to_datetime(bonds.set_index("isin")["maturity"])
    dates = list(ytm.index)
    events = auctions[(auctions["tranche"] > 1) & auctions["isin"].isin(ytm.columns) & auctions["date"].isin(dates)]
    paths, rows = [], []
    for e in events.itertuples():
        k = dates.index(e.date)
        if k - WINDOW < 0 or k + WINDOW >= len(dates):
            continue
        same_day = set(auctions.loc[auctions["date"] == e.date, "isin"])
        legs = fly_for_event(ytm, maturity, e.date, e.isin, same_day)
        if legs is None:
            continue
        s, l, ws, wl = legs
        window = ytm.iloc[k - WINDOW: k + WINDOW + 1]
        fly = (window[e.isin] - ws * window[s] - wl * window[l]) * 100
        if fly.isna().any():
            continue
        f = fly.to_numpy()
        pre = f[WINDOW] - f[WINDOW - PRE]        # short fly: gains if A cheapens (fly rises)
        post = -(f[WINDOW + POST] - f[WINDOW])    # long fly: gains if A recovers (fly falls)
        dv01_eur = NOMINAL_MN * dv01.at[e.date, e.isin]
        paths.append(f - f[0])
        rows.append({"date": e.date, "isin": e.isin, "short_wing": s, "long_wing": l, "w_short": ws, "w_long": wl,
                     "allotted_mln": e.allotted_mln, "pre_gross_bp": pre, "post_gross_bp": post,
                     "pre_net_bp": pre - 4 * COST_BP, "post_net_bp": post - 4 * COST_BP, "dv01_eur": dv01_eur})
    return pd.DataFrame(rows), np.array(paths)


def summary(name: str, x: pd.Series, dv01_eur: pd.Series, years: float) -> None:
    n = len(x)
    t = x.mean() / (x.std() / np.sqrt(n)) if n > 1 else np.nan
    eur_year = (x * dv01_eur).sum() / years
    print(f"{name:<26} events {n:>3}  mean {x.mean():+5.2f} bp  t-stat {t:+5.2f}  "
          f"positive {100 * (x > 0).mean():5.1f}%  € per year {eur_year:>9,.0f}")


if __name__ == "__main__":
    ytm, dv01, bonds, auctions = load()
    ev, paths = study(ytm, dv01, bonds, auctions)
    if ev.empty:
        raise SystemExit("No auction events with enough data.")
    years = (pd.Timestamp(ev["date"].max()) - pd.Timestamp(ev["date"].min())).days / 365.25
    print(f"{len(ev)} BTP reopenings from {ev['date'].min()} to {ev['date'].max()} "
          f"(butterfly vs nearest neighbours, {NOMINAL_MN}mn of the auctioned bond)\n")
    print("A t-stat above 2 (in absolute value) means the average is unlikely to be luck.")
    for label, col in [("PRE  gross (T-3 to T)", "pre_gross_bp"), ("PRE  net of 1 bp cost", "pre_net_bp"),
                       ("POST gross (T to T+3)", "post_gross_bp"), ("POST net of 1 bp cost", "post_net_bp")]:
        summary(label, ev[col], ev["dv01_eur"], years)

    mid = ev["date"].iloc[len(ev) // 2]
    print(f"\nSplit at {mid}, net of costs:")
    for label, part in [("first half", ev[ev["date"] < mid]), ("second half", ev[ev["date"] >= mid])]:
        summary(f"PRE  {label}", part["pre_net_bp"], part["dv01_eur"], years / 2)
        summary(f"POST {label}", part["post_net_bp"], part["dv01_eur"], years / 2)

    with sqlite3.connect(DB_FILE) as conn:
        ev.to_sql("auction_events", conn, if_exists="replace", index=False)

    days = np.arange(-WINDOW, WINDOW + 1)
    mean = paths.mean(axis=0)
    se = paths.std(axis=0) / np.sqrt(len(paths))
    (ROOT / "output").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(days, mean, color="#1f4e79", lw=2, marker="o", ms=3, label="average across reopenings")
    ax.fill_between(days, mean - 2 * se, mean + 2 * se, color="#1f4e79", alpha=0.15, label="± 2 standard errors")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.axvspan(-PRE, 0, color="red", alpha=0.06, label=f"pre-auction trade (T-{PRE} to T)")
    ax.axvspan(0, POST, color="green", alpha=0.06, label=f"post-auction trade (T to T+{POST})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Trading days from the auction (0 = auction day)")
    ax.set_ylabel("Change in butterfly yield since day -10 (bp)\nup = auctioned bond cheapens")
    ax.set_title(f"BTP reopenings: auctioned bond vs neighbours ({len(paths)} auctions)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    out = ROOT / "output" / "auction_event_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
