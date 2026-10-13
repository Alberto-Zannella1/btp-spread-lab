"""Carry and roll-down: what a bond earns in one month if nothing moves.

Usage (from the project folder):  python -m signals.carry

Holding a bond for a month earns (or costs) money even if the market is frozen:

1. CARRY - you receive the coupon but pay to finance the purchase.
   A trader buys the bond with borrowed money (a repo: the bond is the
   collateral). Earning a 4.4% yield while paying 2.4% repo = positive carry.
   We measure it as the break-even: by how many bp the yield could rise over
   the month before the position loses money.
     forward price = what the bond must be worth in a month to break even
                   = dirty today * (1 + repo * days/360) - coupons received
     carry (bp)    = forward yield - today's yield

2. ROLL-DOWN - the bond gets one month shorter. If the curve slopes upward,
   a shorter bond deserves a lower yield, so its price rises.
     roll-down (bp) = model yield today - model yield in a month on the SAME curve

Total (bp) = carry + roll-down. In euro: total x DV01.

The repo rate is proxied by the €STR. Real BTP repo (GC) trades close to it;
a bond in high demand can trade 'special' at a lower repo rate.

Writes the table 'carry' to data/btp.db.
"""

from datetime import date, timedelta
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from curve.nss import PARAM_NAMES, discount_factor
from pricing.bond_math import CALENDAR, accrued_at, make_bond, to_py, to_ql, ytm_at

DB_FILE = Path(__file__).parent.parent / "data" / "btp.db"
HORIZON_DAYS = 30  # calendar days, about one month


def forward_date(settle: date) -> date:
    """Settlement + 30 calendar days, moved to the next TARGET business day if needed."""
    return to_py(CALENDAR.adjust(to_ql(settle + timedelta(days=HORIZON_DAYS))))


def carry_bp(bond, flows, dirty: float, ytm: float, settle: date, fwd: date, repo: float) -> float:
    """Break-even yield change over the horizon, in bp (positive = the position earns)."""
    value = dirty * (1 + repo / 100 * (fwd - settle).days / 360)        # what the financing costs
    for pay_date, amount in flows:
        if settle < pay_date <= fwd:                                     # a coupon paid in the meantime
            value -= amount * (1 + repo / 100 * (fwd - pay_date).days / 360)
    forward_clean = value - accrued_at(bond, fwd)
    return (ytm_at(bond, forward_clean, fwd) - ytm) * 100


def rolldown_bp(bond, flows, params, model_ytm_today: float, fwd: date) -> float:
    """Yield fall from ageing one month on an unchanged curve, in bp."""
    times = np.array([(d - fwd).days / 365.0 for d, _ in flows if d > fwd])
    amounts = np.array([a for d, a in flows if d > fwd])
    model_dirty = (amounts * discount_factor(times, params)).sum()
    model_ytm_fwd = ytm_at(bond, model_dirty - accrued_at(bond, fwd), fwd)
    return (model_ytm_today - model_ytm_fwd) * 100


def load_inputs():
    with sqlite3.connect(DB_FILE) as conn:
        rows = pd.read_sql("""
            SELECT a.isin, a.date, a.settle_date, a.dirty, a.ytm, a.dv01, f.model_ytm
            FROM analytics a JOIN curve_fit f ON f.isin = a.isin AND f.date = a.date""", conn)
        bonds = pd.read_sql("SELECT isin, description, coupon, issue_date, maturity FROM bonds", conn)
        params = pd.read_sql("SELECT * FROM curve_params", conn).set_index("date")
        rates = pd.read_sql("SELECT date, estr FROM reference_rates", conn)
    # €STR is not published every day the bond market is open: carry the last value forward
    repo = rates.set_index("date")["estr"].reindex(sorted(set(rates["date"]) | set(rows["date"]))).ffill().bfill()
    rows["repo"] = rows["date"].map(repo)
    return rows, bonds, params


def compute(rows, bonds, params) -> pd.DataFrame:
    built = {}
    for b in bonds.itertuples():
        bond = make_bond(b.coupon, date.fromisoformat(b.issue_date), date.fromisoformat(b.maturity))
        built[b.isin] = (bond, [(to_py(cf.date()), cf.amount()) for cf in bond.cashflows()])

    out = []
    for r in rows.itertuples():
        bond, flows = built[r.isin]
        settle = date.fromisoformat(r.settle_date)
        fwd = forward_date(settle)
        p = params.loc[r.date, PARAM_NAMES].to_numpy(dtype=float)
        c = carry_bp(bond, flows, r.dirty, r.ytm, settle, fwd, r.repo)
        rd = rolldown_bp(bond, flows, p, r.model_ytm, fwd)
        out.append({"isin": r.isin, "date": r.date, "repo": r.repo, "carry_bp": c, "rolldown_bp": rd,
                    "total_bp": c + rd, "total_eur_per_mn": (c + rd) * r.dv01})
    return pd.DataFrame(out)


if __name__ == "__main__":
    rows, bonds, params = load_inputs()
    print(f"Computing 1-month carry and roll-down for {len(rows)} rows...")
    carry = compute(rows, bonds, params)
    with sqlite3.connect(DB_FILE) as conn:
        carry.to_sql("carry", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_carry ON carry (isin, date)")
    print(f"Saved table 'carry' ({len(carry)} rows)")

    last = carry["date"].max()
    latest = carry[carry["date"] == last].merge(bonds[["isin", "description", "maturity"]], on="isin").sort_values("maturity")
    latest["years"] = (pd.to_datetime(latest["maturity"]) - pd.Timestamp(last)).dt.days / 365.25
    print(f"\n{last} - repo (€STR) {latest['repo'].iloc[0]:.3f}%. One month, in bp of yield and euro per 1mn nominal:")
    cols = ["description", "years", "carry_bp", "rolldown_bp", "total_bp", "total_eur_per_mn"]
    print(latest[cols].round({"years": 1, "carry_bp": 1, "rolldown_bp": 1, "total_bp": 1, "total_eur_per_mn": 0})
          .to_string(index=False))
