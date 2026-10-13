"""Full analytics for one bond, to compare with a published yield.

Usage (from the project folder):
    python -m pricing.check_bond IT0005607970                     latest price in the database
    python -m pricing.check_bond IT0005607970 96.43 2026-10-14    your own price and trade date

Borsa Italiana shows a gross yield ('rendimento effettivo a scadenza lordo')
with ANNUAL compounding, computed from its own reference price. To compare,
use that same price and date and look at the 'annual' line below.
"""

from datetime import date
import sqlite3
import sys

import pandas as pd

from pricing.bond_math import make_bond, risk_measures, to_annual
from pricing.run_pricing import DB_FILE


def main(isin: str, price: float | None, trade_date: date | None) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        bond_row = pd.read_sql("SELECT * FROM bonds WHERE isin = ?", conn, params=(isin,))
        if bond_row.empty:
            sys.exit(f"{isin} is not in the BTP universe.")
        if price is None:
            last = pd.read_sql("SELECT date, price FROM prices WHERE isin = ? ORDER BY date DESC LIMIT 1",
                               conn, params=(isin,))
            price, trade_date = float(last["price"].iloc[0]), date.fromisoformat(last["date"].iloc[0])

    b = bond_row.iloc[0]
    bond = make_bond(b["coupon"], date.fromisoformat(b["issue_date"]), date.fromisoformat(b["maturity"]))
    r = risk_measures(bond, price, trade_date)

    print(f"{b['description']}  ({isin})")
    print(f"Trade date        {trade_date}   settlement {r['settle_date']}")
    print(f"Clean price       {r['clean']:.4f}")
    print(f"Accrued           {r['accrued']:.6f}")
    print(f"Dirty price       {r['dirty']:.4f}")
    print(f"Yield semiannual  {r['ytm']:.4f} %   (market convention for BTPs)")
    print(f"Yield annual      {to_annual(r['ytm']):.4f} %   (compare with Borsa Italiana)")
    print(f"Modified duration {r['mod_duration']:.3f}")
    print(f"Convexity         {r['convexity']:.2f}")
    print(f"DV01              {r['dv01']:.0f} EUR per 1mn nominal")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 3):
        sys.exit(__doc__)
    main(args[0],
         float(args[1]) if len(args) == 3 else None,
         date.fromisoformat(args[2]) if len(args) == 3 else None)
