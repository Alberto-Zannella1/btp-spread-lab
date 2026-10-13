"""Automatic checks on the auction strategy. Run with:  pytest"""

import numpy as np
import pandas as pd
import pytest

from backtest.auction_strategy import COST_BP, study

DATES = pd.bdate_range("2026-01-01", periods=60).strftime("%Y-%m-%d")
AUCTION = DATES[30]


def market():
    """Three bonds 2031-2032-2033. The middle one is reopened on day 30:
    it cheapens by 1 bp a day from day T-3 to the auction day T, then recovers 1 bp a day."""
    ytm = pd.DataFrame({"S": 3.5, "A": 3.6, "L": 3.7}, index=DATES)
    bump = np.zeros(len(DATES))
    bump[28:31] = [1, 2, 3]             # cheapening before the auction (bp), peak on the auction day (30)
    bump[31:34] = [2, 1, 0]             # recovery after
    ytm["A"] += bump / 100
    dv01 = pd.DataFrame(500.0, index=DATES, columns=ytm.columns)
    bonds = pd.DataFrame({"isin": ["S", "A", "L"], "description": ["s", "a", "l"],
                          "maturity": ["2031-01-01", "2032-01-01", "2033-01-01"]})
    auctions = pd.DataFrame({"date": [AUCTION], "isin": ["A"], "tranche": [3], "kind": ["O"], "allotted_mln": [3000]})
    return ytm, dv01, bonds, auctions


def test_pre_and_post_auction_trades_capture_the_cycle():
    ev, paths = study(*market())
    assert len(ev) == 1
    e = ev.iloc[0]
    assert (e["short_wing"], e["long_wing"]) == ("S", "L")
    assert e["pre_gross_bp"] == pytest.approx(3.0)     # day T-3 (bump 0) to T (bump 3)
    assert e["post_gross_bp"] == pytest.approx(3.0)    # day T (3) to T+3 (0)
    assert e["pre_net_bp"] == pytest.approx(3.0 - 4 * COST_BP)


def test_bonds_auctioned_the_same_day_are_not_used_as_wings():
    ytm, dv01, bonds, auctions = market()
    ytm["S2"] = 3.45                                              # another shorter bond, a bit further away
    dv01["S2"] = 450.0
    bonds = pd.concat([bonds, pd.DataFrame({"isin": ["S2"], "description": ["s2"], "maturity": ["2030-10-01"]})])
    auctions = pd.concat([auctions, auctions.assign(isin="S")])  # S is also auctioned that day
    ev, _ = study(ytm, dv01, bonds, auctions)
    assert ev["short_wing"].tolist() == ["S2"]                    # S is skipped, the next one is used
