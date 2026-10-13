"""Automatic checks on the carry strategy. Run with:  pytest"""

import numpy as np
import pandas as pd
import pytest

from backtest.carry_strategy import SIDE_DV01, run

DATES = pd.bdate_range("2025-01-01", "2025-06-30").strftime("%Y-%m-%d")
ISINS = [f"B{k}" for k in range(15)]


def frozen_market():
    """15 bonds, yields almost unchanged; bond k earns k/10 bp of carry per month."""
    rng = np.random.default_rng(0)
    # yields stay at 4% with tiny random moves (0.1 bp), so that volatility is not zero
    ytm = pd.DataFrame(4.0 + rng.normal(0, 1e-3, (len(DATES), 15)), index=DATES, columns=ISINS)
    carry = pd.DataFrame([[k / 10 for k in range(15)]] * len(DATES), index=DATES, columns=ISINS)
    maturity = pd.Series([f"{2030 + k}-01-01" for k in range(15)], index=ISINS)
    return {"ytm": ytm, "carry": carry, "total": carry, "maturity": maturity}


def test_long_high_carry_short_low_carry():
    _, book = run(frozen_market(), cost_bp=0.0)
    assert len(book) >= 4                                  # one rebalance per month
    assert (book["score_gap"] > 0).all()                   # long side always has the higher score


def test_flipped_is_the_mirror_image():
    data = frozen_market()
    normal, _ = run(data, 0.0)
    flipped, _ = run(data, 0.0, flip=True)
    assert flipped.sum() == pytest.approx(-normal.sum())


def test_costs_are_charged_on_turnover():
    data = frozen_market()
    free, book = run(data, 0.0)
    paid, _ = run(data, 0.25)
    # the first rebalance trades the full book: 2 sides x 10,000 € DV01 x 0.25 bp
    assert free.sum() - paid.sum() >= 0.25 * 2 * SIDE_DV01 - 1e-6
