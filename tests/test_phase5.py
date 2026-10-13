"""Automatic checks on smoothing, carry/roll-down and switches. Run with:  pytest"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from curve.nss import discount_factor
from pricing.bond_math import accrued_at, make_bond, risk_measures, to_py, ytm_at
from signals.carry import carry_bp, forward_date, rolldown_bp
from signals.switches import neighbours
from signals.zscore import half_life, weekly_half_life

TRADE = date(2026, 10, 9)


def noisy_ar1(true_half_life=10, n=3000, noise=2.0, seed=0) -> pd.Series:
    """A residual that mean-reverts with a known half-life, plus daily price noise."""
    rng = np.random.default_rng(seed)
    b = -(1 - 0.5 ** (1 / true_half_life))
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 + b) + rng.normal()
    return pd.Series(x + rng.normal(0, noise, n))


def test_daily_half_life_is_fooled_by_noise():
    hl, _ = half_life(noisy_ar1())
    assert hl < 3  # true value is 10: daily data badly underestimates it


def test_weekly_half_life_sees_through_noise():
    assert weekly_half_life(noisy_ar1()) == pytest.approx(10, rel=0.35)


def bond_today(coupon, maturity, price):
    bond = make_bond(coupon, date(2024, 2, 1), maturity)
    flows = [(to_py(cf.date()), cf.amount()) for cf in bond.cashflows()]
    r = risk_measures(bond, price, TRADE)
    return bond, flows, r


def test_carry_sign_depends_on_repo():
    bond, flows, r = bond_today(4.0, date(2036, 2, 1), 98.0)  # yield about 4.3%
    fwd = forward_date(r["settle_date"])
    assert carry_bp(bond, flows, r["dirty"], r["ytm"], r["settle_date"], fwd, repo=2.0) > 0  # earn 4.3, pay 2
    assert carry_bp(bond, flows, r["dirty"], r["ytm"], r["settle_date"], fwd, repo=6.0) < 0  # earn 4.3, pay 6


def model_ytm(bond, flows, params, settle):
    times = np.array([(d - settle).days / 365.0 for d, _ in flows if d > settle])
    amounts = np.array([a for d, a in flows if d > settle])
    dirty = (amounts * discount_factor(times, params)).sum()
    return ytm_at(bond, dirty - accrued_at(bond, settle), settle)


def test_no_rolldown_on_flat_curve():
    flat = [4.0, 0.0, 0.0, 0.0, 1.0, 10.0]
    bond, flows, r = bond_today(4.0, date(2036, 2, 1), 98.0)
    settle = r["settle_date"]
    rd = rolldown_bp(bond, flows, flat, model_ytm(bond, flows, flat, settle), forward_date(settle))
    assert rd == pytest.approx(0.0, abs=0.05)


def test_positive_rolldown_on_upward_curve():
    upward = [5.0, -3.0, 0.0, 0.0, 2.0, 10.0]
    bond, flows, r = bond_today(4.0, date(2029, 2, 1), 98.0)
    settle = r["settle_date"]
    assert rolldown_bp(bond, flows, upward, model_ytm(bond, flows, upward, settle), forward_date(settle)) > 0


def test_dv01_neutral_sizing():
    _, _, a = bond_today(2.65, date(2027, 12, 1), 99.4)
    _, _, b = bond_today(2.0, date(2028, 2, 1), 98.5)
    buy = 10.0
    sell = buy * a["dv01"] / b["dv01"]
    assert buy * a["dv01"] == pytest.approx(sell * b["dv01"])  # a parallel move of 1 bp nets to zero


def test_neighbours():
    assert neighbours(5.0, 6.2)          # 1.2 years apart
    assert not neighbours(5.0, 7.0)      # 2 years apart on a 5-year bond
    assert neighbours(25.0, 29.0)        # 4 years apart but within 20% at the long end
