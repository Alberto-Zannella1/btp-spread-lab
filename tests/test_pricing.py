"""Automatic checks on the pricing engine.

Run from the project folder with:  pytest
Each test_ function is one check. pytest runs them all and reports which fail.
"""

from datetime import date

import pytest

from pricing.bond_math import (
    accrued_manual, accrued_ql, clean_price_ql, make_bond, risk_measures,
    settlement_date, to_annual, ytm_manual, ytm_ql,
)

# A few real BTPs: (coupon, issue date, maturity)
BONDS = [
    (3.85, date(2024, 8, 1), date(2035, 2, 1)),    # long bond, regular coupons
    (6.50, date(1997, 11, 1), date(2027, 11, 1)),  # old high-coupon bond
    (3.95, date(2026, 10, 1), date(2032, 2, 1)),   # new issue: short first coupon
    (2.20, date(2026, 1, 29), date(2028, 2, 28)),  # odd dates
]
TRADE_DATES = [date(2026, 10, 9), date(2026, 10, 14), date(2026, 12, 30), date(2027, 1, 28)]
PRICES = [92.0, 99.5, 104.0]


def test_settlement_is_t_plus_2_business_days():
    assert settlement_date(date(2026, 10, 14)) == date(2026, 10, 16)  # Wednesday -> Friday
    assert settlement_date(date(2026, 10, 15)) == date(2026, 10, 19)  # Thursday -> Monday
    assert settlement_date(date(2026, 12, 23)) == date(2026, 12, 28)  # skips 25 and 26 Dec (TARGET holidays)


@pytest.mark.parametrize("coupon, issue, maturity", BONDS)
@pytest.mark.parametrize("trade_date", TRADE_DATES)
def test_accrued_manual_matches_quantlib(coupon, issue, maturity, trade_date):
    bond = make_bond(coupon, issue, maturity)
    assert accrued_manual(coupon, issue, maturity, trade_date) == pytest.approx(accrued_ql(bond, trade_date), abs=1e-10)


@pytest.mark.parametrize("coupon, issue, maturity", BONDS)
@pytest.mark.parametrize("trade_date", TRADE_DATES)
@pytest.mark.parametrize("price", PRICES)
def test_ytm_manual_matches_quantlib(coupon, issue, maturity, trade_date, price):
    bond = make_bond(coupon, issue, maturity)
    manual = ytm_manual(coupon, issue, maturity, price, trade_date)
    assert manual == pytest.approx(ytm_ql(bond, price, trade_date), abs=1e-8)  # 1e-8 % = 0.000001 bp


def test_accrued_is_zero_on_coupon_date():
    # Trade on Thu 28 Jan 2027 settles on Mon 1 Feb 2027, a coupon date of the 2035 bond
    bond = make_bond(3.85, date(2024, 8, 1), date(2035, 2, 1))
    assert accrued_ql(bond, date(2027, 1, 28)) == pytest.approx(0.0, abs=1e-12)


def test_par_bond_yields_its_coupon():
    # Settling on a coupon date at price 100, the yield must equal the coupon
    bond = make_bond(4.0, date(2024, 2, 1), date(2034, 2, 1))
    assert ytm_ql(bond, 100.0, date(2027, 1, 28)) == pytest.approx(4.0, abs=1e-8)


@pytest.mark.parametrize("coupon, issue, maturity", BONDS)
def test_price_yield_round_trip(coupon, issue, maturity):
    bond = make_bond(coupon, issue, maturity)
    trade_date = date(2026, 10, 14)
    y = ytm_ql(bond, 97.25, trade_date)
    assert clean_price_ql(bond, y, trade_date) == pytest.approx(97.25, abs=1e-8)


def test_dv01_matches_a_1bp_bump():
    # DV01 from duration must match repricing the bond 1bp higher and lower
    bond = make_bond(3.85, date(2024, 8, 1), date(2035, 2, 1))
    trade_date = date(2026, 10, 14)
    r = risk_measures(bond, 96.43, trade_date)
    up = clean_price_ql(bond, r["ytm"] + 0.01, trade_date)
    down = clean_price_ql(bond, r["ytm"] - 0.01, trade_date)
    dv01_bump = (down - up) / 2 / 100 * 1e6  # per 1mn nominal
    assert r["dv01"] == pytest.approx(dv01_bump, rel=1e-4)


def test_dv01_positive_and_growing_with_maturity():
    trade_date = date(2026, 10, 14)
    dv01s = [risk_measures(make_bond(3.5, date(2026, 2, 1), date(2026 + n, 2, 1)), 100.0, trade_date)["dv01"]
             for n in (3, 5, 10, 30)]
    assert all(d > 0 for d in dv01s)
    assert dv01s == sorted(dv01s)


def test_known_values_2035_bond():
    # Regression test: values computed once and checked by hand. If these change, something broke.
    bond = make_bond(3.85, date(2024, 8, 1), date(2035, 2, 1))
    r = risk_measures(bond, 96.43, date(2026, 10, 14))
    assert r["settle_date"] == date(2026, 10, 16)
    assert r["accrued"] == pytest.approx(3.85 / 2 * 76 / 184, abs=1e-12)  # 1 Aug -> 16 Oct = 76 days of 184
    assert r["ytm"] == pytest.approx(4.3670, abs=1e-4)
    assert r["mod_duration"] == pytest.approx(6.94, abs=0.01)


def test_annual_conversion():
    assert to_annual(4.0) == pytest.approx(4.04, abs=1e-12)  # (1.02)^2 - 1 = 4.04%
