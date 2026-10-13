"""Automatic checks on the NSS curve model. Run with:  pytest"""

from datetime import date

import numpy as np
import pytest

from curve.nss import discount_factor, fit, model_dirty_prices, par_yield, starting_points, zero_rate
from pricing.bond_math import make_bond, settlement_date, to_py

# Bundesbank Bund curve of 30 Sep 2026: its published NSS parameters and zero rates
BUNDESBANK_PARAMS = [4.18173, -1.90569, 18.84971, -19.00801, 1.36147, 1.50953]
BUNDESBANK_RATES = {1: 3.04, 2: 3.24, 5: 3.36, 10: 3.64, 20: 3.90, 30: 3.99}

TRUE_CURVE = [4.6, -2.2, 2.0, 1.0, 1.5, 12.0]  # a realistic BTP-like curve for the synthetic tests


def synthetic_market(trade_date=date(2026, 10, 9)):
    """30 made-up BTPs from 2 to 50 years, priced exactly on TRUE_CURVE."""
    settle = settlement_date(trade_date)
    times, amounts, index = [], [], []
    for i, years in enumerate(np.linspace(2, 50, 30)):
        maturity = date(2026 + int(years), 3 if years % 1 < 0.5 else 9, 1)
        bond = make_bond(4.0, date(2020, 3, 1), maturity)
        for cf in bond.cashflows():
            if to_py(cf.date()) > settle:
                times.append((to_py(cf.date()) - settle).days / 365.0)
                amounts.append(cf.amount())
                index.append(i)
    times, amounts, index = np.array(times), np.array(amounts), np.array(index)
    prices = model_dirty_prices(TRUE_CURVE, times, amounts, index, 30)
    return times, amounts, index, prices


def test_formula_reproduces_bundesbank_rates():
    # Same formula, same parameters -> same rates as published (rounded to 2 decimals)
    for years, published in BUNDESBANK_RATES.items():
        assert float(zero_rate(years, BUNDESBANK_PARAMS)) == pytest.approx(published, abs=0.006)


def test_discount_factor_today_is_one():
    assert float(discount_factor(0.0, TRUE_CURVE)) == pytest.approx(1.0)


def test_par_yield_on_flat_curve():
    # Flat 4% continuous curve -> par yield = semiannual equivalent of 4% continuous
    flat = [4.0, 0.0, 0.0, 0.0, 1.0, 10.0]
    assert par_yield(10, flat) == pytest.approx(2 * (np.exp(0.04 / 2) - 1) * 100, abs=1e-10)


def test_fit_recovers_known_curve():
    times, amounts, index, prices = synthetic_market()
    price_per_bp = np.full(len(prices), 0.05)
    fits = [fit(times, amounts, index, prices, price_per_bp, x0) for x0 in starting_points()]
    params = min(fits, key=lambda f: f[1])[0]
    for years in (2, 5, 10, 30):
        assert float(zero_rate(years, params)) == pytest.approx(float(zero_rate(years, TRUE_CURVE)), abs=0.005)


def test_cheap_bond_gets_positive_residual_and_does_not_move_the_curve():
    times, amounts, index, prices = synthetic_market()
    prices = prices.copy()
    prices[10] -= 0.5  # bond 10 now trades lower = higher yield = cheap
    price_per_bp = np.full(len(prices), 0.05)
    fits = [fit(times, amounts, index, prices, price_per_bp, x0) for x0 in starting_points()]
    params = min(fits, key=lambda f: f[1])[0]
    model = model_dirty_prices(params, times, amounts, index, len(prices))
    residual_bp = (model - prices) / price_per_bp
    assert residual_bp[10] > 5                          # clearly flagged as cheap
    assert np.abs(np.delete(residual_bp, 10)).max() < 1  # the other bonds stay on the curve
