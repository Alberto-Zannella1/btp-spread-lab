"""Automatic checks on the backtest engine. Run with:  pytest"""

import numpy as np
import pandas as pd

from backtest.engine import Params, run

N = 400
DATES = pd.date_range("2025-01-01", periods=N, freq="B").strftime("%Y-%m-%d")


def market(spread_bp: np.ndarray) -> dict:
    """Two bonds whose residual spread (A - B) is the given series; yields move with it, no carry."""
    resid = pd.DataFrame({"A": spread_bp, "B": 0.0}, index=DATES)
    ytm = pd.DataFrame({"A": 4.0 + spread_bp / 100, "B": 4.0}, index=DATES)
    dv01 = pd.DataFrame({"A": 1000.0, "B": 1000.0}, index=DATES)
    carry = pd.DataFrame({"A": 0.0, "B": 0.0}, index=DATES)
    return {"residual": resid, "ytm": ytm, "dv01": dv01, "carry": carry}


def mean_reverting(seed=0, half_life=10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = -(1 - 0.5 ** (1 / half_life))
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = x[t - 1] * (1 + b) + rng.normal(0, 1.0)
    return x


def test_mean_reverting_spread_makes_money_before_costs():
    trades, daily = run(market(mean_reverting()), [("A", "B")], Params(cost_bp=0.0))
    assert len(trades) > 3
    assert daily.sum() > 0


def test_flipped_signal_loses_the_same():
    data = market(mean_reverting(seed=1))
    _, normal = run(data, [("A", "B")], Params(cost_bp=0.0))
    _, flipped = run(data, [("A", "B")], Params(cost_bp=0.0), flip=True)
    assert flipped.sum() == -normal.sum()  # same trades, opposite side


def test_costs_are_charged_on_both_legs_entry_and_exit():
    # A tiny cost does not change which trades are taken: each trade must lose exactly
    # 4 x cost (2 legs x entry and exit) compared with the same trade without costs.
    data = market(mean_reverting(seed=2))
    free, _ = run(data, [("A", "B")], Params(cost_bp=0.0))
    paid, _ = run(data, [("A", "B")], Params(cost_bp=0.01))
    assert len(free) == len(paid) > 0
    assert np.allclose(free["pnl_bp"] - paid["pnl_bp"], 4 * 0.01)


def test_no_profit_from_the_signal_day_price():
    # A one-day jump of 10 bp that reverts the next day. A cheating backtest would buy
    # at the jump price and earn about 10 bp; ours trades the day AFTER the signal.
    rng = np.random.default_rng(3)
    spread = rng.normal(0, 0.05, N)
    spread[200] += 10.0
    trades, _ = run(market(spread), [("A", "B")], Params(cost_bp=0.0, min_half_life=0))
    assert trades["pnl_bp"].abs().max() < 0.5  # no trade comes anywhere near 10 bp


def test_same_day_execution_would_capture_the_fake_jump():
    # The diagnostic mode (execution_lag=0) shows why it is forbidden: it earns on the one-day jump.
    rng = np.random.default_rng(3)
    spread = rng.normal(0, 0.05, N)
    spread[200] += 10.0
    trades, _ = run(market(spread), [("A", "B")], Params(cost_bp=0.0, min_half_life=0, execution_lag=0))
    assert trades["pnl_bp"].max() > 1.5


def test_butterfly_is_neutral_to_level_and_slope():
    from backtest.research import build_butterflies
    bonds = pd.DataFrame({"isin": ["S", "A", "L"], "maturity": ["2031-01-01", "2032-01-01", "2033-07-01"]})
    (a, wa), (s, ws), (l, wl) = build_butterflies(bonds, ["S", "A", "L"], "2026-01-01")[0]
    years = {k: (pd.Timestamp(m) - pd.Timestamp("2026-01-01")).days / 365.25 for k, m in zip(bonds["isin"], bonds["maturity"])}
    assert (a, s, l) == ("A", "S", "L")
    assert abs(wa + ws + wl) < 1e-12                                                # parallel move: no P&L
    assert abs(wa * years[a] + ws * years[s] + wl * years[l]) < 1e-12               # slope change: no P&L
