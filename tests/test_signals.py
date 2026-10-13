"""Automatic checks on the signals. Run with:  pytest"""

import numpy as np
import pandas as pd
import pytest

from signals.zscore import add_zscores, half_life


def ar1_series(b: float, n: int = 5000, mean: float = -5.0, seed: int = 0) -> pd.Series:
    """Simulate change_t = b * (x_(t-1) - mean) + noise: a residual pulled back to 'mean'."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = x[t - 1] + b * (x[t - 1] - mean) + rng.normal()
    return pd.Series(x)


def test_half_life_recovers_known_value():
    b = -0.1  # each day the gap shrinks by 10% -> half-life = ln 2 / -ln 0.9 = 6.58 days
    hl, b_hat = half_life(ar1_series(b))
    assert b_hat == pytest.approx(b, abs=0.02)
    assert hl == pytest.approx(np.log(2) / -np.log(0.9), rel=0.2)


def test_random_walk_has_long_half_life():
    rng = np.random.default_rng(1)
    walk = pd.Series(np.cumsum(rng.normal(size=5000)))  # no pull back at all
    hl, _ = half_life(walk)
    assert hl > 200


def test_too_short_series_gives_nan():
    hl, _ = half_life(pd.Series(np.arange(10.0)))
    assert np.isnan(hl)


def test_zscore_flags_a_sudden_cheapening():
    rng = np.random.default_rng(2)
    residuals = list(rng.normal(-8, 1, 99)) + [-1.0]  # always about -8 bp, then suddenly -1 bp
    fit = pd.DataFrame({"isin": "X", "date": pd.date_range("2026-01-01", periods=100).strftime("%Y-%m-%d"),
                        "residual_bp": residuals})
    z = add_zscores(fit)["zscore"]
    assert z.iloc[:39].isna().all()   # not enough history yet
    assert z.iloc[-1] > 3             # the jump is flagged as cheap vs its own history


def test_zscore_is_computed_bond_by_bond():
    # Two bonds with very different levels: each z-score uses only its own history
    dates = pd.date_range("2026-01-01", periods=60).strftime("%Y-%m-%d")
    rng = np.random.default_rng(3)
    fit = pd.concat([
        pd.DataFrame({"isin": "A", "date": dates, "residual_bp": rng.normal(-10, 1, 60)}),
        pd.DataFrame({"isin": "B", "date": dates, "residual_bp": rng.normal(+10, 1, 60)}),
    ])
    z = add_zscores(fit).dropna()
    assert z["zscore"].abs().max() < 4  # no bond looks extreme just because of its level
