"""Backtest engine for BTP relative value trades: switches and butterflies.

The question: if a trader had followed the signals every day for the last
two years, would they have made money after costs?

A TRADE STRUCTURE is a list of legs (ISIN, weight). Weights are in DV01 units:
- switch:    buy A, sell B             -> [(A, +1), (B, -1)]
- butterfly: buy A, sell the shorter neighbour S and the longer neighbour L
             -> [(A, +1), (S, -wS), (L, -wL)], with wS + wL = 1 and
                wS*T_S + wL*T_L = T_A (T = years to maturity).
  wS + wL = 1 makes it neutral to a PARALLEL move of the curve (like a switch);
  the second condition also makes it neutral to a change in the SLOPE of the
  curve around A. A switch between two different maturities is not.

Rules to stay honest (no cheating with the future):
1. Every signal on day t uses only data up to day t (rolling windows, and a
   half-life re-estimated every month on past data only).
2. A decision taken at the close of day t is EXECUTED at the price of day t+1
   (execution_lag = 1). Euronext prices are daily VWAPs: you only know them in
   the evening. execution_lag = 0 is available ONLY as a diagnostic, to measure
   how much the delay costs.
3. Every leg pays the bid-ask cost, when entering and when exiting.

P&L of a structure, in bp, from one day to the next:
    - sum of weight_i x change in yield_i  +  sum of weight_i x daily carry_i
Each yield is curve (model) yield + residual, so the P&L splits into:
    residual part - the distortion closing: what the strategy is betting on;
    curve part    - the curve changing shape between the legs' maturities: unwanted risk;
    carry part.
In euro: P&L (bp) x DV01 of the main leg (10mn nominal).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.zscore import MIN_OBS, SMOOTH, WINDOW, weekly_half_life

TRADING_DAYS_PER_MONTH = 21


@dataclass
class Params:
    z_entry: float = 2.0       # open when the smoothed z-score reaches this
    z_exit: float = 0.5        # close when it comes back below this (the gap has mostly closed)
    z_stop: float = 4.0        # stop-loss: close if the gap keeps widening to this
    max_hold: int = 40         # close anyway after this many trading days
    min_half_life: float = 2   # trading days
    max_half_life: float = 30
    cost_bp: float = 0.25      # half bid-ask per unit of DV01 traded, per trade, in bp of yield
    max_positions: int = 10
    max_per_bond: int = 2      # avoid repeating the same idea many times
    buy_nominal: float = 10e6  # euro nominal of the main leg
    hl_refit: int = 21         # re-estimate half-lives every month...
    hl_lookback: int = 250     # ...on the last year of data only
    execution_lag: int = 1     # 1 = trade the day after the signal (honest); 0 = diagnostic only


def as_structures(items) -> dict:
    """Accept switches as (A, B) pairs or ready-made structures; return {key: legs}."""
    out = {}
    for it in items:
        legs = ((it[0], 1.0), (it[1], -1.0)) if isinstance(it[0], str) else tuple(it)
        out["|".join(i for i, _ in legs)] = legs
    return out


def structure_signals(residuals: pd.DataFrame, structures: dict) -> dict:
    """Smoothed residual of each structure, its 60-day gap from the mean and its z-score.

    Positive z: the main leg is cheap relative to the others (buy the structure).
    """
    spread = pd.DataFrame({k: sum(w * residuals[i] for i, w in legs) for k, legs in structures.items()})
    smooth = spread.rolling(SMOOTH).mean()
    roll = smooth.rolling(WINDOW, min_periods=MIN_OBS)
    gap = smooth - roll.mean()
    return {"spread": spread, "gap": gap, "z": gap / roll.std()}


def rolling_half_life(spread: pd.DataFrame, refit: int, lookback: int) -> pd.DataFrame:
    """Half-life of each structure, re-estimated every `refit` days using only the previous `lookback` days."""
    out = pd.DataFrame(np.nan, index=spread.index, columns=spread.columns)
    for i in range(0, len(spread), refit):
        past = spread.iloc[max(0, i - lookback + 1): i + 1]
        out.iloc[i] = [weekly_half_life(past[c]) for c in spread.columns]
    return out.ffill()


def run(data: dict, items, p: Params, flip: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate the strategy day by day.

    data:  DataFrames indexed by date, one column per ISIN:
           'residual' (bp), 'ytm' (%), 'dv01' (euro per 1mn), 'carry' (bp per month).
    items: switches as (A, B) pairs, or structures as ((isin, weight), ...).
    flip:  trade AGAINST every signal. A sanity check: if the signal is real,
           the flipped strategy must lose money before costs.
    Returns the list of trades and the daily P&L in euro.
    """
    structures = as_structures(items)
    sig = structure_signals(data["residual"], structures)
    hl = rolling_half_life(sig["spread"], p.hl_refit, p.hl_lookback)
    z, gap = sig["z"], sig["gap"]
    resid, ytm, dv01, carry = data["residual"], data["ytm"], data["dv01"], data["carry"]
    model = ytm - resid / 100  # the curve's yield for each bond
    dates = z.index

    open_pos, pending, to_close = {}, [], []
    trades, daily = [], pd.Series(0.0, index=dates)

    def leg_cost(legs):
        return p.cost_bp * sum(abs(w) for _, w in legs)  # bp, per side (entry or exit)

    def execute(i):
        nonlocal pending, to_close
        t = dates[i]
        for key in to_close:
            pos = open_pos.pop(key)
            c = leg_cost(structures[key])
            pos["pnl_bp"] -= c
            pos["cost_bp"] += c
            daily.iat[i] -= c * pos["dv01_eur"]
            trades.append({**pos, "exit": t, "days": i - pos["start_i"]})
        for key, direction in pending:
            legs = structures[key]
            dv01_eur = p.buy_nominal / 1e6 * dv01.at[t, legs[0][0]]
            if not np.isfinite(dv01_eur):
                continue
            c = leg_cost(legs)
            open_pos[key] = {"structure": key, "main": legs[0][0], "legs": len(legs), "dir": direction,
                             "entry": t, "start_i": i, "dv01_eur": dv01_eur, "entry_z": abs(z.at[dates[i - p.execution_lag], key]),
                             "pnl_bp": -c, "resid_bp": 0.0, "curve_bp": 0.0, "carry_bp": 0.0, "cost_bp": c, "reason": ""}
            daily.iat[i] -= c * dv01_eur
        pending, to_close = [], []

    for i in range(1, len(dates)):
        t, y = dates[i], dates[i - 1]
        # 1. P&L of positions held from yesterday's close to today's close
        for key, pos in open_pos.items():
            legs = structures[key]
            d_resid = sum(w * (resid.at[t, s] - resid.at[y, s]) for s, w in legs)
            d_curve = sum(w * (model.at[t, s] - model.at[y, s]) for s, w in legs) * 100
            c = sum(w * carry.at[t, s] for s, w in legs) / TRADING_DAYS_PER_MONTH
            parts = pos["dir"] * np.array([-d_resid, -d_curve, c])
            if np.all(np.isfinite(parts)):
                pos["resid_bp"] += parts[0]
                pos["curve_bp"] += parts[1]
                pos["carry_bp"] += parts[2]
                pos["pnl_bp"] += parts.sum()
                daily.iat[i] += parts.sum() * pos["dv01_eur"]

        # 2. Execute yesterday's decisions at today's price (the honest default)
        if p.execution_lag == 1:
            execute(i)

        # 3. Decide at today's close
        for key, pos in open_pos.items():
            zz = z.at[t, key] * pos["dir"] * (-1 if flip else 1)  # how cheap the structure still looks
            if not np.isfinite(zz):
                pos["reason"] = "no data"
            elif zz <= p.z_exit:
                pos["reason"] = "target"
            elif zz >= p.z_stop:
                pos["reason"] = "stop"
            elif i - pos["start_i"] >= p.max_hold:
                pos["reason"] = "time"
            else:
                continue
            to_close.append(key)

        used = {}
        for key in open_pos:
            for s, _ in structures[key]:
                used[s] = used.get(s, 0) + 1
        n_open = len(open_pos) - len(to_close)
        candidates = z.loc[t].dropna()
        candidates = candidates[candidates.abs() >= p.z_entry].abs().sort_values(ascending=False)
        for key in candidates.index:
            if n_open >= p.max_positions:
                break
            if key in open_pos:
                continue
            legs = structures[key]
            if any(used.get(s, 0) >= p.max_per_bond for s, _ in legs):
                continue
            if not p.min_half_life <= hl.at[t, key] <= p.max_half_life:
                continue
            direction = 1 if z.at[t, key] > 0 else -1
            # Expected gain: the gap closing plus one month of carry must beat the round-trip cost
            expected = abs(gap.at[t, key]) + direction * sum(w * carry.at[t, s] for s, w in legs)
            if not expected > 2 * leg_cost(legs):
                continue
            pending.append((key, -direction if flip else direction))
            for s, _ in legs:
                used[s] = used.get(s, 0) + 1
            n_open += 1

        # Diagnostic only: trade at the same price that generated the signal
        if p.execution_lag == 0:
            execute(i)

    for pos in open_pos.values():  # still open at the end: mark to market, no exit cost
        trades.append({**pos, "exit": dates[-1], "days": len(dates) - 1 - pos["start_i"], "reason": "end"})
    trades = pd.DataFrame(trades)
    if not trades.empty:
        trades["pnl_eur"] = trades["pnl_bp"] * trades["dv01_eur"]
        trades = trades.drop(columns=["start_i"])
    return trades, daily


def metrics(trades: pd.DataFrame, daily: pd.Series) -> dict:
    """Summary numbers a trader would look at."""
    active = daily.loc[daily.ne(0).idxmax():] if daily.ne(0).any() else daily
    cum = active.cumsum()
    years = len(active) / 252
    std = active.std()
    n = len(trades)
    if n == 0:
        trades = pd.DataFrame(columns=["pnl_bp", "days"])
    return {
        "trades": n,
        "hit_rate_%": 100 * (trades["pnl_bp"] > 0).mean() if n else np.nan,
        "avg_pnl_bp": trades["pnl_bp"].mean() if n else np.nan,
        "avg_days": trades["days"].mean() if n else np.nan,
        "total_pnl_eur": daily.sum(),
        "pnl_per_year_eur": daily.sum() / years if years else np.nan,
        "sharpe": active.mean() / std * np.sqrt(252) if std > 0 else np.nan,
        "max_drawdown_eur": (cum - cum.cummax()).min() if len(cum) else np.nan,
    }
