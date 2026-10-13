"""Backtest engine for DV01-neutral BTP switches.

The question: if a trader had followed the switch signals every day for the
last two years, would they have made money after costs?

Rules to stay honest (no cheating with the future):
1. Every signal on day t uses only data up to day t (rolling windows, and a
   half-life re-estimated every month on past data only).
2. A decision taken at the close of day t is EXECUTED at the price of day t+1.
   Euronext prices are daily VWAPs: you cannot trade at a price that you only
   learn at the end of the day.
3. Every trade pays the bid-ask cost on both legs, when entering and when exiting.

How a switch earns money (long A, short B, same DV01):
    daily P&L (bp) = - change in (yield_A - yield_B) + daily carry of A - daily carry of B
If A's yield falls relative to B's, the gap closes and the switch gains.
In euro: P&L (bp) x DV01 of the position.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.zscore import MIN_OBS, SMOOTH, WINDOW, weekly_half_life

TRADING_DAYS_PER_MONTH = 21


@dataclass
class Params:
    z_entry: float = 2.0       # open when the smoothed pair z-score reaches this
    z_exit: float = 0.5        # close when it comes back below this (the gap has mostly closed)
    z_stop: float = 4.0        # stop-loss: close if the gap keeps widening to this
    max_hold: int = 40         # close anyway after this many trading days
    min_half_life: float = 2   # trading days
    max_half_life: float = 30
    cost_bp: float = 0.25      # half bid-ask per leg, per trade, in bp of yield
    max_positions: int = 10
    max_per_bond: int = 2      # avoid repeating the same idea many times
    buy_nominal: float = 10e6  # euro nominal of the long leg
    hl_refit: int = 21         # re-estimate half-lives every month...
    hl_lookback: int = 250     # ...on the last year of data only


def pair_signals(residuals: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict:
    """Smoothed pair residual, its 60-day gap from the mean and its z-score, for every pair.

    Columns are 'A|B'. Positive z: A is cheap relative to B (buy A, sell B).
    """
    keys = [f"{a}|{b}" for a, b in pairs]
    spread = pd.DataFrame({k: residuals[a] - residuals[b] for k, (a, b) in zip(keys, pairs)})
    smooth = spread.rolling(SMOOTH).mean()
    roll = smooth.rolling(WINDOW, min_periods=MIN_OBS)
    gap = smooth - roll.mean()
    return {"spread": spread, "gap": gap, "z": gap / roll.std()}


def rolling_half_life(spread: pd.DataFrame, refit: int, lookback: int) -> pd.DataFrame:
    """Half-life of each pair, re-estimated every `refit` days using only the previous `lookback` days."""
    out = pd.DataFrame(np.nan, index=spread.index, columns=spread.columns)
    for i in range(0, len(spread), refit):
        past = spread.iloc[max(0, i - lookback + 1): i + 1]
        out.iloc[i] = [weekly_half_life(past[c]) for c in spread.columns]
    return out.ffill()


def run(data: dict, pairs: list[tuple[str, str]], p: Params, flip: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate the strategy day by day.

    data: DataFrames indexed by date, one column per ISIN:
          'residual' (bp), 'ytm' (%), 'dv01' (euro per 1mn), 'carry' (bp per month).
    flip: trade AGAINST every signal. A sanity check: if the signal is real,
          the flipped strategy must lose money before costs.
    Returns the list of trades and the daily P&L in euro.
    """
    sig = pair_signals(data["residual"], pairs)
    hl = rolling_half_life(sig["spread"], p.hl_refit, p.hl_lookback)
    z, gap = sig["z"], sig["gap"]
    ytm, dv01, carry = data["ytm"], data["dv01"], data["carry"]
    dates = z.index
    pair_of = dict(zip(z.columns, pairs))

    open_pos = {}   # key -> position
    pending = []    # (key, direction) decided yesterday, executed today
    to_close = []   # keys decided yesterday, closed today
    trades, daily = [], pd.Series(0.0, index=dates)

    for i in range(1, len(dates)):
        t = dates[i]
        # 1. P&L of positions held from yesterday's close to today's close
        for key, pos in open_pos.items():
            a, b = pair_of[key]
            change = (ytm.at[t, a] - ytm.at[t, b]) - (ytm.at[dates[i - 1], a] - ytm.at[dates[i - 1], b])
            bp = pos["dir"] * (-change * 100 + (carry.at[t, a] - carry.at[t, b]) / TRADING_DAYS_PER_MONTH)
            if np.isfinite(bp):
                pos["pnl_bp"] += bp
                daily.iat[i] += bp * pos["dv01_eur"]

        # 2. Execute yesterday's decisions at today's price, paying the costs
        for key in to_close:
            pos = open_pos.pop(key)
            pos["pnl_bp"] -= 2 * p.cost_bp            # two legs
            daily.iat[i] -= 2 * p.cost_bp * pos["dv01_eur"]
            trades.append({**pos, "exit": t, "days": i - pos["start_i"]})
        for key, direction in pending:
            a, b = pair_of[key]
            buy = a if direction > 0 else b
            dv01_eur = p.buy_nominal / 1e6 * dv01.at[t, buy]
            if not np.isfinite(dv01_eur):
                continue
            open_pos[key] = {"pair": key, "buy": buy, "sell": b if direction > 0 else a, "dir": direction,
                             "entry": t, "start_i": i, "dv01_eur": dv01_eur, "pnl_bp": -2 * p.cost_bp,
                             "entry_z": abs(z.iat[i - 1, z.columns.get_loc(key)]), "reason": ""}
            daily.iat[i] -= 2 * p.cost_bp * dv01_eur
        to_close, pending = [], []

        # 3. Decide at today's close what to do tomorrow
        for key, pos in open_pos.items():
            zz = z.at[t, key] * pos["dir"] * (-1 if flip else 1)  # how cheap the long leg still looks
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

        used = pd.Series([b for k in open_pos for b in pair_of[k]]).value_counts()
        n_open = len(open_pos) - len(to_close)
        candidates = z.loc[t].dropna()
        candidates = candidates[candidates.abs() >= p.z_entry].abs().sort_values(ascending=False)
        for key in candidates.index:
            if n_open >= p.max_positions:
                break
            if key in open_pos:
                continue
            a, b = pair_of[key]
            if used.get(a, 0) >= p.max_per_bond or used.get(b, 0) >= p.max_per_bond:
                continue
            if not p.min_half_life <= hl.at[t, key] <= p.max_half_life:
                continue
            direction = 1 if z.at[t, key] > 0 else -1
            # Expected gain: the gap closing plus one month of carry must beat the round-trip cost
            expected = abs(gap.at[t, key]) + direction * (carry.at[t, a] - carry.at[t, b])
            if not expected > 4 * p.cost_bp:
                continue
            pending.append((key, -direction if flip else direction))
            used[a] = used.get(a, 0) + 1
            used[b] = used.get(b, 0) + 1
            n_open += 1

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
    return {
        "trades": len(trades),
        "hit_rate_%": 100 * (trades["pnl_bp"] > 0).mean() if len(trades) else np.nan,
        "avg_pnl_bp": trades["pnl_bp"].mean() if len(trades) else np.nan,
        "avg_days": trades["days"].mean() if len(trades) else np.nan,
        "total_pnl_eur": daily.sum(),
        "pnl_per_year_eur": daily.sum() / years if years else np.nan,
        "sharpe": active.mean() / std * np.sqrt(252) if std > 0 else np.nan,
        "max_drawdown_eur": (cum - cum.cummax()).min() if len(cum) else np.nan,
    }
