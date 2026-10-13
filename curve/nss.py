"""Nelson-Siegel-Svensson (NSS) curve: the formula and the daily fit.

The NSS model describes the whole zero-coupon curve with 6 numbers:

    z(t) = b0
         + b1 * (1 - e^(-t/tau1)) / (t/tau1)
         + b2 * [(1 - e^(-t/tau1)) / (t/tau1) - e^(-t/tau1)]
         + b3 * [(1 - e^(-t/tau2)) / (t/tau2) - e^(-t/tau2)]

- b0   long-term level (where the curve goes for very long maturities)
- b1   slope: short end minus long end
- b2   first hump, located around tau1
- b3   second hump, located around tau2 (the Svensson extension)

z(t) is a continuously compounded zero rate in percent: the discount factor
for a cash flow in t years is exp(-z(t)/100 * t). This is the same model and
convention the Bundesbank uses for its Bund curve (checked against its
published parameters), so the two curves can be compared directly.
"""

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ["b0", "b1", "b2", "b3", "tau1", "tau2"]

# Bounds keep the fit stable: tau1 handles the short/medium hump, tau2 the long end.
LOWER = [0.0, -15.0, -30.0, -30.0, 0.1, 5.0]
UPPER = [15.0, 15.0, 30.0, 30.0, 5.0, 30.0]


def zero_rate(t, params) -> np.ndarray:
    """Zero rate in percent (continuous compounding) for maturities t in years."""
    b0, b1, b2, b3, tau1, tau2 = params
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)  # avoid dividing by zero at t = 0
    x1, x2 = t / tau1, t / tau2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1 * f1 + b2 * f2 + b3 * f3


def discount_factor(t, params) -> np.ndarray:
    return np.exp(-zero_rate(t, params) / 100 * np.asarray(t, dtype=float))


def par_yield(maturity: float, params) -> float:
    """Coupon (percent, semiannual) that makes a bond of this maturity price at 100.

    This is the curve to draw next to market yields: a bond priced at par has
    yield = coupon, so the par curve and the bond yields speak the same language.

    Coupon times step back from maturity every half year. If the maturity is not
    a whole number of half years, we are part-way through the first period:
    a fraction (1 - f) of that coupon has already accrued and is paid on top of
    the clean price of 100. Solving  c/2 * (sum DF - (1 - f)) + 100 * DF(T) = 100:
        c = 200 * (1 - DF(T)) / (sum DF - (1 - f))
    """
    times = np.arange(maturity, 1e-9, -0.5)[::-1]
    f = times[0] / 0.5  # share of the current coupon period still to run
    df = discount_factor(times, params)
    return 200 * (1 - df[-1]) / (df.sum() - (1 - f))


def model_dirty_prices(params, times, amounts, bond_index, n_bonds) -> np.ndarray:
    """Price every bond by discounting its cash flows on the NSS curve.

    All the cash flows of all bonds sit in three flat arrays (time, amount,
    which bond); np.bincount adds them up bond by bond. Much faster than a loop.
    """
    pv = amounts * discount_factor(times, params)
    return np.bincount(bond_index, weights=pv, minlength=n_bonds)


def fit(times, amounts, bond_index, market_dirty, price_per_bp, x0) -> tuple[np.ndarray, float]:
    """Find the 6 NSS parameters that best fit one day of BTP prices.

    The error for each bond is (model price - market price) / (price change for
    1bp of yield). Dividing by the bond's price sensitivity turns price errors
    into yield errors in basis points, so a 30-year bond does not count more
    than a 2-year one just because its price moves more.

    loss='soft_l1' is a robust fit: bonds far from the curve (special, illiquid)
    count less, so they cannot drag the whole curve towards them.
    Returns the parameters and the cost of the fit.
    """
    n_bonds = len(market_dirty)

    def errors_bp(params):
        model = model_dirty_prices(params, times, amounts, bond_index, n_bonds)
        return (model - market_dirty) / price_per_bp

    x0 = np.clip(x0, LOWER, UPPER)
    result = least_squares(errors_bp, x0, bounds=(LOWER, UPPER), loss="soft_l1", f_scale=5.0)
    return result.x, result.cost


def starting_points(previous=None) -> list:
    """Where the optimiser starts. NSS has several local minima, so we try a few.

    First day: a grid of 6 starting points.
    After that: yesterday's curve (curves move little from one day to the next,
    so it is usually the best start and it keeps the parameters stable over
    time) plus one generic point as a safety net.
    """
    if previous is None:
        return [[4.0, -1.0, 0.0, 0.0, tau1, tau2] for tau1 in (0.5, 1.5, 3.0) for tau2 in (8.0, 15.0)]
    return [list(previous), [4.0, -1.0, 0.0, 0.0, 1.5, 10.0]]
