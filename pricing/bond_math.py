"""Bond math for fixed-rate BTPs.

Two implementations of the same things:
- QuantLib (used in production: fast, tested, handles every edge case);
- by hand (used only to understand the formulas and to check QuantLib in the tests).

BTP conventions:
- semiannual coupons, coupon dates stepped back from maturity every 6 months;
- day count ACT/ACT ICMA: accrued = coupon/2 * days since last coupon / days in the period;
- settlement T+2 on the TARGET calendar;
- yield to maturity with semiannual compounding (the market standard for BTPs).
  Borsa Italiana shows an annual-compounded "rendimento effettivo": see to_annual().
"""

from datetime import date

import QuantLib as ql
from dateutil.relativedelta import relativedelta

SETTLEMENT_DAYS = 2
CALENDAR = ql.TARGET()
FACE = 100.0


# ---------------------------------------------------------------- date helpers

def to_ql(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def to_py(d: ql.Date) -> date:
    return date(d.year(), d.month(), d.dayOfMonth())


def settlement_date(trade_date: date) -> date:
    """Trade date + 2 TARGET business days."""
    return to_py(CALENDAR.advance(to_ql(trade_date), SETTLEMENT_DAYS, ql.Days))


# ---------------------------------------------------------------- QuantLib

def make_bond(coupon: float, issue_date: date, maturity: date) -> ql.FixedRateBond:
    """Build a BTP in QuantLib. coupon is the annual rate in percent (e.g. 3.85).

    Coupon dates are generated backward from maturity, so a new issue gets a
    short first coupon, exactly like the real bond. Payment dates are kept
    unadjusted: this is the street convention for quoting yields.
    """
    schedule = ql.Schedule(
        to_ql(issue_date), to_ql(maturity), ql.Period(ql.Semiannual), CALENDAR,
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False,
    )
    day_count = ql.ActualActual(ql.ActualActual.ISMA, schedule)
    return ql.FixedRateBond(
        SETTLEMENT_DAYS, FACE, schedule, [coupon / 100], day_count,
        ql.Unadjusted, FACE, to_ql(issue_date),
    )


def _day_count(bond: ql.FixedRateBond) -> ql.DayCounter:
    return ql.as_fixed_rate_coupon(bond.cashflows()[0]).dayCounter()


def _set_date(trade_date: date) -> ql.Date:
    """QuantLib works 'as of' a global evaluation date: set it to the trade date."""
    ql.Settings.instance().evaluationDate = to_ql(trade_date)
    return to_ql(settlement_date(trade_date))


def accrued_ql(bond: ql.FixedRateBond, trade_date: date) -> float:
    settle = _set_date(trade_date)
    return bond.accruedAmount(settle)


def ytm_ql(bond: ql.FixedRateBond, clean_price: float, trade_date: date) -> float:
    """Yield to maturity in percent, semiannual compounding."""
    settle = _set_date(trade_date)
    price = ql.BondPrice(clean_price, ql.BondPrice.Clean)
    y = ql.BondFunctions.bondYield(bond, price, _day_count(bond), ql.Compounded, ql.Semiannual, settle)
    return y * 100


def clean_price_ql(bond: ql.FixedRateBond, ytm: float, trade_date: date) -> float:
    """The inverse: clean price from a yield in percent."""
    settle = _set_date(trade_date)
    rate = ql.InterestRate(ytm / 100, _day_count(bond), ql.Compounded, ql.Semiannual)
    return ql.BondFunctions.cleanPrice(bond, rate, settle)


def risk_measures(bond: ql.FixedRateBond, clean_price: float, trade_date: date) -> dict:
    """Everything a trader looks at for one bond on one day.

    dv01 is in euro per 1 million nominal: how much the position gains when
    the yield falls by 1 basis point (0.01%).
    """
    settle = _set_date(trade_date)
    day_count = _day_count(bond)
    accrued = bond.accruedAmount(settle)
    dirty = clean_price + accrued
    ytm = ytm_ql(bond, clean_price, trade_date)
    rate = ql.InterestRate(ytm / 100, day_count, ql.Compounded, ql.Semiannual)
    mod_duration = ql.BondFunctions.duration(bond, rate, ql.Duration.Modified, settle)
    convexity = ql.BondFunctions.convexity(bond, rate, settle)
    dv01 = mod_duration * dirty / FACE * 1e-4 * 1e6
    return {
        "settle_date": to_py(settle),
        "clean": clean_price,
        "accrued": accrued,
        "dirty": dirty,
        "ytm": ytm,
        "mod_duration": mod_duration,
        "convexity": convexity,
        "dv01": dv01,
    }


def accrued_at(bond: ql.FixedRateBond, settle: date) -> float:
    """Accrued interest at a given settlement date (not a trade date)."""
    return bond.accruedAmount(to_ql(settle))


def ytm_at(bond: ql.FixedRateBond, clean_price: float, settle: date) -> float:
    """Yield in percent (semiannual) for a given settlement date, e.g. a forward date."""
    ql.Settings.instance().evaluationDate = to_ql(settle)
    price = ql.BondPrice(clean_price, ql.BondPrice.Clean)
    y = ql.BondFunctions.bondYield(bond, price, _day_count(bond), ql.Compounded, ql.Semiannual, to_ql(settle))
    return y * 100


def to_annual(ytm_semiannual: float) -> float:
    """Semiannual-compounded yield -> annual-compounded yield (both in percent)."""
    return ((1 + ytm_semiannual / 200) ** 2 - 1) * 100


# ---------------------------------------------------------------- by hand

def coupon_dates(issue_date: date, maturity: date) -> list[date]:
    """All coupon dates, from the last one before issue up to maturity.

    Each date is maturity minus k*6 months (always counted from maturity, so a
    maturity on the 31st does not drift to the 30th and then the 28th).
    The first date returned is before or on the issue date: it is the start of
    the 'notional' first period, needed when the first coupon is short.
    """
    dates = []
    k = 0
    while True:
        d = maturity - relativedelta(months=6 * k)
        dates.append(d)
        if d <= issue_date:
            break
        k += 1
    return sorted(dates)


def _current_period(issue_date: date, maturity: date, settle: date) -> tuple[date, date, list[date]]:
    """Coupon dates around settlement: previous, next, and all the ones still to be paid."""
    dates = coupon_dates(issue_date, maturity)
    future = [d for d in dates if d > settle]
    previous = max(d for d in dates if d <= settle)
    return previous, future[0], future


def accrued_manual(coupon: float, issue_date: date, maturity: date, trade_date: date) -> float:
    """Accrued interest per 100 nominal, ACT/ACT ICMA.

    accrued = coupon/2 * (days from start of accrual to settlement) / (days in the coupon period)
    For the first (short) coupon, accrual starts on the issue date, not on the
    notional period start.
    """
    settle = settlement_date(trade_date)
    previous, next_date, _ = _current_period(issue_date, maturity, settle)
    accrual_start = max(previous, issue_date)
    return coupon / 2 * (settle - accrual_start).days / (next_date - previous).days


def remaining_cash_flows(coupon: float, issue_date: date, maturity: date, settle: date) -> tuple[float, list[float]]:
    """Cash flows still to be received, and w = fraction of the current period still to run.

    Every coupon pays coupon/2, except a short first coupon (it pays only
    from the issue date). The last cash flow also includes the 100 of principal.
    """
    previous, next_date, future = _current_period(issue_date, maturity, settle)
    w = (next_date - settle).days / (next_date - previous).days
    cash_flows = []
    for i, pay_date in enumerate(future):
        cash = coupon / 2
        if i == 0 and previous < issue_date:
            cash = coupon / 2 * (next_date - issue_date).days / (next_date - previous).days
        if pay_date == maturity:
            cash += FACE
        cash_flows.append(cash)
    return w, cash_flows


def dirty_from_yield(w: float, cash_flows: list[float], ytm: float) -> float:
    """Present value at ytm (percent, semiannual compounding).

    dirty = sum over remaining coupons i = 0, 1, ...:  CF_i / (1 + y/2) ** (w + i)
    """
    y = ytm / 100
    return sum(cash / (1 + y / 2) ** (w + i) for i, cash in enumerate(cash_flows))


def ytm_manual(coupon: float, issue_date: date, maturity: date, clean_price: float, trade_date: date) -> float:
    """Yield to maturity in percent, found by bisection.

    The price falls as the yield rises, so we keep halving the interval
    [low, high] until the yield gives back the observed dirty price.
    60 halvings shrink the interval below 1e-15: far more precise than needed.
    """
    settle = settlement_date(trade_date)
    target = clean_price + accrued_manual(coupon, issue_date, maturity, trade_date)
    w, cash_flows = remaining_cash_flows(coupon, issue_date, maturity, settle)
    low, high = -5.0, 50.0
    for _ in range(60):
        mid = (low + high) / 2
        if dirty_from_yield(w, cash_flows, mid) > target:
            low = mid  # price too high -> yield must be higher
        else:
            high = mid
    return (low + high) / 2
