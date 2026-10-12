"""Check that the Euronext price download is complete and sensible.

Run after data/prezzi.py. Prints a report and a list of bonds to re-download.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
UNIVERSE_FILE = DATA_DIR / "btp_universe.csv"
PRICES_FILE = DATA_DIR / "btp_prices.csv"

# Euronext gives at most two years of history.
HISTORY_LIMIT_YEARS = 2
# Tolerances
MAX_MISSING_DAYS = 10        # missing trading days allowed inside a bond's history
MAX_START_DELAY_DAYS = 10    # trading days allowed between expected and actual first day
MAX_DAILY_MOVE = 3.0         # price points; a bigger daily move is flagged for review


def main() -> None:
    universe = pd.read_csv(UNIVERSE_FILE, parse_dates=["issue_date", "maturity"])
    prices = pd.read_csv(PRICES_FILE, parse_dates=["date"])

    # The market calendar = every day on which at least one bond traded.
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    last_market_day = calendar.max()
    history_start = last_market_day - pd.DateOffset(years=HISTORY_LIMIT_YEARS)

    problems = []   # (isin, description, problem)

    for _, bond in universe.iterrows():
        isin, desc = bond["isin"], bond["description"]
        p = prices[prices["isin"] == isin].sort_values("date")

        # 1. Bond missing completely
        if p.empty:
            problems.append((isin, desc, "no prices at all"))
            continue

        # 2. History starts later than it should (wrong 'From' date?)
        expected_start = max(bond["issue_date"], history_start)
        expected_days = calendar[(calendar >= expected_start) & (calendar <= p["date"].min())]
        if len(expected_days) > MAX_START_DELAY_DAYS:
            problems.append((isin, desc,
                             f"starts {p['date'].min().date()}, expected around {expected_start.date()}"))

        # 3. History does not reach the latest market day (old download?)
        if p["date"].max() < last_market_day:
            problems.append((isin, desc, f"ends {p['date'].max().date()}, market data to {last_market_day.date()}"))

        # 4. Holes inside the history
        span = calendar[(calendar >= p["date"].min()) & (calendar <= p["date"].max())]
        missing = len(span) - p["date"].nunique()
        if missing > MAX_MISSING_DAYS:
            problems.append((isin, desc, f"{missing} trading days missing inside the history"))

        # 5. Prices that look wrong
        if not p["price"].between(50, 150).all():
            problems.append((isin, desc, "price outside 50-150"))
        jumps = p["price"].diff().abs()
        big = p.loc[jumps > MAX_DAILY_MOVE, "date"]
        if len(big):
            problems.append((isin, desc,
                             f"{len(big)} daily moves > {MAX_DAILY_MOVE} points, e.g. {big.iloc[0].date()}"))

    # Report
    print(f"Universe: {len(universe)} bonds | with prices: {prices['isin'].nunique()}")
    print(f"Market days: {calendar.min().date()} to {last_market_day.date()} ({len(calendar)} days)")
    print()
    if not problems:
        print("All checks passed.")
        return

    report = pd.DataFrame(problems, columns=["isin", "description", "problem"])
    print(f"{report['isin'].nunique()} bonds to review:\n")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
