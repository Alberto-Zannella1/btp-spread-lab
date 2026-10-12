"""Load daily BTP prices downloaded from Euronext (MOT) and build one clean table.

Input:  data/raw/euronext/*.csv, one file per bond (Euronext "Historical data" download).
Output: data/btp_prices.csv, one row per bond per trading day.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw" / "euronext"
UNIVERSE_FILE = DATA_DIR / "btp_universe.csv"
OUT_FILE = DATA_DIR / "btp_prices.csv"

COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Last": "last",
    "Close": "close",
    "Number of Shares": "nominal_traded",  # nominal amount traded, in euro
    "Number of Trades": "trades",
    "Turnover": "turnover",
    "vwap": "vwap",
}


def read_euronext_file(path: Path) -> pd.DataFrame:
    """Read one Euronext file. Line 3 holds the ISIN, the table starts on line 4."""
    with open(path, encoding="utf-8") as f:
        header = [next(f).strip().strip('"') for _ in range(3)]
    isin = header[2]

    table = pd.read_csv(path, sep=";", skiprows=3)
    table = table.rename(columns=COLUMNS)
    table["date"] = pd.to_datetime(table["date"], format="%d/%m/%Y")
    table.insert(0, "isin", isin)
    return table


def clean(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the data-quality rules.

    1. Drop days with no trading (exchange holidays): everything is 0 and the
       close is just the previous day's value carried forward.
    2. VWAP must lie between the day's low and high. If it does not, the value
       is a data error (e.g. 1.0224 instead of 102.24) and is discarded.
    3. Reference price = VWAP when valid, otherwise the close. VWAP is less
       sensitive to a single odd trade at the end of the day.
    """
    table = table[table["nominal_traded"] > 0].copy()

    valid_vwap = table["vwap"].between(table["low"], table["high"])
    table["vwap_error"] = ~valid_vwap
    table.loc[~valid_vwap, "vwap"] = float("nan")

    table["price"] = table["vwap"].fillna(table["close"])
    return table


def load_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read every file in the folder and stack them in one table."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    frames = [read_euronext_file(f) for f in files]
    prices = pd.concat(frames, ignore_index=True)
    # The same bond downloaded twice: keep one row per bond per day.
    return prices.drop_duplicates(subset=["isin", "date"])


if __name__ == "__main__":
    universe = pd.read_csv(UNIVERSE_FILE)
    raw = load_all()

    # Keep only the bonds in our universe (e.g. ignore a BTP Italia downloaded by mistake).
    extra = set(raw["isin"]) - set(universe["isin"])
    raw = raw[raw["isin"].isin(universe["isin"])]

    prices = clean(raw).sort_values(["isin", "date"]).reset_index(drop=True)
    prices.to_csv(OUT_FILE, index=False)

    have = set(prices["isin"])
    missing = universe[~universe["isin"].isin(have)]

    print(f"Bonds loaded: {len(have)} of {len(universe)} in the universe")
    print(f"Rows: {len(raw)} raw, {len(prices)} after cleaning")
    print(f"Holidays removed: {len(raw) - len(prices)}")
    print(f"VWAP errors fixed: {int(prices['vwap_error'].sum())}")
    print(f"Saved to: {OUT_FILE}")
    print()

    summary = prices.groupby("isin").agg(
        first_day=("date", "min"),
        last_day=("date", "max"),
        days=("date", "count"),
        avg_trades=("trades", "mean"),
    )
    summary["avg_trades"] = summary["avg_trades"].round(1)
    summary["first_day"] = summary["first_day"].dt.date
    summary["last_day"] = summary["last_day"].dt.date
    print(summary.to_string())

    if extra:
        print(f"\nIgnored (not in the universe): {sorted(extra)}")
    print(f"\nStill to download: {len(missing)} bonds")
