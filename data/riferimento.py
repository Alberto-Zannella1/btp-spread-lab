"""Download reference rates from central-bank APIs.

- €STR (euro short-term rate), from the ECB Data Portal.
- German government (Bund) yields by residual maturity, from the Deutsche Bundesbank.
  These are zero-coupon yields from the Bundesbank's term structure estimate.

Both institutions publish these series openly through their APIs.
Output: data/reference_rates.csv, one row per day.
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
OUT_FILE = DATA_DIR / "reference_rates.csv"
START = "2024-01-01"

ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
BBK_URL = "https://api.statistiken.bundesbank.de/rest/data/{flow}/{key}"

# €STR, volume-weighted trimmed mean rate
ESTR = ("EST", "B.EU000A2X2A25.WT")

# Bund yields by residual maturity: the maturity code replaces {m}
BUND_KEY = "D.I.ZAR.ZI.EUR.S1311.B.A604.{m}.R.A.A._Z._Z.A"
BUND_MATURITIES = {"bund_2y": "R02XX", "bund_5y": "R05XX", "bund_10y": "R10XX", "bund_30y": "R30XX"}

SDMX_CSV_HEADER = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}

# Each API accepts a slightly different way of asking for CSV: try them in order.
ECB_REQUESTS = [({"format": "csvdata"}, {})]
BBK_REQUESTS = [({"format": "sdmx_csv"}, {}), ({}, SDMX_CSV_HEADER), ({"format": "csv"}, {})]


def parse_csv(text: str, name: str) -> pd.Series:
    """Turn a CSV answer into values indexed by date.

    Handles two layouts:
    - SDMX-CSV: columns TIME_PERIOD and OBS_VALUE (ECB and Bundesbank);
    - Bundesbank plain CSV: a few header lines, then rows 'date;value'.
    """
    sep = ";" if text.count(";") > text.count(",") else ","
    first_line = text.splitlines()[0] if text else ""

    if "TIME_PERIOD" in first_line and "OBS_VALUE" in first_line:
        table = pd.read_csv(StringIO(text), sep=sep, dtype=str)
        dates, values = table["TIME_PERIOD"], table["OBS_VALUE"]
    else:
        rows = [line.split(sep) for line in text.splitlines()]
        rows = [r for r in rows if len(r) >= 2 and len(r[0]) == 10 and r[0][4] == "-"]
        if not rows:
            raise ValueError(f"{name}: unexpected format. First lines:\n{text[:400]}")
        dates = pd.Series([r[0] for r in rows])
        values = pd.Series([r[1] for r in rows])

    numbers = pd.to_numeric(values.str.strip().str.replace(",", "."), errors="coerce")
    series = pd.Series(numbers.values, index=pd.to_datetime(dates.values), name=name)
    return series.dropna().sort_index()


def fetch(url: str, name: str, attempts: list) -> pd.Series:
    """Try each (params, headers) combination until one returns usable data."""
    errors = []
    for extra_params, headers in attempts:
        params = {"startPeriod": START, **extra_params}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            return parse_csv(response.text, name)
        except Exception as error:
            errors.append(str(error)[:200])
    raise RuntimeError(f"{name}: all attempts failed:\n  " + "\n  ".join(errors))


def get_estr() -> pd.Series:
    flow, key = ESTR
    return fetch(ECB_URL.format(flow=flow, key=key), "estr", ECB_REQUESTS)


def get_bund() -> pd.DataFrame:
    columns = []
    for name, maturity in BUND_MATURITIES.items():
        url = BBK_URL.format(flow="BBSIS", key=BUND_KEY.format(m=maturity))
        try:
            columns.append(fetch(url, name, BBK_REQUESTS))
        except Exception as error:  # keep going: one missing maturity should not stop the rest
            print(f"Warning: could not download {name}: {error}")
    return pd.concat(columns, axis=1)


if __name__ == "__main__":
    estr = get_estr()
    bund = get_bund()
    rates = pd.concat([estr, bund], axis=1).sort_index()
    rates.index.name = "date"
    rates.to_csv(OUT_FILE)

    print(f"Saved to: {OUT_FILE}")
    print(f"Days: {len(rates)}, from {rates.index.min().date()} to {rates.index.max().date()}")
    print("\nLast 5 days (% per year):")
    print(rates.tail(5).round(3).to_string())
    print("\nMissing values per column:")
    print(rates.isna().sum().to_string())
