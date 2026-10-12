"""Build the BTP universe from the Bank of Italy list of outstanding government bonds.

Source: Banca d'Italia, "Titoli di Stato in circolazione" (updated monthly).
Output: data/btp_universe.csv, one row per fixed-rate nominal BTP.
"""

from pathlib import Path
import re

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR / "raw" / "titoli_in_circolazione.xls"
OUT_FILE = DATA_DIR / "btp_universe.csv"

# Excel stores dates as the number of days since 30 Dec 1899.
EXCEL_EPOCH = pd.Timestamp("1899-12-30")

COLUMNS = [
    "isin",
    "description",
    "type",
    "issue_date",
    "maturity",
    "rate_type",
    "outstanding_mln",
    "current_coupon",
]


def to_date(value) -> pd.Timestamp:
    """Convert an Excel serial number or a 'dd-mm-yyyy' string to a date."""
    if isinstance(value, (int, float)):
        return EXCEL_EPOCH + pd.Timedelta(days=value)
    return pd.to_datetime(value, format="%d-%m-%Y")


def load_raw(path: Path = RAW_FILE) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Read the Bank of Italy file and return (table, reference date)."""
    raw = pd.read_excel(path, header=None, engine="xlrd")

    # Cell H1 holds the date the file refers to ("dati aggiornati al").
    ref_date = to_date(raw.iloc[0, 7])

    # Rows 1-4 are headers and blank lines; the bonds start on row 5.
    table = raw.iloc[4:, :8].copy()
    table.columns = COLUMNS
    table = table[table["isin"].notna()]  # drop empty rows at the bottom

    table["issue_date"] = table["issue_date"].map(to_date)
    table["maturity"] = table["maturity"].map(to_date)
    return table, ref_date


def annual_coupon(description: str) -> float:
    """Read the annual coupon from the description, e.g. 'BTP 4,65% ...' -> 4.65.

    We do not use the 'current_coupon' column: for newly issued bonds it holds
    the first (short) coupon, not the regular semi-annual one.
    """
    match = re.search(r"([\d,]+)%", description)
    return float(match.group(1).replace(",", "."))


def build_btp_universe(
    table: pd.DataFrame, as_of: pd.Timestamp, min_years: float = 1.0
) -> pd.DataFrame:
    """Keep only fixed-rate nominal BTPs with at least `min_years` to maturity.

    Excluded: BOT (zero coupon), CCTeu (floating), BTP€i and BTP Italia
    (inflation-linked), BTP Valore and Futura (retail), BTP Green (green premium).
    """
    btp = table[table["type"] == "BTP"].copy()
    btp["coupon"] = btp["description"].map(annual_coupon)
    btp["years_to_maturity"] = (btp["maturity"] - as_of).dt.days / 365.25
    btp = btp[btp["years_to_maturity"] >= min_years]

    columns = [
        "isin",
        "description",
        "coupon",
        "issue_date",
        "maturity",
        "years_to_maturity",
        "outstanding_mln",
    ]
    return btp[columns].sort_values("maturity").reset_index(drop=True)


if __name__ == "__main__":
    table, ref_date = load_raw()
    today = pd.Timestamp.today().normalize()
    universe = build_btp_universe(table, as_of=today)
    universe.to_csv(OUT_FILE, index=False)

    print(f"Bank of Italy file as of: {ref_date.date()}")
    print(f"Bonds in file: {len(table)}")
    print(f"Fixed-rate nominal BTPs with at least 1 year to maturity: {len(universe)}")
    print(f"Maturities from {universe['maturity'].min().date()} to {universe['maturity'].max().date()}")
    print(f"Saved to: {OUT_FILE}")
    print()
    print(universe.head(10).to_string(index=False))
