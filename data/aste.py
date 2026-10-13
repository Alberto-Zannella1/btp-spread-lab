"""Read the Bank of Italy files with the results of government bond auctions.

Source: Banca d'Italia, "Risultati aste dei titoli di Stato"
(https://www.bancaditalia.it/compiti/operazioni-mef/risultati-aste/), two files
downloaded by hand into data/raw/aste/: the history (2002 to last year) and the
current year. Download the current-year file again from time to time.

Output: data/auctions.csv, one row per BTP auction tranche.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw" / "aste"
OUT_FILE = DATA_DIR / "auctions.csv"

# Column positions in the Bank of Italy layout (the header spans 3 rows)
COLUMNS = {0: "date", 1: "settle_date", 2: "isin", 3: "tranche", 4: "kind", 5: "description",
           8: "type", 13: "allotted_mln", 16: "yield"}


def read_file(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, skiprows=4)
    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    for col in ["date", "settle_date"]:
        # the history file stores real dates, the current-year file stores text like '13-01-2026'
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date", "isin"])


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("*.xls*"))
    if not files:
        raise SystemExit(f"No auction files in {RAW_DIR}")
    auctions = pd.concat([read_file(f) for f in files]).drop_duplicates(subset=["date", "isin", "kind"])
    btp = auctions[auctions["type"] == "BTP"].sort_values("date")  # fixed-rate nominal BTPs only
    btp.to_csv(OUT_FILE, index=False, date_format="%Y-%m-%d")

    print(f"Read {len(files)} files: {len(auctions)} auction tranches, {len(btp)} of fixed-rate BTPs")
    print(f"From {btp['date'].min().date()} to {btp['date'].max().date()}")
    print("\nKind: O = ordinary auction, S = supplementary reopening for specialists (the day after)")
    print(btp["kind"].value_counts().to_string())
    print(f"\nSaved to {OUT_FILE}")
