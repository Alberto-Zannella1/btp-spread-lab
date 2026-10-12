"""Store all project data in one SQLite database: data/btp.db.

Tables:
- bonds           the BTP universe (from btp_universe.csv)
- prices          daily clean prices (from btp_prices.csv)
- reference_rates €STR and Bund yields (from reference_rates.csv)

The later modules (pricing, curve, signals...) read from here, so they do not
need to know where each CSV comes from.
"""

from pathlib import Path
import sqlite3

import pandas as pd

DATA_DIR = Path(__file__).parent
DB_FILE = DATA_DIR / "btp.db"

SOURCES = {
    "bonds": (DATA_DIR / "btp_universe.csv", ["issue_date", "maturity"]),
    "prices": (DATA_DIR / "btp_prices.csv", ["date"]),
    "reference_rates": (DATA_DIR / "reference_rates.csv", ["date"]),
}


def build() -> None:
    """Rebuild the database from the CSV files (replaces existing tables)."""
    with sqlite3.connect(DB_FILE) as conn:
        for table, (csv_file, date_columns) in SOURCES.items():
            df = pd.read_csv(csv_file, parse_dates=date_columns)
            for col in date_columns:
                df[col] = df[col].dt.strftime("%Y-%m-%d")  # store dates as text 'YYYY-MM-DD'
            df.to_sql(table, conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices ON prices (isin, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rates ON reference_rates (date)")


def read(query: str) -> pd.DataFrame:
    """Run a SQL query and return a table. Example: read('SELECT * FROM bonds')."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql(query, conn)


if __name__ == "__main__":
    build()
    print(f"Database: {DB_FILE}")
    for table in SOURCES:
        n = read(f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0]
        print(f"  {table:<16} {n:>7} rows")

    print("\nExample query - latest price of each BTP maturing in 2035:")
    print(read("""
        SELECT b.description, p.date, ROUND(p.price, 2) AS price
        FROM prices p JOIN bonds b ON b.isin = p.isin
        WHERE b.maturity LIKE '2035%'
          AND p.date = (SELECT MAX(date) FROM prices WHERE isin = p.isin)
        ORDER BY b.maturity
    """).to_string(index=False))
