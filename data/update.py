"""Run the whole data pipeline in the right order.

Usage: python data/update.py

1. anagrafica.py     BTP universe from the Bank of Italy file
2. prezzi.py         clean prices from the Euronext files in data/raw/euronext/
3. check_prices.py   data-quality report
4. riferimento.py    €STR and Bund yields from the ECB and Bundesbank APIs
5. database.py       everything into data/btp.db
6. pricing           yield, duration, DV01 for every bond and day (table 'analytics')
7. curve             daily NSS curve fit, rich/cheap residuals, BTP-Bund spread
8. signals           z-scores and half-lives of the residuals

Before running: put new Euronext downloads in data/raw/euronext/ and, at the
start of each month, the new Bank of Italy file in data/raw/.
"""

from pathlib import Path
import subprocess
import sys

DATA_DIR = Path(__file__).parent
ROOT = DATA_DIR.parent

# Each step is the command after "python"
STEPS = [
    [str(DATA_DIR / "anagrafica.py")],
    [str(DATA_DIR / "prezzi.py")],
    [str(DATA_DIR / "check_prices.py")],
    [str(DATA_DIR / "riferimento.py")],
    [str(DATA_DIR / "database.py")],
    ["-m", "pricing.run_pricing"],
    ["-m", "curve.fit_curve"],
    ["-m", "signals.zscore"],
]


def main() -> None:
    for step in STEPS:
        name = Path(step[-1]).name
        print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}")
        result = subprocess.run([sys.executable, *step], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStopped: {name} failed. Fix the error above and run again.")
            sys.exit(1)
    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
