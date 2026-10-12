"""Run the whole data pipeline in the right order.

Usage: python data/update.py

1. anagrafica.py     BTP universe from the Bank of Italy file
2. prezzi.py         clean prices from the Euronext files in data/raw/euronext/
3. check_prices.py   data-quality report
4. riferimento.py    €STR and Bund yields from the ECB and Bundesbank APIs
5. database.py       everything into data/btp.db

Before running: put new Euronext downloads in data/raw/euronext/ and, at the
start of each month, the new Bank of Italy file in data/raw/.
"""

from pathlib import Path
import subprocess
import sys

DATA_DIR = Path(__file__).parent
STEPS = ["anagrafica.py", "prezzi.py", "check_prices.py", "riferimento.py", "database.py"]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 60}\n>>> {step}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(DATA_DIR / step)])
        if result.returncode != 0:
            print(f"\nStopped: {step} failed. Fix the error above and run again.")
            sys.exit(1)
    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
