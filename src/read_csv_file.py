from pathlib import Path
import pandas as pd

# Get the absolute path to the project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "csv" / "game.csv"

def read_csv_file() -> pd.DataFrame:
    """
    Reads a CSV file and returns its content as a pandas DataFrame.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")

    if DATA_PATH.suffix != ".csv":
        raise ValueError(f"File is not a CSV file: {DATA_PATH}")
    elif DATA_PATH.exists() and DATA_PATH.suffix == ".csv":
        return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    read_csv_file()