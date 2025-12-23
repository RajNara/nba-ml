from pathlib import Path
import pandas as pd

# Get the absolute path to the project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "csv"

def read_csv_file(path) -> pd.DataFrame:
    """
    Reads a CSV file and returns its content as a pandas DataFrame.
    """
    full_path = DATA_PATH / path

    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    return pd.read_csv(full_path)

if __name__ == "__main__":
    read_csv_file()