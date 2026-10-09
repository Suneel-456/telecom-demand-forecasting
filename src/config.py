from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "telecom_demand_sample.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "clean_data.csv"

DATE_COL = "date"
TARGET = "data_usage_TB"
TEST_SIZE = 0.2      # last 20% of dates used for testing
HORIZON = 7          # forecast days