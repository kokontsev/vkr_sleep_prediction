from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "nhanes_day_level.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "nhanes_day_level_processed.csv"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = BASE_DIR / "figures"

RANDOM_STATE = 42
ID_COLUMN = "SEQN"
DATE_COLUMN = "calendar_date"
TARGET_SOURCE_COLUMN = "sleep_efficiency"
TARGET_COLUMN = "target_bad_sleep"