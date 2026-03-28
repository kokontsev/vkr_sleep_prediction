from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "nhanes_day_level.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "nhanes_day_level_processed.csv"
MODEL_DATA_PATH = BASE_DIR / "data" / "processed" / "nhanes_model_table.csv"

MODEL_DIR = BASE_DIR / "models"
CATBOOST_MODEL_DIR = MODEL_DIR / "catboost"

REPORT_DIR = BASE_DIR / "reports"
REPORT_TABLES_DIR = REPORT_DIR / "tables"
REPORT_TEXT_DIR = REPORT_DIR / "text"
REPORT_LOGS_DIR = REPORT_DIR / "logs"

FIGURE_DIR = BASE_DIR / "figures"
CM_FIG_DIR = FIGURE_DIR / "confusion_matrices"
FI_FIG_DIR = FIGURE_DIR / "feature_importance"
THRESHOLD_FIG_DIR = FIGURE_DIR / "thresholds"

RANDOM_STATE = 42
ID_COLUMN = "SEQN"
DATE_COLUMN = "calendar_date"
TARGET_SOURCE_COLUMN = "sleep_efficiency"
TARGET_COLUMN = "target_bad_sleep"

for path in [
    CATBOOST_MODEL_DIR,
    REPORT_TABLES_DIR,
    REPORT_TEXT_DIR,
    REPORT_LOGS_DIR,
    CM_FIG_DIR,
    FI_FIG_DIR,
    THRESHOLD_FIG_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)