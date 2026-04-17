from __future__ import annotations

from pathlib import Path

MODEL_PATH = Path("models/catboost/catboost_tuned_no_sleep_quality_lag.cbm")
CLASSIFICATION_THRESHOLD = 0.30
TARGET_SLEEP_EFFICIENCY_THRESHOLD = 0.85

CURRENT_SAFE_COLS = [
    "SEQN",
    "calendar_date",
    "weekday",
    "dayofweek",
    "window_number",
    "night_number",
]

LAG_SOURCE_COLS = [
    "sleep_efficiency",
    "nonwear_perc_day",
    "nonwear_perc_spt",
    "nonwear_perc_day_spt",
    "sleeponset",
    "wakeup",
    "L5VALUE",
    "L5TIME_num",
    "M5VALUE",
    "M5TIME_num",
    "L10VALUE",
    "L10TIME_num",
    "M10VALUE",
    "M10TIME_num",
    "dur_spt_sleep_min",
    "dur_spt_wake_IN_min",
    "dur_spt_wake_LIG_min",
    "dur_spt_wake_MOD_min",
    "dur_spt_wake_VIG_min",
    "dur_day_min",
    "dur_spt_min",
    "dur_day_spt_min",
    "ACC_spt_sleep_mg",
    "excluded",
]

TIME_DERIVED_COLS = [
    "month",
    "day",
    "dayofyear",
]

DROP_FOR_INFERENCE = [
    "SEQN",
    "calendar_date",
    "sleep_efficiency_lag1",
    "sleep_efficiency_lag2",
    "sleep_efficiency_lag3",
]

TARGET_COL = "target_bad_sleep"


def build_model_columns() -> list[str]:
    lag_feature_cols: list[str] = []
    for lag in [1, 2, 3]:
        lag_feature_cols.extend([f"{col}_lag{lag}" for col in LAG_SOURCE_COLS])

    model_cols = (
        CURRENT_SAFE_COLS
        + TIME_DERIVED_COLS
        + [TARGET_COL]
        + lag_feature_cols
    )
    return model_cols


def build_inference_features() -> list[str]:
    model_cols = build_model_columns()
    features = [c for c in model_cols if c != TARGET_COL]
    features = [c for c in features if c not in DROP_FOR_INFERENCE]
    return features


MODEL_FEATURES = build_inference_features()