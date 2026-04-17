from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_list import (
    CURRENT_SAFE_COLS,
    LAG_SOURCE_COLS,
    MODEL_FEATURES,
    TARGET_COL,
    TARGET_SLEEP_EFFICIENCY_THRESHOLD,
)


@dataclass
class PreprocessResult:
    raw_df: pd.DataFrame
    model_df: pd.DataFrame
    inference_df: pd.DataFrame


def _ensure_calendar_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["calendar_date"] = pd.to_datetime(df["calendar_date"], errors="coerce")
    return df


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_base = ["SEQN", "calendar_date", "sleep_efficiency"]
    df = df.dropna(subset=[c for c in required_base if c in df.columns])

    df = _ensure_calendar_date(df)
    df = df.sort_values(["SEQN", "calendar_date"]).reset_index(drop=True)

    return df


def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = (df["sleep_efficiency"] < TARGET_SLEEP_EFFICIENCY_THRESHOLD).astype("int8")
    return df


def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in [1, 2, 3]:
        for col in LAG_SOURCE_COLS:
            if col not in df.columns:
                continue
            df[f"{col}_lag{lag}"] = df.groupby("SEQN")[col].shift(lag)

    return df


def _build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["month"] = df["calendar_date"].dt.month
    df["day"] = df["calendar_date"].dt.day
    df["dayofyear"] = df["calendar_date"].dt.dayofyear

    return df


def _build_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lag_feature_cols: list[str] = []
    for lag in [1, 2, 3]:
        lag_feature_cols.extend([f"{col}_lag{lag}" for col in LAG_SOURCE_COLS])

    model_cols = (
        CURRENT_SAFE_COLS
        + ["month", "day", "dayofyear", TARGET_COL]
        + lag_feature_cols
    )

    existing_cols = [c for c in model_cols if c in df.columns]
    model_df = df[existing_cols].copy()

    # Как и в ноутбуке: удаляем записи, где нет трёх лагов эффективности сна
    for c in ["sleep_efficiency_lag1", "sleep_efficiency_lag2", "sleep_efficiency_lag3"]:
        if c not in model_df.columns:
            raise ValueError(f"Не найден обязательный столбец после лагов: {c}")

    model_df = model_df.dropna(
        subset=[
            "sleep_efficiency_lag1",
            "sleep_efficiency_lag2",
            "sleep_efficiency_lag3",
        ]
    ).reset_index(drop=True)

    return model_df


def _prepare_for_inference(model_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.copy()

    drop_cols = [
        "SEQN",
        "calendar_date",
        TARGET_COL,
        "sleep_efficiency_lag1",
        "sleep_efficiency_lag2",
        "sleep_efficiency_lag3",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "После предобработки отсутствуют признаки, нужные модели: "
            + ", ".join(missing)
        )

    return df[MODEL_FEATURES].copy()


def preprocess_uploaded_dataframe(df: pd.DataFrame) -> PreprocessResult:
    raw_df = df.copy()

    df = _basic_cleaning(df)
    df = _build_target(df)
    df = _build_lag_features(df)
    df = _build_time_features(df)

    model_df = _build_model_dataframe(df)
    inference_df = _prepare_for_inference(model_df)

    return PreprocessResult(
        raw_df=raw_df,
        model_df=model_df,
        inference_df=inference_df,
    )