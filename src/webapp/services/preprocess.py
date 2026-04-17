from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class PreprocessConfig:
    id_col: str = "SEQN"
    date_col: str = "calendar_date"


def ensure_datetime_columns(df: pd.DataFrame, datetime_columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sort_panel_dataframe(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    df = df.copy()
    sort_cols = [c for c in [config.id_col, config.date_col] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    lag_columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    df = df.copy()
    for col in lag_columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Пример. Замените на вашу фактическую логику из ноутбука.
    if "calendar_date" in df.columns:
        dt = pd.to_datetime(df["calendar_date"], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day

    return df


def fill_missing_values(df: pd.DataFrame, fill_map: dict[str, float | int | str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if fill_map:
        for col, value in fill_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
    return df


def select_model_features(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    missing = [col for col in model_features if col not in df.columns]
    if missing:
        raise ValueError(
            "После предобработки отсутствуют признаки, нужные модели: "
            + ", ".join(missing)
        )
    return df[model_features].copy()


def preprocess_for_inference(
    df: pd.DataFrame,
    model_features: list[str],
    lag_columns: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    df = df.copy()

    # 1. Базовые преобразования
    df = ensure_datetime_columns(df, ["calendar_date", "date_time_onset", "date_time_wakeup"])
    df = sort_panel_dataframe(df, PreprocessConfig())

    # 2. Ваша инженерия признаков
    df = add_basic_time_features(df)

    # 3. Лаговые признаки
    if lag_columns and lags and "SEQN" in df.columns:
        df = add_lag_features(
            df=df,
            group_col="SEQN",
            lag_columns=lag_columns,
            lags=lags,
        )

    # 4. Заполнение пропусков
    df = fill_missing_values(df)

    # 5. Оставляем только нужные признаки
    return select_model_features(df, model_features)