from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import pandas as pd


ALLOWED_EXTENSIONS = {".csv"}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    dataframe: pd.DataFrame | None = None


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    return [str(col).strip() for col in columns]


def validate_file_extension(filename: str) -> tuple[bool, str | None]:
    if not filename:
        return False, "Имя файла не передано."
    dot_index = filename.rfind(".")
    ext = filename[dot_index:].lower() if dot_index != -1 else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Допустим только CSV-файл. Получено: {ext or 'без расширения'}."
    return True, None


def read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("Файл пустой.")
    return pd.read_csv(BytesIO(file_bytes))


def validate_input_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    min_rows: int = 1,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if df is None:
        return ValidationResult(False, ["DataFrame не передан."], warnings)

    if df.empty:
        return ValidationResult(False, ["Файл не содержит строк данных."], warnings)

    if len(df) < min_rows:
        errors.append(f"Недостаточно строк для прогноза. Требуется минимум: {min_rows}.")

    df.columns = _normalize_columns(df.columns)
    required_columns = _normalize_columns(required_columns)

    missing = [col for col in required_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in required_columns]

    if missing:
        errors.append(
            "В файле отсутствуют обязательные столбцы: "
            + ", ".join(sorted(missing))
        )

    if extra:
        warnings.append(
            "В файле присутствуют дополнительные столбцы, они будут проигнорированы: "
            + ", ".join(sorted(extra))
        )

    if df.isna().all(axis=1).any():
        warnings.append("Обнаружены полностью пустые строки. Их лучше удалить.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        dataframe=df,
    )


def validate_uploaded_csv(
    filename: str,
    file_bytes: bytes,
    required_columns: list[str],
    min_rows: int = 1,
) -> ValidationResult:
    ok, ext_error = validate_file_extension(filename)
    if not ok:
        return ValidationResult(False, [ext_error or "Некорректный файл."], [])

    try:
        df = read_csv_from_bytes(file_bytes)
    except Exception as exc:
        return ValidationResult(
            False,
            [f"Не удалось прочитать CSV-файл: {exc}"],
            [],
        )

    return validate_input_dataframe(
        df=df,
        required_columns=required_columns,
        min_rows=min_rows,
    )