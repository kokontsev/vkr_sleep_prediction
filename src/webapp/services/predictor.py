from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from .feature_list import CLASSIFICATION_THRESHOLD, MODEL_FEATURES, MODEL_PATH


@dataclass
class PredictionOutput:
    probability: float
    predicted_class: int
    predicted_label: str
    used_threshold: float
    used_features: list[str]


class SleepPredictor:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))

    def predict_last_row(self, inference_df: pd.DataFrame) -> PredictionOutput:
        if inference_df.empty:
            raise ValueError("После предобработки не осталось строк для прогноза.")

        row_df = inference_df.tail(1).copy()
        proba = float(self.model.predict_proba(row_df)[0, 1])
        pred = int(proba >= self.threshold)

        return PredictionOutput(
            probability=proba,
            predicted_class=pred,
            predicted_label="Неблагоприятный сон" if pred == 1 else "Благоприятный сон",
            used_threshold=self.threshold,
            used_features=MODEL_FEATURES,
        )

    def predict_dataframe(self, inference_df: pd.DataFrame) -> pd.DataFrame:
        if inference_df.empty:
            raise ValueError("После предобработки не осталось строк для прогноза.")

        proba = self.model.predict_proba(inference_df)[:, 1]
        pred = (proba >= self.threshold).astype(int)

        result = inference_df.copy()
        result["probability_bad_sleep"] = proba
        result["predicted_class"] = pred
        result["predicted_label"] = result["predicted_class"].map(
            {0: "Благоприятный сон", 1: "Неблагоприятный сон"}
        )
        return result