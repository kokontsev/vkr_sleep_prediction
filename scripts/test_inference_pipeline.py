from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from webapp.services.preprocess import preprocess_uploaded_dataframe
from webapp.services.predictor import SleepPredictor


# Укажите тестовый CSV из вашего проекта
TEST_CSV = ROOT / "data" / "raw" / "nhanes_day_level.csv"


def main() -> None:
    df = pd.read_csv(TEST_CSV)

    prep = preprocess_uploaded_dataframe(df)
    print("Raw shape:", prep.raw_df.shape)
    print("Model DF shape:", prep.model_df.shape)
    print("Inference DF shape:", prep.inference_df.shape)

    predictor = SleepPredictor()
    result = predictor.predict_last_row(prep.inference_df)

    print("\n=== Prediction ===")
    print("Probability:", round(result.probability, 4))
    print("Predicted class:", result.predicted_class)
    print("Predicted label:", result.predicted_label)
    print("Threshold:", result.used_threshold)


if __name__ == "__main__":
    main()