from __future__ import annotations

import pandas as pd


def get_global_top_features(model, feature_names: list[str], top_n: int = 5) -> list[dict]:
    importances = model.get_feature_importance()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top_df = df.head(top_n).copy()

    return top_df.to_dict(orient="records")