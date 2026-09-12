 
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import build_features
from src.utils.io import load_json


def load_artifacts(model_dir):
    """
    Load trained model, feature columns, and metadata.
    """
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "model.joblib")
    feature_columns = load_json(model_dir / "feature_columns.json")
    metadata = load_json(model_dir / "metadata.json")

    return model, feature_columns, metadata


def predict_raw_dataframe(
    raw_df: pd.DataFrame,
    model,
    feature_columns: list,
    metadata: dict
) -> pd.DataFrame:
    """
    Predict RUL for a raw dataframe containing sensor readings.
    """
    df = raw_df.copy()

    # Convert important columns to numeric
    df["engine_no"] = pd.to_numeric(df["engine_no"], errors="coerce")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")

    numeric_columns = [col for col in df.columns if col != "engine_no"]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.dropna(subset=["engine_no", "cycle"])

    if df.empty:
        return df

    df["engine_no"] = df["engine_no"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    # Rebuild the same features used during training
    windows = tuple(metadata.get("windows", [5, 10]))
    df = build_features(df, windows=windows)

    missing_columns = [
        col for col in feature_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input data is missing required feature columns: {missing_columns}"
        )

    predictions = model.predict(df[feature_columns])

    df["predicted_rul"] = predictions

    target_cap = metadata.get("target_cap")

    if target_cap is not None:
        df["predicted_rul"] = df["predicted_rul"].clip(
            lower=0,
            upper=target_cap
        )
    else:
        df["predicted_rul"] = df["predicted_rul"].clip(lower=0)

    return df