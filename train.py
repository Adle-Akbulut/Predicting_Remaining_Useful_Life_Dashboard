from pathlib import Path

import pandas as pd

from src.data.preprocess import (
    load_train,
    load_test,
    split_train_by_engine,
    last_cycle,
    SENSOR_COLUMNS,
    SETTING_COLUMNS
)
from src.features.build_features import (
    build_features,
    get_feature_columns
)
from src.models.train import (
    train_model,
    save_artifacts,
    get_feature_importance
)
from src.models.evaluate import regression_metrics
from src.utils.io import ensure_dir


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "NASA C-MAPSS-1 Turbofan Engine Degradation Dataset"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

TARGET = "rul"

# Cap RUL to avoid the model spending too much effort
# on very early healthy cycles.
TARGET_CAP = 125

# Rolling feature windows
WINDOWS = (5, 10)

# Validation split fraction by engine
VAL_FRACTION = 0.2

SEED = 42


def cap_target(df: pd.DataFrame, target: str, cap: int) -> pd.DataFrame:
    """
    Cap target RUL.

    Example:
    If true RUL is 200, cap it to 125.

    This is common in turbofan RUL modeling because
    early-life degradation may not be very informative.
    """
    df = df.copy()

    if cap is not None:
        df[target] = df[target].clip(upper=cap)

    return df


def print_metrics(name: str, metrics: dict):
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main():
    print("Loading raw train and test data...")

    train_raw = load_train(
        RAW_DIR,
        train_file="train_FD002.txt"
    )

    test_raw = load_test(
        RAW_DIR,
        test_file="test_FD002.txt",
        rul_file="RUL_FD002.txt"
    )

    train_raw = cap_target(train_raw, TARGET, TARGET_CAP)
    test_raw = cap_target(test_raw, TARGET, TARGET_CAP)

    print("Building features...")

    train_features = build_features(train_raw, windows=WINDOWS)
    test_features = build_features(test_raw, windows=WINDOWS)

    # Split training data by engine to avoid leakage
    train_split, val_split = split_train_by_engine(
        train_features,
        val_frac=VAL_FRACTION,
        seed=SEED
    )

    feature_columns = get_feature_columns(
        train_split,
        exclude_cols={"engine_no", TARGET}
    )

    X_train = train_split[feature_columns]
    y_train = train_split[TARGET]

    # Evaluate validation on last cycle per engine
    val_last = last_cycle(val_split)

    X_val = val_last[feature_columns]
    y_val = val_last[TARGET]

    print("Training XGBoost model...")

    model, val_pred = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    val_metrics = regression_metrics(y_val, val_pred)

    print("Evaluating on test set using last cycle per engine...")

    test_last = last_cycle(test_features)

    test_pred = model.predict(test_last[feature_columns])
    test_metrics = regression_metrics(test_last[TARGET], test_pred)

    top_features = get_feature_importance(
        model=model,
        feature_names=feature_columns,
        top_n=15
    )

    metadata = {
        "target": TARGET,
        "target_cap": TARGET_CAP,
        "windows": list(WINDOWS),
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "feature_count": len(feature_columns),
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics
        },
        "top_features": top_features
    }

    print("Saving model artifacts...")

    save_artifacts(
        model=model,
        feature_columns=feature_columns,
        metadata=metadata,
        model_dir=MODEL_DIR
    )

    print("Saving dashboard dataset...")

    # Predict RUL for all test rows so dashboard can show trends
    test_features["predicted_rul"] = model.predict(
        test_features[feature_columns]
    )

    if TARGET_CAP is not None:
        test_features["predicted_rul"] = test_features["predicted_rul"].clip(
            lower=0,
            upper=TARGET_CAP
        )
    else:
        test_features["predicted_rul"] = test_features["predicted_rul"].clip(
            lower=0
        )

    dashboard_columns = (
        ["engine_no", "cycle", TARGET, "predicted_rul"]
        + SETTING_COLUMNS
        + SENSOR_COLUMNS
    )

    ensure_dir(PROCESSED_DIR)

    test_features[dashboard_columns].to_csv(
        PROCESSED_DIR / "test_dashboard.csv",
        index=False
    )

    print_metrics("Validation metrics", val_metrics)
    print_metrics("Test metrics", test_metrics)

    print("\nTraining complete.")
    print("Run the dashboard with:")
    print("streamlit run app/main.py")


if __name__ == "__main__":
    main()