 
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils.io import ensure_dir, save_json


DEFAULT_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 350,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": 0
}


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    model_params=None
):
    """
    Train an XGBoost regression model inside a scikit-learn pipeline.

    Pipeline steps:
    1. Impute missing values
    2. Scale features
    3. Train XGBoost model
    """
    params = DEFAULT_MODEL_PARAMS.copy()

    if model_params is not None:
        params.update(model_params)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(**params))
    ])

    model.fit(X_train, y_train)

    val_pred = None
    if X_val is not None:
        val_pred = model.predict(X_val)

    return model, val_pred


def get_feature_importance(model, feature_names, top_n=15):
    """
    Get top feature importances from trained XGBoost model.
    """
    xgb_model = model.named_steps["model"]
    importances = xgb_model.feature_importances_

    order = np.argsort(importances)[::-1][:top_n]

    top_features = [
        {
            "feature": feature_names[i],
            "importance": float(importances[i])
        }
        for i in order
    ]

    return top_features


def save_artifacts(model, feature_columns, metadata, model_dir):
    """
    Save model, feature columns, and metadata.
    """
    model_dir = Path(model_dir)
    ensure_dir(model_dir)

    joblib.dump(model, model_dir / "model.joblib")
    save_json(list(feature_columns), model_dir / "feature_columns.json")
    save_json(metadata, model_dir / "metadata.json")