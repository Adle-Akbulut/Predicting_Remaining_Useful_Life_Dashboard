import numpy as np
import pandas as pd

from src.data.preprocess import SENSOR_COLUMNS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-related features.
    """
    df = df.copy()

    df["cycle_log"] = np.log1p(df["cycle"])
    df["cycle_squared"] = df["cycle"] ** 2

    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list,
    windows=(5, 10)
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features
    for each sensor and each engine.

    Rolling features use only current and past values,
    which avoids future data leakage.
    """
    df = df.sort_values(["engine_no", "cycle"]).copy()
    grouped = df.groupby("engine_no")

    for window in windows:
        for col in sensor_cols:
            series = grouped[col]

            df[f"{col}_rmean_{window}"] = series.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            df[f"{col}_rstd_{window}"] = series.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).fillna(0.0)

    return df


def build_features(
    df: pd.DataFrame,
    windows=(5, 10)
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df = add_time_features(df)

    present_sensors = [
        col for col in SENSOR_COLUMNS if col in df.columns
    ]

    df = add_rolling_features(
        df,
        sensor_cols=present_sensors,
        windows=windows
    )

    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: set) -> list:
    """
    Return numeric feature columns for modeling.
    """
    exclude = set(exclude_cols)

    feature_columns = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_columns 
