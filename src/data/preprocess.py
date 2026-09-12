#RUL = max_cycle - current_cycle

from pathlib import Path
import pandas as pd


# CMAPSS dataset columns
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]

COLUMN_NAMES = ["engine_no", "cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS


def load_cmapss(raw_dir: Path, file_name: str) -> pd.DataFrame:
    """
    Load a CMAPSS text file into a pandas DataFrame.
    """
    path = Path(raw_dir) / file_name

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    df = df.sort_values(["engine_no", "cycle"]).reset_index(drop=True)
    return df


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) target for training data.
    """
    df = df.copy()

    max_cycle = df.groupby("engine_no")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]

    return df


def load_train(
    raw_dir: Path,
    train_file: str = "train_FD001.txt"
) -> pd.DataFrame:
    """
    Load training data and add RUL target.
    """
    df = load_cmapss(raw_dir, train_file)
    df = add_train_rul(df)
    return df


def load_test(
    raw_dir: Path,
    test_file: str = "test_FD001.txt",
    rul_file: str = "RUL_FD001.txt"
) -> pd.DataFrame:
    """
    Load test data and add RUL target using the provided RUL file.
    """
    df = load_cmapss(raw_dir, test_file)

    rul_path = Path(raw_dir) / rul_file
    end_rul = pd.read_csv(
        rul_path,
        header=None,
        names=["rul_at_end"]
    )

    engines = sorted(df["engine_no"].unique())

    if len(end_rul) != len(engines):
        raise ValueError(
            "RUL file length does not match number of engines in test file."
        )

    end_rul["engine_no"] = engines

    df = df.merge(end_rul, on="engine_no", how="left")

    max_cycle = df.groupby("engine_no")["cycle"].transform("max")

    # If final cycle has RUL = X,
    # earlier cycles have larger RUL.
    df["rul"] = df["rul_at_end"] + max_cycle - df["cycle"]

    df = df.drop(columns=["rul_at_end"])

    return df


def split_train_by_engine(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42
):
    """
    Split data by engine, not by random rows.

    This prevents data leakage where cycles from the same engine
    appear in both training and validation sets.
    """
    engines = pd.Series(sorted(df["engine_no"].unique()))
    engines = engines.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_val = max(1, int(len(engines) * val_frac))
    n_val = min(n_val, len(engines) - 1)

    val_engines = set(engines.iloc[:n_val])
    train_engines = set(engines.iloc[n_val:])

    train_df = df[df["engine_no"].isin(train_engines)].copy()
    val_df = df[df["engine_no"].isin(val_engines)].copy()

    return train_df, val_df


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the last observed cycle for each engine.

    This is useful because in real predictive maintenance,
    you usually make decisions using the latest available reading.
    """
    df = df.sort_values(["engine_no", "cycle"])
    return df.groupby("engine_no").tail(1).reset_index(drop=True)
