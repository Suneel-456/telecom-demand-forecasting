import numpy as np
from src.config import TEST_SIZE


def train_test_split_ts(df, test_size=TEST_SIZE):
    """Time-based split: first 80% to train, last 20% to test (no shuffling)."""
    split = int(len(df) * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def report(name, y_true, y_pred):
    return {
        "model": name,
        "MAPE": round(mape(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(mae(y_true, y_pred), 2),
    }