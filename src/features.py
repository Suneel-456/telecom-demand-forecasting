import pandas as pd
from src.config import PROCESSED_PATH, DATE_COL, TARGET, HORIZON, BASE_DIR

LAGS = [7, 14, 21, 28]
ROLL_WINDOWS = [7, 28]
BASE_FEATURES = ["holiday_flag", "promo_flag", "active_subscribers", "is_weekend", "month"]
CALENDAR_FEATURES = ["dow", "week_of_year", "day_of_month"]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"


def feature_columns():
    return (BASE_FEATURES + CALENDAR_FEATURES
            + [f"lag_{l}" for l in LAGS]
            + [f"roll_mean_{w}" for w in ROLL_WINDOWS])


def add_features(df, horizon=HORIZON):
    assert min(LAGS) >= horizon, "Lags must be >= forecast horizon to avoid leakage"
    df = df.copy()
    y = df[TARGET]

    # calendar features
    df["dow"] = df.index.dayofweek
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["day_of_month"] = df.index.day

    # lag features: usage 7, 14, 21, 28 days ago
    for lag in LAGS:
        df[f"lag_{lag}"] = y.shift(lag)

    # rolling averages, built only from data at least `horizon` days old
    shifted = y.shift(horizon)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()

    return df.dropna(subset=feature_columns())


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).asfreq("D")
    feats = add_features(df)
    feats.to_csv(FEATURES_PATH)
    print("Shape:", feats.shape)
    print("Features used:", len(feature_columns()))
    print(feats[[TARGET] + feature_columns()].head(3).T)
    print("Saved:", FEATURES_PATH)