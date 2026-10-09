import pandas as pd
from src.config import TARGET, PROCESSED_PATH
from src.data_loader import load_data


def flag_outages(df, col=TARGET, window=7, drop_ratio=0.6):
    """Mark sudden downward dips (outages / bad records) as missing.
    Upward spikes are kept, because they are real festival demand."""
    
    df = df.copy()
    rolling_median = df[col].rolling(window, center=True, min_periods=1).median()
    outage = df[col] < drop_ratio * rolling_median
    df.loc[outage, col] = float("nan")
    return df, int(outage.sum())


def fill_missing(df, col=TARGET):
    df = df.copy()
    df[col] = df[col].interpolate(method="time", limit=3)      # short gaps
    if df[col].isna().any():                                    # longer gaps
        weekday_mean = df.groupby(df.index.dayofweek)[col].transform("mean")
        df[col] = df[col].fillna(weekday_mean)
    return df


def clean(df):
    missing_before = int(df[TARGET].isna().sum())
    df, n_outages = flag_outages(df)
    df = fill_missing(df)
    print(f"Missing values before : {missing_before}")
    print(f"Outage dips flagged   : {n_outages}")
    print(f"Missing values after  : {int(df[TARGET].isna().sum())}")
    return df


if __name__ == "__main__":
    df = clean(load_data())
    df.to_csv(PROCESSED_PATH)
    print("Saved:", PROCESSED_PATH, "| rows:", len(df))