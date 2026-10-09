import pandas as pd
from src.config import RAW_PATH, DATE_COL


def load_data(path=RAW_PATH):

    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(subset=DATE_COL)
    df = df.set_index(DATE_COL).asfreq("D")   # continuous daily index
    return df


if __name__ == "__main__":
    
    df = load_data()
    print(df.shape)
    print(df.index.min(), "to", df.index.max())
    print(df.isna().sum())
    print(df.head())