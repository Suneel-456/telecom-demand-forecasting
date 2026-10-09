import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.config import PROCESSED_PATH, DATE_COL, TARGET, BASE_DIR

FIG_DIR = BASE_DIR / "reports" / "figures"


def adf_test(series, name):
    stat, p, *_ = adfuller(series.dropna())
    verdict = "stationary" if p < 0.05 else "NOT stationary"
    print(f"ADF | {name:<28} stat={stat:7.3f}  p={p:.4f}  -> {verdict}")


if __name__ == "__main__":
    
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).asfreq("D")
    y = df[TARGET]

    adf_test(y, "original")
    adf_test(y.diff(1), "1st difference")
    adf_test(y.diff(7).diff(1), "seasonal(7) + 1st difference")

    plt.figure(figsize=(12, 4))
    y.plot(title="Daily data usage (TB)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_usage_over_time.png"); plt.close()

    seasonal_decompose(y, model="additive", period=7).plot()
    plt.gcf().set_size_inches(12, 8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_decomposition.png"); plt.close()

    d = y.diff(7).diff(1).dropna()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(d, lags=40, ax=ax[0]); plot_pacf(d, lags=40, ax=ax[1])
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_acf_pacf.png"); plt.close()

    print("Figures saved in:", FIG_DIR)