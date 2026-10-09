import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import PROCESSED_PATH, DATE_COL, TARGET, HORIZON, BASE_DIR
from src.evaluate import train_test_split_ts, report, mape
from src.models.sarima_model import fit_sarima
from src.models.prophet_model import to_prophet, fit_prophet, forecast_prophet

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def sarima_predictions(df, test_index, order=(1, 1, 1), seasonal=(0, 1, 1, 7)):
    """Parameters learned on training data only; each test day is forecast
    from data up to HORIZON days earlier."""
    y = df[TARGET]
    res = fit_sarima(y[y.index < test_index[0]], order, seasonal)
    params = np.asarray(res.params)
    preds = []
    for t in test_index:
        origin = t - pd.Timedelta(days=HORIZON)
        mod = SARIMAX(y.loc[:origin], order=order, seasonal_order=seasonal,
                      enforce_stationarity=False, enforce_invertibility=False)
        preds.append(mod.filter(params).forecast(HORIZON).iloc[-1])
    return np.array(preds)


def prophet_predictions(df, test_index):
    """Prophet refit on data up to HORIZON days before each test day."""
    preds = []
    for i, t in enumerate(test_index, 1):
        origin = t - pd.Timedelta(days=HORIZON)
        m = fit_prophet(to_prophet(df.loc[:origin]), use_regressors=True)
        fc = forecast_prophet(m, to_prophet(df.loc[[t]]), use_regressors=True)
        preds.append(float(fc["yhat"].iloc[0]))
        if i % 20 == 0:
            print(f"  Prophet {i}/{len(test_index)} done")
    return np.array(preds)


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).asfreq("D")
    _, test_raw = train_test_split_ts(df)
    test_index = test_raw.index
    actual = df.loc[test_index, TARGET].values
    holiday = df.loc[test_index, "holiday_flag"].values == 1
    print(f"Test period: {test_index[0].date()} to {test_index[-1].date()} ({len(test_index)} days)")

    def make_row(name, pred):
        row = report(name, actual, pred)
        row["MAPE_holiday_days"] = round(mape(actual[holiday], pred[holiday]), 2)
        return row

    print("Running SARIMA ...")
    sarima_row = make_row("SARIMA(1,1,1)(0,1,1,7)", sarima_predictions(df, test_index))
    print("Running Prophet (this takes a few minutes) ...")
    prophet_row = make_row("Prophet (holiday + promo)", prophet_predictions(df, test_index))

    rep = BASE_DIR / "reports"
    saved = pd.concat([pd.read_csv(rep / "xgboost_metrics.csv"),
                       pd.read_csv(rep / "lstm_metrics.csv")], ignore_index=True)
    table = pd.concat([pd.DataFrame([sarima_row, prophet_row]), saved], ignore_index=True)
    table = table.sort_values("MAPE").reset_index(drop=True)
    print("\nFinal comparison (all models: prediction uses data at least 7 days old)")
    print(table.to_string(index=False))

    (rep / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(rep / "model_comparison.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.barh(table["model"], table["MAPE"])
    plt.gca().invert_yaxis()
    for i, v in enumerate(table["MAPE"]):
        plt.text(v + 0.05, i, f"{v:.2f}%", va="center")
    plt.xlabel("Test MAPE (%)")
    plt.title("Model comparison (7-day-ahead, same test period)")
    plt.tight_layout(); plt.savefig(rep / "figures" / "08_model_comparison.png"); plt.close()
    print("Saved model_comparison.csv and figure 08.")