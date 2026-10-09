import json
import logging
import warnings
import pandas as pd
from src.config import PROCESSED_PATH, DATE_COL, BASE_DIR
from src.models.prophet_model import to_prophet, fit_prophet, forecast_prophet

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

TEST_CASES = {
    "Test case 1: Normal week": {
        "region": "Bengaluru-South",
        "metric": "daily_data_usage_TB",
        "forecast_start": "2026-10-12",
        "horizon_days": 7,
        "holiday_flag": [0, 0, 0, 0, 0, 0, 0],
        "promo_active": False,
    },
    "Test case 2: Festival week": {
        "region": "Bengaluru-South",
        "metric": "daily_data_usage_TB",
        "forecast_start": "2026-11-06",
        "horizon_days": 7,
        "holiday_flag": [0, 0, 1, 1, 1, 1, 0],
        "promo_active": True,
    },
}


def forecast_case(model, case, with_events=True):
    days = case["horizon_days"]
    assert len(case["holiday_flag"]) == days, "holiday_flag must have one value per day"
    future = pd.DataFrame({
        "ds": pd.date_range(case["forecast_start"], periods=days, freq="D"),
        "holiday_flag": case["holiday_flag"] if with_events else [0] * days,
        "promo_flag": int(case["promo_active"]) if with_events else 0,
    })
    return forecast_prophet(model, future, use_regressors=True)


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_PATH, parse_dates=[DATE_COL], index_col=DATE_COL).asfreq("D")
    model = fit_prophet(to_prophet(df), use_regressors=True)
    print(f"Model trained on data up to {df.index[-1].date()}\n")

    frames = []
    for name, case in TEST_CASES.items():
        fc = forecast_case(model, case, with_events=True)
        base = forecast_case(model, case, with_events=False)    # same dates, no events

        out = pd.DataFrame({
            "Date": list(fc.index.strftime("%d %b")),
            "Day": list(fc.index.strftime("%a")),
            "Forecast_TB": fc["yhat"].round(1).values,
            "Lower": fc["yhat_lower"].round(1).values,
            "Upper": fc["yhat_upper"].round(1).values,
        })
        print("=" * 60)
        print(name)
        print("INPUT")
        print(json.dumps(case, indent=2))
        print("OUTPUT")
        print(out.to_string(index=False))

        peak = fc["yhat"].values.argmax()
        uplift = (fc["yhat"].values[peak] / base["yhat"].values[peak] - 1) * 100
        print(f"Peak: {fc.index[peak].strftime('%d %b')} at {fc['yhat'].values[peak]:.1f} TB "
              f"| {uplift:+.1f}% vs the same day without holiday/promo\n")

        out.insert(0, "Case", name)
        frames.append(out)

    (BASE_DIR / "reports").mkdir(exist_ok=True)
    pd.concat(frames).to_csv(BASE_DIR / "reports" / "test_case_outputs.csv", index=False)
    print("Saved reports/test_case_outputs.csv")