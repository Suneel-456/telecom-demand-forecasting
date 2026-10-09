# Telecom Demand Forecasting

Time-series project that forecasts daily data usage (TB) for a telecom region, so the network team can plan capacity for normal weeks and festival peaks.
**Note:** the dataset is simulated (weekly pattern, trend, festival and promo effects, plus injected missing values and outages).

## Approach
1. **Cleaning:** continuous daily index, interpolation for missing days, outage dips treated as errors, festival spikes kept
2. **EDA:** trend, weekly seasonality (period 7), ADF test, ACF/PACF
3. **Features:** lags (7/14/21/28), rolling means, calendar, holiday and promo flags
4. **Models:** SARIMA, Prophet (holiday and promo regressors), XGBoost, LSTM (PyTorch), seasonal-naive baseline
5. **Evaluation:** time-based split (last 20% = 130 days, 1 Jun to 8 Oct 2026), every prediction made from data at least 7 days old; metrics MAPE, RMSE, MAE

## Results (test period)
| Model | MAPE | RMSE | MAE | MAPE on holiday days |
|---|---|---|---|---|
| Prophet (holiday + promo) | 1.88% | 2.87 | 2.37 | 1.35% |
| XGBoost | 2.21% | 4.38 | 2.94 | 8.47% |
| SARIMA(1,1,1)(0,1,1,7) | 2.59% | 5.00 | 3.35 | 17.52% |
| LSTM | 2.66% | 4.77 | 3.41 | 14.01% |
| Seasonal naive | 3.11% | 6.37 | 4.06 | 18.10% |

Caveats: Prophet was refit daily while the other models were trained once; the test set has only 2 holiday days; the data is simulated, so rankings may differ on real data.

## Sample scenarios (Prophet, Bengaluru-South)
**Normal week (12 to 18 Oct 2026):** forecast 115.7 TB (Mon) to 138.0 TB (Sat)
**Festival week with promo (6 to 12 Nov 2026):** peak 169.2 TB on 8 Nov, about 25% above the same day without events
Full tables: `reports/test_case_outputs.csv`

## Run
```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m src.preprocessing
python -m src.models.sarima_model
python -m src.models.prophet_model
python -m src.models.xgboost_model
python -m src.models.lstm_model
python -m src.compare
python -m src.predict
```

## Structure
`data/` raw and processed data, `src/` pipeline code, `src/models/` model scripts, `reports/` metrics and figures