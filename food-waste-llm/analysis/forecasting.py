from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class WasteForecaster:
    horizon_days: int = 7
    arima_order: tuple[int, int, int] = (2, 1, 1)

    def forecast(self, daily_waste: pd.Series) -> pd.DataFrame:
        if daily_waste.empty:
            return pd.DataFrame(columns=["date", "predicted_waste_kg", "lower_kg", "upper_kg"])

        ts = daily_waste.copy()
        ts.index = pd.to_datetime(ts.index)
        ts = ts.asfreq("D", fill_value=0.0)

        if len(ts) < 10:
            mean_val = float(ts.mean())
            future_dates = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=self.horizon_days, freq="D")
            rows = [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "predicted_waste_kg": round(mean_val, 2),
                    "lower_kg": round(max(mean_val * 0.8, 0.0), 2),
                    "upper_kg": round(mean_val * 1.2, 2),
                }
                for d in future_dates
            ]
            return pd.DataFrame(rows)

        try:
            model = ARIMA(ts, order=self.arima_order)
            fitted = model.fit()
            forecast_res = fitted.get_forecast(steps=self.horizon_days)
            preds = forecast_res.predicted_mean
            conf = forecast_res.conf_int(alpha=0.2)

            out = pd.DataFrame(
                {
                    "date": preds.index.strftime("%Y-%m-%d"),
                    "predicted_waste_kg": np.maximum(preds.values, 0),
                    "lower_kg": np.maximum(conf.iloc[:, 0].values, 0),
                    "upper_kg": np.maximum(conf.iloc[:, 1].values, 0),
                }
            )
            return out.round(2)
        except Exception:
            # Fallback keeps pipeline resilient if ARIMA fails to converge.
            rolling_mean = float(ts.tail(7).mean())
            rolling_std = float(ts.tail(7).std(ddof=0) or 0.0)
            future_dates = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=self.horizon_days, freq="D")

            rows = [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "predicted_waste_kg": round(max(rolling_mean, 0), 2),
                    "lower_kg": round(max(rolling_mean - rolling_std, 0), 2),
                    "upper_kg": round(max(rolling_mean + rolling_std, 0), 2),
                }
                for d in future_dates
            ]
            return pd.DataFrame(rows)
