from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None


@dataclass
class WasteForecaster:
    horizon_days: int = 7

    def forecast(self, daily_series: pd.Series) -> list[dict[str, object]]:
        if daily_series.empty:
            return []

        ts = daily_series.copy()
        ts.index = pd.to_datetime(ts.index)
        ts = ts.asfreq("D", fill_value=0.0)

        if Prophet is not None and len(ts) >= 10:
            return self._forecast_prophet(ts)
        return self._forecast_regression(ts)

    def _forecast_prophet(self, ts: pd.Series) -> list[dict[str, object]]:
        p_df = pd.DataFrame({"ds": ts.index, "y": ts.values})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(p_df)
        future = model.make_future_dataframe(periods=self.horizon_days, freq="D")
        pred = model.predict(future).tail(self.horizon_days)

        out = []
        for _, row in pred.iterrows():
            out.append(
                {
                    "date": pd.Timestamp(row["ds"]).strftime("%Y-%m-%d"),
                    "predicted_waste_kg": round(max(float(row["yhat"]), 0.0), 2),
                    "lower_kg": round(max(float(row["yhat_lower"]), 0.0), 2),
                    "upper_kg": round(max(float(row["yhat_upper"]), 0.0), 2),
                }
            )
        return out

    def _forecast_regression(self, ts: pd.Series) -> list[dict[str, object]]:
        y = ts.values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)

        future_idx = np.arange(len(y), len(y) + self.horizon_days).reshape(-1, 1)
        preds = model.predict(future_idx)
        residual_std = float(np.std(y - model.predict(x))) if len(y) > 1 else 0.5

        start_date = ts.index.max() + pd.Timedelta(days=1)
        dates = pd.date_range(start=start_date, periods=self.horizon_days, freq="D")

        out = []
        for i, dt in enumerate(dates):
            p = max(float(preds[i]), 0.0)
            out.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "predicted_waste_kg": round(p, 2),
                    "lower_kg": round(max(p - residual_std, 0.0), 2),
                    "upper_kg": round(max(p + residual_std, 0.0), 2),
                }
            )
        return out
