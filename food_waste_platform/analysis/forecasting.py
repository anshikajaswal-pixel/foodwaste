from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def forecast_waste(df: pd.DataFrame, horizon_days: int = 14) -> list[dict[str, object]]:
    daily = df.groupby(df["date"].dt.date)["waste_kg"].sum()
    daily.index = pd.to_datetime(daily.index)
    ts = daily.asfreq("D", fill_value=0.0)

    x = np.arange(len(ts)).reshape(-1, 1)
    y = ts.values

    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(ts), len(ts) + horizon_days).reshape(-1, 1)
    preds = model.predict(future_x)
    residual_std = float(np.std(y - model.predict(x))) if len(y) > 1 else 0.5

    dates = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "predicted_waste_kg": round(max(float(p), 0.0), 2),
            "lower_kg": round(max(float(p - residual_std), 0.0), 2),
            "upper_kg": round(max(float(p + residual_std), 0.0), 2),
        }
        for d, p in zip(dates, preds)
    ]
