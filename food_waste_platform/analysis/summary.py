from __future__ import annotations

from typing import Any

import pandas as pd


def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    daily = df.groupby(df["date"].dt.date)["waste_kg"].sum().sort_index()

    return {
        "total_waste_kg": round(float(df["waste_kg"].sum()), 2),
        "waste_by_meal": {
            k: round(float(v), 2)
            for k, v in df.groupby("meal_type")["waste_kg"].sum().sort_values(ascending=False).items()
        },
        "waste_by_commodity": {
            k: round(float(v), 2)
            for k, v in df.groupby("commodity")["waste_kg"].sum().sort_values(ascending=False).items()
        },
        "waste_by_kitchen": {
            k: round(float(v), 2)
            for k, v in df.groupby("kitchen")["waste_kg"].sum().sort_values(ascending=False).items()
        },
        "daily_trend": [
            {"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "waste_kg": round(float(val), 2)}
            for day, val in daily.items()
        ],
    }
