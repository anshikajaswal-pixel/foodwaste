from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnomalyDetector:
    z_threshold: float = 2.0

    def detect(self, df: pd.DataFrame) -> list[dict[str, object]]:
        if df.empty:
            return []

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date", "waste_kg", "commodity"])
        work["day"] = work["date"].dt.date

        daily = work.groupby("day")["waste_kg"].sum()
        if len(daily) < 3:
            return []

        mean = daily.mean()
        std = daily.std(ddof=0)
        if std == 0 or np.isnan(std):
            return []

        z_scores = (daily - mean) / std
        abnormal_days = daily[z_scores > self.z_threshold]
        if abnormal_days.empty:
            return []

        commodity_daily = (
            work.groupby(["day", "commodity"], as_index=False)["waste_kg"].sum()
            .sort_values(["day", "waste_kg"], ascending=[True, False])
        )

        anomalies: list[dict[str, object]] = []
        for day, total in abnormal_days.sort_values(ascending=False).items():
            day_rows = commodity_daily[commodity_daily["day"] == day]
            top_commodity = day_rows.iloc[0]["commodity"] if not day_rows.empty else "N/A"
            top_amount = float(day_rows.iloc[0]["waste_kg"]) if not day_rows.empty else 0.0
            anomalies.append(
                {
                    "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                    "waste_kg": round(float(total), 2),
                    "affected_commodity": str(top_commodity),
                    "commodity_waste_kg": round(top_amount, 2),
                    "z_score": round(float(z_scores.loc[day]), 2),
                }
            )

        return anomalies
