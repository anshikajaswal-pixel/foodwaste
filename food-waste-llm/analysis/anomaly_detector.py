from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WasteAnomalyDetector:
    z_threshold: float = 2.0

    def detect(self, daily_waste: pd.Series) -> pd.DataFrame:
        if daily_waste.empty:
            return pd.DataFrame(columns=["date", "waste_kg", "z_score", "threshold_kg"])

        mean = daily_waste.mean()
        std = daily_waste.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.DataFrame(columns=["date", "waste_kg", "z_score", "threshold_kg"])

        z_scores = (daily_waste - mean) / std
        threshold = mean + self.z_threshold * std

        anomalies = daily_waste[z_scores > self.z_threshold]
        if anomalies.empty:
            return pd.DataFrame(columns=["date", "waste_kg", "z_score", "threshold_kg"])

        result = pd.DataFrame(
            {
                "date": anomalies.index,
                "waste_kg": anomalies.values,
                "z_score": z_scores.loc[anomalies.index].values,
                "threshold_kg": threshold,
            }
        )

        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        result["waste_kg"] = result["waste_kg"].round(2)
        result["z_score"] = result["z_score"].round(2)
        result["threshold_kg"] = result["threshold_kg"].round(2)
        return result.sort_values("waste_kg", ascending=False).reset_index(drop=True)
