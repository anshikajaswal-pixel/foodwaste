from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "SR NO",
    "Date",
    "Device Serial No",
    "Food Waste Type",
    "Meal Type",
    "Weight (KG)",
    "Commodity",
]


@dataclass
class WasteSummary:
    total_waste_kg: float
    waste_by_meal_type: dict[str, float]
    waste_by_food_type: dict[str, float]
    top_wasted_commodities: dict[str, float]
    waste_by_device: dict[str, float]
    waste_by_meal_service: list[dict[str, Any]]
    daily_trend: list[dict[str, Any]]
    operational_efficiency_metrics: dict[str, float]


class WasteDataProcessor:
    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.df = self._load_and_clean_data()

    def _load_and_clean_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[REQUIRED_COLUMNS].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Weight (KG)"] = pd.to_numeric(df["Weight (KG)"], errors="coerce")
        df["Device Serial No"] = df["Device Serial No"].astype(str).str.strip()
        df["Food Waste Type"] = df["Food Waste Type"].astype(str).str.strip()
        df["Meal Type"] = df["Meal Type"].astype(str).str.strip()
        df["Commodity"] = df["Commodity"].astype(str).str.strip()

        df = df.dropna(subset=["Date", "Weight (KG)"])
        df = df[df["Weight (KG)"] >= 0]
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def get_clean_data(self) -> pd.DataFrame:
        return self.df.copy()

    def get_daily_waste_series(self) -> pd.Series:
        daily = self.df.groupby(self.df["Date"].dt.date)["Weight (KG)"].sum()
        daily.index = pd.to_datetime(daily.index)
        return daily.sort_index()

    def _operational_efficiency_metrics(self) -> dict[str, float]:
        total_waste = float(self.df["Weight (KG)"].sum())
        service_count = max(len(self.df), 1)

        plate_waste = float(
            self.df.loc[self.df["Food Waste Type"].str.lower() == "plate waste", "Weight (KG)"].sum()
        )
        prep_waste = float(
            self.df.loc[
                self.df["Food Waste Type"].str.lower().isin(["production waste", "bain marie waste"]),
                "Weight (KG)",
            ].sum()
        )

        top3_share = (
            self.df.groupby("Commodity")["Weight (KG)"].sum().sort_values(ascending=False).head(3).sum()
        )

        device_totals = self.df.groupby("Device Serial No")["Weight (KG)"].sum()
        device_cv = float(device_totals.std() / device_totals.mean()) if device_totals.mean() else 0.0

        return {
            "avg_waste_per_service_kg": round(total_waste / service_count, 3),
            "plate_waste_ratio_pct": round((plate_waste / total_waste) * 100, 2) if total_waste else 0.0,
            "prep_to_plate_ratio": round((prep_waste / plate_waste), 2) if plate_waste else 0.0,
            "top3_commodity_concentration_pct": round((float(top3_share) / total_waste) * 100, 2)
            if total_waste
            else 0.0,
            "device_waste_cv": round(device_cv, 3),
        }

    def build_summary(self) -> dict[str, Any]:
        total_waste = float(self.df["Weight (KG)"].sum())

        by_meal = self.df.groupby("Meal Type")["Weight (KG)"].sum().sort_values(ascending=False)
        by_food_type = self.df.groupby("Food Waste Type")["Weight (KG)"].sum().sort_values(ascending=False)
        by_commodity = self.df.groupby("Commodity")["Weight (KG)"].sum().sort_values(ascending=False)
        by_device = self.df.groupby("Device Serial No")["Weight (KG)"].sum().sort_values(ascending=False)

        meal_service_df = (
            self.df.assign(service_date=self.df["Date"].dt.strftime("%Y-%m-%d"))
            .groupby(["service_date", "Meal Type"], as_index=False)["Weight (KG)"]
            .sum()
            .sort_values(["service_date", "Meal Type"])
        )

        daily_trend = self.get_daily_waste_series().reset_index()
        daily_trend.columns = ["date", "waste_kg"]
        daily_trend["date"] = daily_trend["date"].dt.strftime("%Y-%m-%d")

        summary = WasteSummary(
            total_waste_kg=round(total_waste, 2),
            waste_by_meal_type={k: round(float(v), 2) for k, v in by_meal.items()},
            waste_by_food_type={k: round(float(v), 2) for k, v in by_food_type.items()},
            top_wasted_commodities={
                k: round(float(v), 2) for k, v in by_commodity.head(5).items()
            },
            waste_by_device={k: round(float(v), 2) for k, v in by_device.items()},
            waste_by_meal_service=meal_service_df.rename(
                columns={"service_date": "date", "Weight (KG)": "waste_kg"}
            ).to_dict(orient="records"),
            daily_trend=daily_trend.to_dict(orient="records"),
            operational_efficiency_metrics=self._operational_efficiency_metrics(),
        )

        return summary.__dict__
