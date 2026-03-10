from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["date", "kitchen", "meal_type", "commodity", "waste_kg"]
ALTERNATE_MAP = {
    "Date": "date",
    "Meal Type": "meal_type",
    "Commodity": "commodity",
    "Weight (KG)": "waste_kg",
    "Device Serial No": "kitchen",
}


@dataclass
class AnalysisResult:
    total_waste_kg: float
    waste_by_meal_type: dict[str, float]
    waste_by_commodity: dict[str, float]
    waste_by_kitchen: dict[str, float]
    daily_waste_trend: list[dict[str, Any]]


class WasteAnalyzer:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.df = self._load_and_clean()

    @staticmethod
    def generate_mock_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        end = pd.Timestamp.now().floor("D")
        start = end - pd.Timedelta(days=59)

        dates = start + pd.to_timedelta(rng.integers(0, 60 * 24, size=rows), unit="h")
        kitchens = rng.choice(["Kitchen-A", "Kitchen-B", "Kitchen-C"], size=rows, p=[0.4, 0.35, 0.25])
        meals = rng.choice(["Breakfast", "Lunch", "Dinner"], size=rows, p=[0.25, 0.45, 0.30])
        commodities = rng.choice(
            ["Rice", "Roti", "Dal", "Vegetables", "Chicken", "Salad"],
            size=rows,
            p=[0.2, 0.16, 0.17, 0.2, 0.15, 0.12],
        )

        weights = []
        for meal in meals:
            if meal == "Breakfast":
                weights.append(rng.uniform(0.2, 2.5))
            elif meal == "Lunch":
                weights.append(rng.uniform(0.8, 5.5))
            else:
                weights.append(rng.uniform(0.6, 4.3))

        data = pd.DataFrame(
            {
                "date": pd.to_datetime(dates).astype(str),
                "kitchen": kitchens,
                "meal_type": meals,
                "commodity": commodities,
                "waste_kg": np.round(weights, 2),
            }
        )

        spike_idx = rng.choice(data.index, size=max(6, rows // 25), replace=False)
        data.loc[spike_idx, "waste_kg"] = (data.loc[spike_idx, "waste_kg"] * rng.uniform(1.8, 3.0)).round(2)
        return data

    def ensure_data_exists(self) -> None:
        if self.csv_path.exists():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.generate_mock_data(rows=500).to_csv(self.csv_path, index=False)

    def _load_and_clean(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV data file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Allow legacy schema and map to the expected schema.
        if not set(EXPECTED_COLUMNS).issubset(df.columns):
            rename_candidates = {k: v for k, v in ALTERNATE_MAP.items() if k in df.columns}
            if rename_candidates:
                df = df.rename(columns=rename_candidates)

        if not set(EXPECTED_COLUMNS).issubset(df.columns):
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            raise ValueError(f"Invalid schema. Missing columns: {missing}")

        df = df[EXPECTED_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["waste_kg"] = pd.to_numeric(df["waste_kg"], errors="coerce")

        for col in ["kitchen", "meal_type", "commodity"]:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan})

        df = df.dropna(subset=["date", "waste_kg", "kitchen", "meal_type", "commodity"])
        df = df[df["waste_kg"] >= 0]
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def daily_series(self) -> pd.Series:
        daily = self.df.groupby(self.df["date"].dt.date)["waste_kg"].sum().sort_index()
        daily.index = pd.to_datetime(daily.index)
        return daily

    def by_day_commodity(self) -> pd.DataFrame:
        out = (
            self.df.groupby([self.df["date"].dt.date, "commodity"], as_index=False)["waste_kg"].sum()
            .rename(columns={"date": "day"})
        )
        out["date"] = pd.to_datetime(out["date"])
        return out

    def run_analysis(self) -> dict[str, Any]:
        by_meal = self.df.groupby("meal_type")["waste_kg"].sum().sort_values(ascending=False)
        by_commodity = self.df.groupby("commodity")["waste_kg"].sum().sort_values(ascending=False)
        by_kitchen = self.df.groupby("kitchen")["waste_kg"].sum().sort_values(ascending=False)

        daily = self.daily_series().reset_index()
        daily.columns = ["date", "waste_kg"]
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

        result = AnalysisResult(
            total_waste_kg=round(float(self.df["waste_kg"].sum()), 2),
            waste_by_meal_type={k: round(float(v), 2) for k, v in by_meal.items()},
            waste_by_commodity={k: round(float(v), 2) for k, v in by_commodity.items()},
            waste_by_kitchen={k: round(float(v), 2) for k, v in by_kitchen.items()},
            daily_waste_trend=daily.to_dict(orient="records"),
        )
        return result.__dict__
