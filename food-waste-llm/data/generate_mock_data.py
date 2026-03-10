from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def generate_mock_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    now = datetime.now()
    start = now - timedelta(days=30)

    devices = ["DEV-01", "DEV-02", "DEV-03"]
    waste_types = ["Production Waste", "Bain Marie Waste", "Plate Waste"]
    meal_types = ["Breakfast", "Lunch", "Dinner"]
    commodities = ["Rice", "Roti", "Dal", "Vegetables", "Chicken", "Salad"]

    base_waste = {
        "Breakfast": (0.3, 3.2),
        "Lunch": (0.8, 6.0),
        "Dinner": (0.6, 5.2),
    }

    records: list[dict[str, object]] = []
    spike_days = {((now - timedelta(days=i)).date()) for i in random.sample(range(1, 30), 3)}

    for sr_no in range(1, rows + 1):
        random_seconds = random.randint(0, int((now - start).total_seconds()))
        timestamp = start + timedelta(seconds=random_seconds)

        meal = random.choice(meal_types)
        low, high = base_waste[meal]
        weight = random.uniform(low, high)

        if timestamp.date() in spike_days and random.random() < 0.35:
            weight *= random.uniform(1.8, 2.8)

        record = {
            "SR NO": sr_no,
            "Date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Device Serial No": random.choice(devices),
            "Food Waste Type": random.choice(waste_types),
            "Meal Type": meal,
            "Weight (KG)": round(weight, 2),
            "Commodity": random.choice(commodities),
        }
        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock food waste data")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent / "waste_data.csv"
    df = generate_mock_data(rows=args.rows, seed=args.seed)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows at {output_path}")


if __name__ == "__main__":
    main()
