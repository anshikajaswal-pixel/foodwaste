from __future__ import annotations

import pandas as pd


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> list[dict[str, object]]:
    work = df.copy()
    work["day"] = work["date"].dt.date

    daily = work.groupby("day")["waste_kg"].sum()
    if len(daily) < 3:
        return []

    mean = daily.mean()
    std = daily.std(ddof=0)
    if std == 0 or pd.isna(std):
        return []

    z_scores = (daily - mean) / std
    spikes = daily[z_scores > z_threshold].sort_values(ascending=False)

    by_day_commodity = (
        work.groupby(["day", "commodity"], as_index=False)["waste_kg"].sum().sort_values(
            ["day", "waste_kg"], ascending=[True, False]
        )
    )

    results: list[dict[str, object]] = []
    for day, waste in spikes.items():
        day_rows = by_day_commodity[by_day_commodity["day"] == day]
        top = day_rows.iloc[0] if not day_rows.empty else None
        results.append(
            {
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "waste_kg": round(float(waste), 2),
                "affected_commodity": str(top["commodity"]) if top is not None else "N/A",
                "commodity_waste_kg": round(float(top["waste_kg"]), 2) if top is not None else 0.0,
                "z_score": round(float(z_scores.loc[day]), 2),
            }
        )
    return results
