from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ChartBuilder:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save_all(
        self,
        clean_df: pd.DataFrame,
        anomalies: list[dict[str, object]],
        forecast: list[dict[str, object]],
    ) -> dict[str, str]:
        clean_df = clean_df.copy()
        clean_df["date"] = pd.to_datetime(clean_df["date"])
        clean_df["day"] = clean_df["date"].dt.date

        daily = clean_df.groupby("day", as_index=False)["waste_kg"].sum()
        daily = daily.rename(columns={"day": "date"})
        daily["date"] = pd.to_datetime(daily["date"])

        fig_trend = px.line(daily, x="date", y="waste_kg", title="Waste Trend Over Time", markers=True)

        top_comm = (
            clean_df.groupby("commodity", as_index=False)["waste_kg"].sum().sort_values("waste_kg", ascending=False).head(8)
        )
        fig_top = px.bar(top_comm, x="commodity", y="waste_kg", title="Top Wasted Commodities", text_auto=".2f")

        by_meal = clean_df.groupby("meal_type", as_index=False)["waste_kg"].sum().sort_values("waste_kg", ascending=False)
        fig_meal = px.bar(by_meal, x="meal_type", y="waste_kg", title="Waste by Meal Type", text_auto=".2f")

        fig_anom = px.line(daily, x="date", y="waste_kg", title="Anomaly Detection Plot", markers=True)
        if anomalies:
            adf = pd.DataFrame(anomalies)
            adf["date"] = pd.to_datetime(adf["date"])
            fig_anom.add_scatter(
                x=adf["date"],
                y=adf["waste_kg"],
                mode="markers",
                name="Anomalies",
                marker=dict(color="red", size=11, symbol="diamond"),
            )

        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(x=daily["date"], y=daily["waste_kg"], mode="lines+markers", name="Historical")
        )
        if forecast:
            fdf = pd.DataFrame(forecast)
            fdf["date"] = pd.to_datetime(fdf["date"])
            fig_forecast.add_trace(
                go.Scatter(x=fdf["date"], y=fdf["predicted_waste_kg"], mode="lines+markers", name="Forecast")
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=fdf["date"].tolist() + fdf["date"].tolist()[::-1],
                    y=fdf["upper_kg"].tolist() + fdf["lower_kg"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(0,176,246,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Forecast Interval",
                    hoverinfo="skip",
                )
            )
        fig_forecast.update_layout(title="Forecast Chart")

        figures = {
            "waste_trend": fig_trend,
            "top_wasted_commodities": fig_top,
            "waste_by_meal_type": fig_meal,
            "anomaly_detection": fig_anom,
            "forecast": fig_forecast,
        }

        paths: dict[str, str] = {}
        for name, fig in figures.items():
            out = self.out_dir / f"{name}.html"
            fig.write_html(str(out), include_plotlyjs="cdn")
            paths[name] = str(out)
        return paths
