from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class WasteVisualizer:
    def __init__(self, charts_dir: str | Path) -> None:
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def waste_by_meal_chart(self, df: pd.DataFrame) -> go.Figure:
        grouped = df.groupby("Meal Type", as_index=False)["Weight (KG)"].sum()
        fig = px.bar(grouped, x="Meal Type", y="Weight (KG)", title="Waste by Meal Type", text_auto=".2f")
        fig.update_layout(template="plotly_white")
        return fig

    def waste_by_food_type_chart(self, df: pd.DataFrame) -> go.Figure:
        grouped = df.groupby("Food Waste Type", as_index=False)["Weight (KG)"].sum()
        fig = px.pie(grouped, names="Food Waste Type", values="Weight (KG)", title="Waste by Food Waste Type")
        fig.update_layout(template="plotly_white")
        return fig

    def daily_trend_chart(self, daily_waste: pd.Series, anomalies: pd.DataFrame | None = None) -> go.Figure:
        trend_df = daily_waste.reset_index()
        trend_df.columns = ["Date", "Waste (KG)"]

        fig = px.line(trend_df, x="Date", y="Waste (KG)", title="Daily Waste Trend", markers=True)
        fig.update_layout(template="plotly_white")

        if anomalies is not None and not anomalies.empty:
            marker_df = anomalies.copy()
            marker_df["date"] = pd.to_datetime(marker_df["date"])
            fig.add_scatter(
                x=marker_df["date"],
                y=marker_df["waste_kg"],
                mode="markers",
                marker=dict(size=10, color="red", symbol="diamond"),
                name="Anomalies",
            )

        return fig

    def top_wasted_commodities_chart(self, df: pd.DataFrame, top_n: int = 5) -> go.Figure:
        grouped = (
            df.groupby("Commodity", as_index=False)["Weight (KG)"]
            .sum()
            .sort_values("Weight (KG)", ascending=False)
            .head(top_n)
        )
        fig = px.bar(
            grouped,
            x="Commodity",
            y="Weight (KG)",
            title=f"Top {top_n} Wasted Commodities",
            text_auto=".2f",
        )
        fig.update_layout(template="plotly_white")
        return fig

    def forecast_chart(self, daily_waste: pd.Series, forecast_df: pd.DataFrame) -> go.Figure:
        historical = daily_waste.reset_index()
        historical.columns = ["Date", "Waste (KG)"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=historical["Date"],
                y=historical["Waste (KG)"],
                mode="lines+markers",
                name="Historical",
            )
        )

        if not forecast_df.empty:
            fdf = forecast_df.copy()
            fdf["date"] = pd.to_datetime(fdf["date"])
            fig.add_trace(
                go.Scatter(
                    x=fdf["date"],
                    y=fdf["predicted_waste_kg"],
                    mode="lines+markers",
                    name="Forecast",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fdf["date"].tolist() + fdf["date"].tolist()[::-1],
                    y=fdf["upper_kg"].tolist() + fdf["lower_kg"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(0, 100, 80, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Forecast Interval",
                )
            )

        fig.update_layout(title="7-Day Waste Forecast", template="plotly_white")
        return fig

    def export_all(
        self,
        df: pd.DataFrame,
        daily_waste: pd.Series,
        anomalies: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ) -> dict[str, str]:
        charts = {
            "waste_by_meal": self.waste_by_meal_chart(df),
            "waste_by_food_type": self.waste_by_food_type_chart(df),
            "daily_trend": self.daily_trend_chart(daily_waste, anomalies),
            "top_commodities": self.top_wasted_commodities_chart(df),
            "forecast": self.forecast_chart(daily_waste, forecast_df),
        }

        paths: dict[str, str] = {}
        for name, fig in charts.items():
            chart_path = self.charts_dir / f"{name}.html"
            fig.write_html(str(chart_path), include_plotlyjs="cdn")
            paths[name] = str(chart_path)

        return paths
