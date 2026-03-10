from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from food_waste_ai.analytics.anomaly_detection import AnomalyDetector
from food_waste_ai.analytics.forecasting import WasteForecaster
from food_waste_ai.analytics.waste_analysis import WasteAnalyzer
from food_waste_ai.config.settings import settings
from food_waste_ai.visualization.charts import ChartBuilder


def run_dashboard() -> None:
    st.set_page_config(page_title="Food Waste AI", layout="wide")
    st.title("Food Waste Intelligence Platform")

    analyzer = WasteAnalyzer(settings.data_path)
    summary = analyzer.run_analysis()
    clean_df = analyzer.df.copy()

    anomalies = AnomalyDetector().detect(clean_df)
    forecast = WasteForecaster(horizon_days=14).forecast(analyzer.daily_series())

    charts = ChartBuilder(settings.charts_dir)
    chart_paths = charts.save_all(clean_df, anomalies, forecast)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Waste (KG)", summary["total_waste_kg"])
    c2.metric("Anomaly Days", len(anomalies))
    c3.metric("Forecast Horizon", "14 days")

    st.subheader("Analytics")
    st.json(
        {
            "waste_by_meal_type": summary["waste_by_meal_type"],
            "waste_by_commodity": summary["waste_by_commodity"],
            "waste_by_kitchen": summary["waste_by_kitchen"],
        }
    )

    st.subheader("Charts")
    for _, p in chart_paths.items():
        st.components.v1.html(open(p, "r", encoding="utf-8").read(), height=420, scrolling=True)

    if anomalies:
        st.subheader("Anomaly Alerts")
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True)

    report_file = settings.reports_dir / "llm_report.md"
    st.subheader("AI Sustainability Insights")
    if report_file.exists():
        st.markdown(report_file.read_text(encoding="utf-8"))
    else:
        st.info("AI report not found. Run python main.py first.")

    summary_file = settings.reports_dir / "summary.json"
    if summary_file.exists():
        with st.expander("Raw Summary JSON"):
            st.code(json.dumps(json.loads(summary_file.read_text(encoding="utf-8")), indent=2), language="json")


if __name__ == "__main__":
    run_dashboard()
