from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.anomaly_detector import WasteAnomalyDetector
from analysis.data_processor import WasteDataProcessor
from analysis.forecasting import WasteForecaster
from config.settings import settings
from dashboard.visualizer import WasteVisualizer


st.set_page_config(page_title="Food Waste Intelligence", layout="wide")
st.title("Food Waste Intelligence Platform")


@st.cache_data
def load_pipeline_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    processor = WasteDataProcessor(settings.data_file)
    summary = processor.build_summary()
    daily = processor.get_daily_waste_series()

    anomalies = WasteAnomalyDetector(z_threshold=2.0).detect(daily)
    forecast = WasteForecaster(horizon_days=7).forecast(daily)

    return summary, anomalies, forecast


summary, anomalies_df, forecast_df = load_pipeline_data()
processor = WasteDataProcessor(settings.data_file)
clean_df = processor.get_clean_data()
daily_series = processor.get_daily_waste_series()
visualizer = WasteVisualizer(settings.charts_dir)

col1, col2, col3 = st.columns(3)
col1.metric("Total Waste (KG)", f"{summary['total_waste_kg']}")
col2.metric("Anomaly Days", f"{len(anomalies_df)}")
col3.metric("Top Commodity", next(iter(summary["top_wasted_commodities"].keys()), "N/A"))

fig1 = visualizer.waste_by_meal_chart(clean_df)
fig2 = visualizer.waste_by_food_type_chart(clean_df)
fig3 = visualizer.top_wasted_commodities_chart(clean_df)
fig4 = visualizer.daily_trend_chart(daily_series, anomalies_df)
fig5 = visualizer.forecast_chart(daily_series, forecast_df)

left, right = st.columns(2)
with left:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)
with right:
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Operational Efficiency Metrics")
st.json(summary["operational_efficiency_metrics"])

report_path = settings.reports_dir / "llm_report.md"
if report_path.exists():
    st.subheader("AI Sustainability Report")
    st.markdown(report_path.read_text(encoding="utf-8"))

summary_path = settings.reports_dir / "summary.json"
if summary_path.exists():
    with st.expander("Raw Summary JSON"):
        st.code(summary_path.read_text(encoding="utf-8"), language="json")
else:
    st.info("Run `python main.py --no-dashboard` once to generate persisted summary/report artifacts.")
