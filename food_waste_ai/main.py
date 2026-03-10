from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from food_waste_ai.analytics.anomaly_detection import AnomalyDetector
from food_waste_ai.analytics.forecasting import WasteForecaster
from food_waste_ai.analytics.waste_analysis import WasteAnalyzer
from food_waste_ai.config.settings import settings
from food_waste_ai.llm.insight_generator import LLMInsightGenerator
from food_waste_ai.visualization.charts import ChartBuilder


def _ensure_dirs() -> None:
    settings.data_path.parent.mkdir(parents=True, exist_ok=True)
    settings.charts_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)


def _ensure_data() -> None:
    if settings.data_path.exists():
        return
    df = WasteAnalyzer.generate_mock_data(rows=500, seed=42)
    df.to_csv(settings.data_path, index=False)


def _save_outputs(payload: dict[str, Any], report: str) -> None:
    (settings.reports_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (settings.reports_dir / "llm_report.md").write_text(report, encoding="utf-8")


def _launch_dashboard() -> None:
    dashboard_entry = settings.project_root / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_entry)], check=False)


def run_pipeline(launch_dashboard: bool = True) -> None:
    print("FOOD WASTE AI PLATFORM")
    _ensure_dirs()
    _ensure_data()

    analyzer = WasteAnalyzer(settings.data_path)
    summary = analyzer.run_analysis()
    clean_df = analyzer.df.copy()
    print("Data Loaded")
    print("Analytics Completed")

    anomaly_detector = AnomalyDetector(z_threshold=2.0)
    anomalies = anomaly_detector.detect(clean_df)
    print("Anomalies Detected")

    forecaster = WasteForecaster(horizon_days=14)
    forecast = forecaster.forecast(analyzer.daily_series())
    print("Forecast Generated")

    chart_builder = ChartBuilder(settings.charts_dir)
    chart_paths = chart_builder.save_all(clean_df, anomalies, forecast)
    print("Charts Saved")

    llm = LLMInsightGenerator(
        api_key=settings.grok_api_key,
        model=settings.grok_model,
        base_url=settings.grok_base_url,
    )
    report = llm.generate_report(summary, anomalies, forecast)
    print("AI Report Generated")

    payload = {
        "analytics_summary": summary,
        "anomalies": anomalies,
        "forecast": forecast,
        "chart_paths": chart_paths,
    }
    _save_outputs(payload, report)

    if launch_dashboard:
        _launch_dashboard()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Food Waste Intelligence Platform pipeline")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip Streamlit launch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(launch_dashboard=not args.no_dashboard)


if __name__ == "__main__":
    main()
