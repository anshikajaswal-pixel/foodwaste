from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from analysis.anomaly_detector import WasteAnomalyDetector
from analysis.data_processor import WasteDataProcessor
from analysis.forecasting import WasteForecaster
from config.settings import settings
from dashboard.visualizer import WasteVisualizer
from data.generate_mock_data import generate_mock_data
from llm.insight_generator import LLMInsightGenerator


def _ensure_data() -> None:
    if settings.data_file.exists():
        return

    settings.data_file.parent.mkdir(parents=True, exist_ok=True)
    df = generate_mock_data(rows=500, seed=42)
    df.to_csv(settings.data_file, index=False)


def _to_records(df_like: Any) -> list[dict[str, Any]]:
    if hasattr(df_like, "to_dict"):
        return df_like.to_dict(orient="records")
    return []


def _save_artifacts(summary: dict[str, Any], anomalies: list[dict[str, Any]], forecast: list[dict[str, Any]], report: str) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "summary": summary,
        "anomalies": anomalies,
        "forecast_next_7_days": forecast,
    }

    (settings.reports_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    (settings.reports_dir / "llm_report.md").write_text(report, encoding="utf-8")


def _print_console_report(summary: dict[str, Any], anomalies: list[dict[str, Any]], chart_paths: dict[str, str], report: str) -> None:
    print("\n" + "=" * 80)
    print("FOOD WASTE INTELLIGENCE PLATFORM")
    print("=" * 80)
    print(f"Total Waste (KG): {summary['total_waste_kg']}")
    print(f"Top Wasted Commodities: {summary['top_wasted_commodities']}")
    print(f"Anomaly Days: {len(anomalies)}")
    print("\nGenerated Charts:")
    for name, path in chart_paths.items():
        print(f"- {name}: {path}")
    print("\nAI Sustainability Report:\n")
    print(report)
    print("=" * 80 + "\n")


def _launch_dashboard() -> None:
    app_path = settings.base_dir / "dashboard" / "dashboard_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd, check=False, cwd=settings.base_dir)


def run_pipeline(launch_dashboard: bool = True) -> None:
    _ensure_data()

    processor = WasteDataProcessor(settings.data_file)
    clean_df = processor.get_clean_data()
    summary = processor.build_summary()
    daily_waste = processor.get_daily_waste_series()

    anomaly_detector = WasteAnomalyDetector(z_threshold=2.0)
    anomalies_df = anomaly_detector.detect(daily_waste)

    forecaster = WasteForecaster(horizon_days=7)
    forecast_df = forecaster.forecast(daily_waste)

    visualizer = WasteVisualizer(settings.charts_dir)
    chart_paths = visualizer.export_all(clean_df, daily_waste, anomalies_df, forecast_df)

    anomalies = _to_records(anomalies_df)
    forecast = _to_records(forecast_df)

    llm = LLMInsightGenerator(api_key=settings.grok_api_key, model=settings.grok_model)
    report = llm.generate_report(summary_data=summary, anomalies=anomalies, forecast=forecast)

    _save_artifacts(summary, anomalies, forecast, report)
    _print_console_report(summary, anomalies, chart_paths, report)

    if launch_dashboard:
        _launch_dashboard()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Food Waste Intelligence Platform")
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run pipeline without launching Streamlit dashboard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(launch_dashboard=not args.no_dashboard)


if __name__ == "__main__":
    main()
