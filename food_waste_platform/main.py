from __future__ import annotations

import argparse
import json
import subprocess
import sys

from food_waste_platform.analysis.anomaly_detector import detect_anomalies
from food_waste_platform.analysis.forecasting import forecast_waste
from food_waste_platform.analysis.summary import compute_summary
from food_waste_platform.config.settings import settings
from food_waste_platform.data.live_sheet_loader import fetch_live_sheet
from food_waste_platform.llm.insight_generator import generate_insights


def run_pipeline(launch_dashboard: bool = True) -> None:
    print("FOOD WASTE AI PLATFORM")

    try:
        df = fetch_live_sheet(settings.sheet_url)
    except Exception as exc:
        print(f"Data loading failed: {exc}")
        return

    print("Data Loaded")

    try:
        summary = compute_summary(df)
        print("Analytics Completed")

        anomalies = detect_anomalies(df)
        print("Anomalies Detected")

        forecast = forecast_waste(df, horizon_days=14)
        print("Forecast Generated")

        report = generate_insights(
            summary=summary,
            anomalies=anomalies,
            forecast=forecast,
            api_key=settings.grok_api_key,
            model=settings.grok_model,
            base_url=settings.grok_base_url,
        )
        print("AI Report Generated")
    except Exception as exc:
        print(f"Pipeline computation failed: {exc}")
        return

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "anomalies": anomalies,
        "forecast": forecast,
        "rows": len(df),
    }
    (settings.reports_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (settings.reports_dir / "llm_report.md").write_text(report, encoding="utf-8")

    print("Charts Saved")

    if launch_dashboard:
        dashboard_file = settings.base_dir / "dashboard" / "dashboard_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_file)], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Food Waste live pipeline")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching streamlit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(launch_dashboard=not args.no_dashboard)


if __name__ == "__main__":
    main()
