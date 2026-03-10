from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a senior sustainability consultant for commercial food operations. "
    "Provide practical, data-backed recommendations with direct operational actions."
)


def build_report_prompt(
    analytics_summary: dict[str, Any],
    anomalies: list[dict[str, Any]],
    forecast: list[dict[str, Any]],
) -> str:
    payload = {
        "analytics_summary": analytics_summary,
        "anomalies": anomalies,
        "forecast_next_days": forecast,
    }

    return (
        "Generate a Food Waste Intelligence Report with these sections:\n"
        "1. Executive Summary\n"
        "2. Key Waste Drivers\n"
        "3. Operational Issues\n"
        "4. Sustainability Recommendations\n\n"
        "Rules:\n"
        "- Use exact numbers from the data.\n"
        "- Highlight anomaly dates and affected commodities.\n"
        "- Provide at least three concrete actions (batch size, prep planning, demand forecasting).\n"
        "- Keep concise and professional.\n\n"
        f"Input data:\n{json.dumps(payload, indent=2)}"
    )
