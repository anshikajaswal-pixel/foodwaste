from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a senior sustainability consultant specializing in food service operations. "
    "Use the provided data only. Be specific, actionable, and concise."
)


def build_report_prompt(summary: dict[str, Any], anomalies: list[dict[str, Any]], forecast: list[dict[str, Any]]) -> str:
    payload = {
        "summary": summary,
        "anomalies": anomalies,
        "forecast_next_7_days": forecast,
    }

    return (
        "Generate a consulting-style Food Waste Intelligence Report with the sections below:\n"
        "1) Executive Summary\n"
        "2) Key Waste Drivers\n"
        "3) Operational Problems\n"
        "4) Sustainability Recommendations\n\n"
        "Requirements:\n"
        "- Mention quantitative evidence (kg, %, trend direction).\n"
        "- Provide at least 3 specific operational actions tied to commodities/meal services.\n"
        "- Flag anomaly dates and why they matter.\n"
        "- Keep report under 450 words and use professional tone.\n\n"
        f"Data:\n{json.dumps(payload, indent=2)}"
    )
