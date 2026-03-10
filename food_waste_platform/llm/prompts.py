from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a senior sustainability consultant for food-service operations. "
    "Provide concise and practical recommendations based on the data."
)


def build_prompt(summary: dict[str, Any], anomalies: list[dict[str, Any]], forecast: list[dict[str, Any]]) -> str:
    payload = {"summary": summary, "anomalies": anomalies, "forecast": forecast}
    return (
        "Generate a Food Waste Intelligence report with sections:\n"
        "1) Executive Summary\n2) Key Waste Drivers\n3) Operational Issues\n4) Sustainability Recommendations\n"
        "Use explicit numbers and provide at least 3 actionable recommendations.\n\n"
        f"Data:\n{json.dumps(payload, indent=2)}"
    )
