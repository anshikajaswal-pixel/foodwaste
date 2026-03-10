from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from food_waste_ai.llm.prompts import SYSTEM_PROMPT, build_report_prompt


@dataclass
class LLMInsightGenerator:
    api_key: str
    model: str = "grok-3-fast"
    base_url: str = "https://api.x.ai/v1"

    def generate_report(
        self,
        analytics_summary: dict[str, Any],
        anomalies: list[dict[str, Any]],
        forecast: list[dict[str, Any]],
    ) -> str:
        if not self.api_key:
            return self._fallback_report(
                analytics_summary, anomalies, "Missing GROK_API_KEY in environment; using fallback report."
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        prompt = build_report_prompt(analytics_summary, anomalies, forecast)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            return self._fallback_report(
                analytics_summary, anomalies, f"Grok API failure ({type(exc).__name__}); using fallback report."
            )
        except Exception as exc:  # pragma: no cover
            return self._fallback_report(
                analytics_summary, anomalies, f"Unexpected LLM error ({type(exc).__name__}); using fallback report."
            )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            return self._fallback_report(
                analytics_summary, anomalies, "Grok API returned empty content; using fallback report."
            )
        return content

    def _fallback_report(
        self,
        analytics_summary: dict[str, Any],
        anomalies: list[dict[str, Any]],
        reason: str,
    ) -> str:
        total = analytics_summary.get("total_waste_kg", 0)
        by_meal = analytics_summary.get("waste_by_meal_type", {})
        top_items = analytics_summary.get("waste_by_commodity", {})

        top3 = list(top_items.items())[:3]
        top3_text = ", ".join([f"{k} ({v} kg)" for k, v in top3]) if top3 else "N/A"
        meal_text = ", ".join([f"{k}: {v} kg" for k, v in by_meal.items()]) if by_meal else "N/A"
        anomaly_text = (
            ", ".join([f"{a['date']} - {a['affected_commodity']} ({a['waste_kg']} kg)" for a in anomalies[:4]])
            if anomalies
            else "No significant anomalies detected"
        )

        return (
            "## Executive Summary\n"
            f"Total observed waste is **{total} kg**. Major waste commodities are {top3_text}. "
            f"Meal-wise waste distribution: {meal_text}.\n\n"
            "## Key Waste Drivers\n"
            "Concentration in a small set of commodities indicates overproduction and demand mismatch.\n\n"
            "## Operational Issues\n"
            f"Anomaly review: {anomaly_text}. These spikes indicate inconsistent prep controls.\n\n"
            "## Sustainability Recommendations\n"
            "1. Reduce batch size for the highest-waste commodities during peak meal windows.\n"
            "2. Shift to rolling prep with refill triggers instead of full-batch pre-cook.\n"
            "3. Add kitchen-level daily variance checks and corrective action logging.\n\n"
            f"_Fallback mode_: {reason}"
        )
