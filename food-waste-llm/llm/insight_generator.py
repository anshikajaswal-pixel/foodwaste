from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from llm.prompts import SYSTEM_PROMPT, build_report_prompt


@dataclass
class LLMInsightGenerator:
    api_key: str
    model: str = "grok-3-fast"
    base_url: str = "https://api.x.ai/v1"

    def generate_report(
        self,
        summary_data: dict[str, Any],
        anomalies: list[dict[str, Any]],
        forecast: list[dict[str, Any]],
    ) -> str:
        if not self.api_key:
            return self._fallback_report(
                summary_data,
                anomalies,
                reason="Missing GROK_API_KEY. Set it in .env before running the pipeline.",
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        prompt = build_report_prompt(summary_data, anomalies, forecast)

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
                summary_data,
                anomalies,
                reason=f"Grok API request failed ({type(exc).__name__}). Using deterministic fallback.",
            )
        except Exception as exc:
            return self._fallback_report(
                summary_data,
                anomalies,
                reason=f"Unexpected Grok error ({type(exc).__name__}). Using deterministic fallback.",
            )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            return self._fallback_report(
                summary_data,
                anomalies,
                reason="Grok API returned an empty response. Using deterministic fallback.",
            )
        return content

    def _fallback_report(
        self,
        summary_data: dict[str, Any],
        anomalies: list[dict[str, Any]],
        reason: str,
    ) -> str:
        top_items = summary_data.get("top_wasted_commodities", {})
        meal_breakdown = summary_data.get("waste_by_meal_type", {})
        total = summary_data.get("total_waste_kg", 0)

        top_items_text = ", ".join([f"{k} ({v} kg)" for k, v in list(top_items.items())[:3]]) or "N/A"
        meal_text = ", ".join([f"{k}: {v} kg" for k, v in meal_breakdown.items()]) or "N/A"
        anomaly_text = ", ".join([f"{a['date']} ({a['waste_kg']} kg)" for a in anomalies[:5]]) or "No significant anomalies"

        return (
            "Executive Summary\n"
            f"Total measured waste is {total} kg. Primary contributors are {top_items_text}. "
            f"Meal-service distribution indicates: {meal_text}.\n\n"
            "Key Waste Drivers\n"
            f"High concentration in top commodities suggests overproduction and mismatch with demand.\n\n"
            "Operational Problems\n"
            f"Detected spike days: {anomaly_text}. These indicate inconsistent batch planning and service-level forecasting gaps.\n\n"
            "Sustainability Recommendations\n"
            "1. Reduce batch size for the highest-waste commodity during peak meal services.\n"
            "2. Introduce meal-wise prep caps with mid-service replenishment triggers.\n"
            "3. Review device-level outliers weekly and enforce corrective action logs.\n\n"
            f"Note: {reason}"
        )
