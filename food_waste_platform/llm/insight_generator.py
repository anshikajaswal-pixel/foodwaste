from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from food_waste_platform.llm.prompts import SYSTEM_PROMPT, build_prompt


def generate_insights(
    summary: dict[str, Any],
    anomalies: list[dict[str, Any]],
    forecast: list[dict[str, Any]],
    api_key: str,
    model: str,
    base_url: str = "https://api.x.ai/v1",
) -> str:
    if not api_key:
        return _fallback("Missing GROK_API_KEY.", summary, anomalies)

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = build_prompt(summary, anomalies, forecast)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return _fallback("Empty Grok response.", summary, anomalies)
        return content
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        return _fallback(f"Grok API error: {type(exc).__name__}", summary, anomalies)
    except Exception as exc:  # pragma: no cover
        return _fallback(f"Unexpected LLM error: {type(exc).__name__}", summary, anomalies)


def _fallback(reason: str, summary: dict[str, Any], anomalies: list[dict[str, Any]]) -> str:
    top = list(summary.get("waste_by_commodity", {}).items())[:3]
    top_txt = ", ".join([f"{k} ({v} kg)" for k, v in top]) if top else "N/A"
    anomaly_txt = ", ".join([f"{a['date']} ({a['waste_kg']} kg)" for a in anomalies[:4]]) or "None"
    return (
        "## Executive Summary\n"
        f"Total waste: {summary.get('total_waste_kg', 0)} kg. Top commodities: {top_txt}.\n\n"
        "## Key Waste Drivers\nOverproduction and demand mismatch in high-loss items.\n\n"
        f"## Operational Issues\nAnomaly days: {anomaly_txt}.\n\n"
        "## Sustainability Recommendations\n"
        "1. Reduce batch sizes for top-waste commodities.\n"
        "2. Use rolling prep and refill thresholds.\n"
        "3. Track kitchen variance daily and enforce corrective actions.\n\n"
        f"Fallback reason: {reason}"
    )
