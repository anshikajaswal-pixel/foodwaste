from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Any

import pandas as pd
import requests

WEBSITE_CONTEXT = (
    "This website provides a live food-waste monitoring dashboard for kitchens. "
    "It highlights waste trends, top commodities, meal-type patterns, anomalies, and "
    "sustainability insights to help operators reduce waste."
)

COLUMN_MAP = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "device_serial_no": "kitchen",
    "kitchen": "kitchen",
    "meal_type": "meal_type",
    "commodity": "commodity",
    "food_waste_type": "waste_type",
    "waste_type": "waste_type",
    "weight_(kg)": "waste_kg",
    "weight_kg": "waste_kg",
    "waste_kg": "waste_kg",
}


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.copy()
    df.columns = normalized
    renamed = {c: COLUMN_MAP.get(c, c) for c in df.columns}
    return df.rename(columns=renamed)


def _build_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df))}

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        if dates.notna().any():
            summary["date_min"] = str(dates.min())
            summary["date_max"] = str(dates.max())
            if "waste_kg" in df.columns:
                by_day = (
                    df.assign(_date=dates.dt.date)
                    .groupby("_date")["waste_kg"]
                    .sum()
                )
                if not by_day.empty:
                    max_date = by_day.idxmax()
                    min_date = by_day.idxmin()
                    summary["top_waste_day"] = {
                        "date": str(max_date),
                        "waste_kg": float(by_day.loc[max_date]),
                    }
                    summary["least_waste_day"] = {
                        "date": str(min_date),
                        "waste_kg": float(by_day.loc[min_date]),
                    }

    if "waste_kg" in df.columns:
        summary["total_waste_kg"] = float(pd.to_numeric(df["waste_kg"], errors="coerce").sum())

    if "waste_type" in df.columns and "waste_kg" in df.columns:
        by_type = (
            df.groupby("waste_type")["waste_kg"].sum().sort_values(ascending=False)
        )
        summary["waste_by_type"] = {
            str(k): float(v) for k, v in by_type.items()
        }
        production = by_type.get("Production Waste")
        if production is not None:
            summary["production_waste_kg"] = float(production)

    if "kitchen" in df.columns and "waste_kg" in df.columns:
        by_kitchen = (
            df.groupby("kitchen")["waste_kg"].sum().sort_values(ascending=False)
        )
        if not by_kitchen.empty:
            summary["top_kitchen"] = {
                "kitchen": str(by_kitchen.index[0]),
                "waste_kg": float(by_kitchen.iloc[0]),
            }

    if "commodity" in df.columns and "waste_kg" in df.columns:
        by_commodity = (
            df.groupby("commodity")["waste_kg"].sum().sort_values(ascending=False)
        )
        if not by_commodity.empty:
            summary["top_commodity"] = {
                "commodity": str(by_commodity.index[0]),
                "waste_kg": float(by_commodity.iloc[0]),
            }

    if "meal_type" in df.columns and "waste_kg" in df.columns:
        by_meal = (
            df.groupby("meal_type")["waste_kg"].sum().sort_values(ascending=False)
        )
        if not by_meal.empty:
            summary["top_meal_type"] = {
                "meal_type": str(by_meal.index[0]),
                "waste_kg": float(by_meal.iloc[0]),
            }

    return summary


def _read_published_sheet_csv(sheet_url: str, timeout_s: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    response = requests.get(sheet_url, timeout=timeout_s)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    if df.empty:
        raise ValueError("Published Google Sheet returned no data.")

    df = _normalize_columns(df)
    summary = _build_summary(df)
    headers = [str(c) for c in df.columns]
    records = df.fillna("").to_dict(orient="records")
    full_data = os.getenv("GEMINI_FULL_DATA", "0").strip() == "1"
    rows_payload = records if full_data else records[:20]

    payload = {
        "sheet_title": "Published CSV",
        "summary": summary,
        "headers": headers,
        "rows_sample": rows_payload,
        "rows_note": "rows_sample contains ALL rows because GEMINI_FULL_DATA=1"
        if full_data
        else "rows_sample is a small sample; summary is computed from full dataset",
    }
    return df, payload


def _answer_local(question: str, df: pd.DataFrame, summary: dict[str, Any]) -> str | None:
    q = question.lower().strip()
    q_norm = re.sub(r"(?<=\d),(?=\d)", ".", q)

    def _extract_query_date(text: str) -> pd.Timestamp | None:
        text = re.sub(r"\d+(?:\.\d+)?\s*kg(?:s)?\b", " ", text)
        text = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", text)
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
        if m:
            day, month, year = m.groups()
            year_i = int(year)
            if year_i < 100:
                year_i += 2000
            return pd.Timestamp(year=year_i, month=int(month), day=int(day))
        m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
        if m:
            year, month, day = m.groups()
            return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", text):
            dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
            if pd.notna(dt):
                return pd.Timestamp(dt.date())
        try:
            from dateutil import parser as dateparser

            dt = dateparser.parse(text, dayfirst=True, fuzzy=True)
            if dt:
                return pd.Timestamp(dt.date())
        except Exception:
            return None
        return None

    query_date = _extract_query_date(q_norm)
    if query_date is not None:
        if "waste_kg" not in df.columns or "date" not in df.columns:
            return None
        work = df.copy()
        work["waste_kg"] = pd.to_numeric(work["waste_kg"], errors="coerce")
        work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
        work = work.dropna(subset=["waste_kg", "date"])
        work["day"] = work["date"].dt.date
        day = query_date.date()
        if "production" in q_norm and "waste" in q_norm and "waste_type" in work.columns:
            prod = work[work["waste_type"].astype(str).str.contains("Production", case=False, na=False)]
            total = prod.loc[prod["day"] == day, "waste_kg"].sum()
            if total <= 0:
                return f"No production waste recorded on {day.isoformat()}."
            return f"Production waste on {day.isoformat()} was {total:.2f} kg."
        total = work.loc[work["day"] == day, "waste_kg"].sum()
        if total <= 0:
            return f"No waste recorded on {day.isoformat()}."
        return f"Total waste on {day.isoformat()} was {total:.2f} kg."
    if "weight" in q or "kg" in q:
        target = None
        kg_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*kg\b", q_norm))
        if not kg_matches:
            kg_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*kgs\b", q_norm))
        if kg_matches:
            target = float(kg_matches[-1].group(1))
        if target is None:
            return None
        if "waste_kg" not in df.columns or "date" not in df.columns:
            return None

        work = df.copy()
        work["waste_kg"] = pd.to_numeric(work["waste_kg"], errors="coerce")
        work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
        work = work.dropna(subset=["waste_kg", "date"])
        hits = work[work["waste_kg"].sub(target).abs() < 1e-6]
        if hits.empty:
            return f"No rows found with weight exactly {target:.2f} kg."
        days = sorted({d.date().isoformat() for d in hits["date"]})
        count = len(days)
        if count <= 20:
            return f"Exact weight {target:.2f} kg occurred on {count} days: {', '.join(days)}."
        return f"Exact weight {target:.2f} kg occurred on {count} days. First 20: {', '.join(days[:20])}."
    if "production waste" in q and "amount" in q:
        value = summary.get("production_waste_kg")
        if value is not None:
            return f"The total amount of production waste is {float(value):.2f} kg."
    if "least waste" in q or "minimum waste" in q:
        least = summary.get("least_waste_day")
        if least:
            return f"The least waste was on {least['date']} with {least['waste_kg']:.2f} kg."
    if "most waste" in q or "highest waste" in q:
        top = summary.get("top_waste_day")
        if top:
            return f"The most waste was on {top['date']} with {top['waste_kg']:.2f} kg."
    return None


def format_sheet_payload(payload: dict[str, Any], max_chars: int = 6000) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...truncated"
    return text


def answer_question_with_sheet(question: str) -> str:
    from google import genai

    api_key = _get_required_env("GEMINI_API_KEY")
    sheet_url = _get_required_env("PUBLISHED_SHEET_CSV_URL")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    df, payload = _read_published_sheet_csv(sheet_url)
    sheet_context = format_sheet_payload(payload)
    local = _answer_local(question, df, payload.get("summary", {}))
    if local:
        return local

    client = genai.Client(api_key=api_key)
    system_instruction = (
        "You are a helpful assistant for this website. "
        "Use the provided database context to answer the user's questions. "
        "If the answer is not in the context, use your general knowledge but keep it relevant."
    )

    contents = [
        {
            "role": "user",
            "parts": [{"text": f"SYSTEM INSTRUCTION:\n{system_instruction}"}],
        },
        {
            "role": "user",
            "parts": [{"text": f"Website context:\n{WEBSITE_CONTEXT}"}],
        },
        {
            "role": "user",
            "parts": [{"text": f"Google Sheet summary and sample rows (summary is full dataset):\n{sheet_context}"}],
        },
        {
            "role": "user",
            "parts": [{"text": f"User question:\n{question}"}],
        },
    ]

    max_attempts = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    backoff_s = float(os.getenv("GEMINI_RETRY_BACKOFF_S", "1.2"))
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return "No response generated."
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(backoff_s * attempt)
                continue
            break

    fallback = _answer_local(question, df, payload.get("summary", {}))
    if fallback:
        return fallback
    return f"Assistant error: {last_exc}"
