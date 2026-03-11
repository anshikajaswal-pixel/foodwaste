from __future__ import annotations

import time
from urllib.parse import parse_qs, urlencode, urlparse
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["date", "kitchen", "meal_type", "commodity", "waste_kg"]
OPTIONAL_COLUMNS = ["waste_type"]
COLUMN_MAP = {
    "Date": "date",
    "Device Serial No": "kitchen",
    "Food Waste Type": "waste_type",
    "Meal Type": "meal_type",
    "Commodity": "commodity",
    "Weight (KG)": "waste_kg",
}


def _normalize_columns(columns: list[str]) -> list[str]:
    return [str(c).strip().lower().replace(" ", "_") for c in columns]


def _signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "rows:0"
    latest_date = pd.to_datetime(df["date"], errors="coerce").max()
    return f"rows:{len(df)}|latest:{latest_date}"


def _to_live_csv_url(sheet_url: str) -> str:
    parsed = urlparse(sheet_url)
    if "docs.google.com" not in parsed.netloc or "/spreadsheets/d/" not in parsed.path:
        return sheet_url

    if "/export" in parsed.path and "format=csv" in parsed.query:
        return sheet_url

    parts = parsed.path.split("/")
    try:
        sheet_id = parts[parts.index("d") + 1]
    except Exception:
        return sheet_url

    query = parse_qs(parsed.query)
    gid = query.get("gid", ["0"])[0]
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?"
        f"{urlencode({'format': 'csv', 'gid': gid})}"
    )


def fetch_live_sheet(sheet_url: str) -> pd.DataFrame:
    sheet_url = _to_live_csv_url(sheet_url)
    cache_buster = f"_ts={int(time.time())}"
    connector = "&" if "?" in sheet_url else "?"
    live_url = f"{sheet_url}{connector}{cache_buster}"

    try:
        df = pd.read_csv(live_url)
    except Exception as exc:
        raise RuntimeError(f"Google sheet not accessible: {exc}") from exc

    if df is None or df.empty:
        raise ValueError("Google Sheet returned no data.")

    renamed = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if renamed:
        df = df.rename(columns=renamed)

    df.columns = _normalize_columns(df.columns.tolist())

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected_columns = REQUIRED_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in df.columns]
    df = df[selected_columns].copy()
    # Source sheet uses DD-MM-YYYY HH:MM; parse day-first to avoid dropping valid rows.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["waste_kg"] = pd.to_numeric(df["waste_kg"], errors="coerce")

    for col in ["kitchen", "meal_type", "commodity"]:
        df[col] = df[col].astype(str).str.strip()
        if col in {"meal_type", "commodity"}:
            df[col] = df[col].str.title()
        if col == "kitchen":
            df[col] = df[col].str.upper().str.replace(" ", "", regex=False)
            # Normalize common human typo: CFSO* (letter O) -> CFS0* (zero)
            df[col] = df[col].str.replace("CFSO", "CFS0", regex=False)

    df = df.dropna(subset=["date", "kitchen", "meal_type", "commodity", "waste_kg"])
    df = df[df["waste_kg"] >= 0].sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("Malformed data: all rows invalid after cleaning.")

    df.attrs["signature"] = _signature(df)
    return df
