from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    outputs_dir: Path
    reports_dir: Path
    sheet_url: str
    grok_api_key: str
    grok_model: str
    grok_base_url: str
    dashboard_refresh_ms: int


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    outputs_dir = base_dir / "outputs"
    reports_dir = outputs_dir / "reports"

    default_sheet = (
        "https://docs.google.com/spreadsheets/d/"
        "1MwUMdrzEeH0WOaEzzoLXWI-BPY5mSpAiFzJt_e_R5QA/export?format=csv&gid=0"
    )

    return Settings(
        base_dir=base_dir,
        outputs_dir=outputs_dir,
        reports_dir=reports_dir,
        sheet_url=os.getenv("LIVE_SHEET_URL", default_sheet),
        grok_api_key=os.getenv("GROK_API_KEY", "").strip(),
        grok_model=os.getenv("GROK_MODEL", "grok-3-fast").strip() or "grok-3-fast",
        grok_base_url="https://api.x.ai/v1",
        dashboard_refresh_ms=int(os.getenv("DASHBOARD_REFRESH_MS", "30000")),
    )


settings = get_settings()
