from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_file: Path
    output_dir: Path
    charts_dir: Path
    reports_dir: Path
    grok_api_key: str
    grok_model: str


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "outputs"
    charts_dir = output_dir / "charts"
    reports_dir = output_dir / "reports"

    return Settings(
        base_dir=base_dir,
        data_file=base_dir / "data" / "waste_data.csv",
        output_dir=output_dir,
        charts_dir=charts_dir,
        reports_dir=reports_dir,
        grok_api_key=os.getenv("GROK_API_KEY", "").strip(),
        grok_model=os.getenv("GROK_MODEL", "grok-3-fast"),
    )


settings = get_settings()
