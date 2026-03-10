from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_path: Path
    outputs_dir: Path
    charts_dir: Path
    reports_dir: Path
    grok_api_key: str
    grok_model: str
    grok_base_url: str


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "food_waste_ai" / "data" / "waste_data.csv"
    outputs_dir = project_root / "food_waste_ai" / "outputs"
    charts_dir = outputs_dir / "charts"
    reports_dir = outputs_dir / "reports"

    return Settings(
        project_root=project_root,
        data_path=data_path,
        outputs_dir=outputs_dir,
        charts_dir=charts_dir,
        reports_dir=reports_dir,
        grok_api_key=os.getenv("GROK_API_KEY", "").strip(),
        grok_model=os.getenv("GROK_MODEL", "grok-3-fast").strip() or "grok-3-fast",
        grok_base_url="https://api.x.ai/v1",
    )


settings = get_settings()
