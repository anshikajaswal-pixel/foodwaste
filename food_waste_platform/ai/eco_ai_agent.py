from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError as exc:
    raise ImportError(
        "Gemini dependency missing. Install with: pip install google-generativeai"
    ) from exc

import pandas as pd

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to the .env file."
    )

genai.configure(api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "FoodWaste.csv"

if not data_path.exists():
    raise FileNotFoundError(f"FoodWaste.csv not found at: {data_path}")

df = pd.read_csv(data_path)


def generate_pandas_query(question: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are a professional data analyst working with a pandas dataframe called df.

Columns available:
{list(df.columns)}

User question:
{question}

Write Python pandas code that answers the question.

Rules:
- Use only the dataframe df
- Do not import libraries
- Do not print anything
- The final line must assign the answer to a variable named result

Return ONLY the Python code.
""".strip()

    response = model.generate_content(prompt)
    code = (response.text or "").strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code


def execute_query(code: str) -> Any:
    local_vars: dict[str, Any] = {"df": df.copy(), "pd": pd}
    exec(code, {}, local_vars)
    return local_vars.get("result")


def eco_ai(question: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    code = generate_pandas_query(question)
    result = execute_query(code)

    prompt = f"""
You are a sustainability data analyst.

User question:
{question}

Computed result:
{result}

Explain the answer clearly using the dataset.
""".strip()

    response = model.generate_content(prompt)
    return (response.text or "No explanation generated.").strip()
