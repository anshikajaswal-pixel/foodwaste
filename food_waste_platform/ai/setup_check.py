from __future__ import annotations


def check_gemini_dependency() -> bool:
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        print("Gemini library missing. Run: pip install google-genai")
        return False


if __name__ == "__main__":
    check_gemini_dependency()
