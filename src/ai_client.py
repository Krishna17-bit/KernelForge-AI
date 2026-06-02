from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AIReasoningClient:
    configured: bool
    status_help: str

    @classmethod
    def from_env(cls) -> "AIReasoningClient":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return cls(True, "AI narrative mode is configured locally. Static benchmarking still runs without it.")
        return cls(False, "Optional AI narrative mode is not configured. Benchmarks, exports, and kernel generation still work.")

    def summarize(self, prompt: str) -> str:
        if not self.configured:
            return "AI narrative mode is not configured. Run benchmarks and use the built-in profiler recommendation instead."
        try:
            import google.generativeai as genai  # type: ignore
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or "No narrative response returned."
        except Exception as exc:
            return f"AI narrative generation failed locally: {exc}"
