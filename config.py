"""
Application configuration — all tuneable knobs in one place.
Values are read from environment variables (via .env) with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str      = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS: int = int(os.environ.get("CLAUDE_MAX_TOKENS", "512"))

# ── Server ────────────────────────────────────────────────────────────────────
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8000"))

# ── Analysis defaults ─────────────────────────────────────────────────────────
DEFAULT_PERIOD: str         = os.environ.get("DEFAULT_PERIOD", "3y")
DEFAULT_BENCHMARK: str      = os.environ.get("DEFAULT_BENCHMARK", "ES3.SI")
CORR_WINDOW_DEFAULT: int    = int(os.environ.get("CORR_WINDOW_DEFAULT", "90"))
FACTOR_WINDOW_DEFAULT: int  = int(os.environ.get("FACTOR_WINDOW_DEFAULT", "252"))
