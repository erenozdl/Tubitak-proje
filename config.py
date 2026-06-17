"""
Centralized configuration loader.

All settings are pulled from environment variables (via a .env file).
Copy .env.example to .env and fill in your credentials before running.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def _require(var_name: str) -> str:
    """Return an env var or exit with a clear error if it's missing."""
    value = os.getenv(var_name)
    if not value:
        sys.exit(f"[CONFIG ERROR] Required environment variable '{var_name}' is not set. "
                 f"Check your .env file.")
    return value


# ── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.2")

# ── Gmail credentials ────────────────────────────────────
GMAIL_ADDRESS: str = _require("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD: str = _require("GMAIL_APP_PASSWORD")

# IMAP / SMTP servers (Gmail defaults, override if needed)
IMAP_SERVER: str = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT: int = int(os.getenv("IMAP_PORT", "993"))
SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))

# ── Department forwarding addresses ──────────────────────
AIR_DEPT_EMAIL: str = os.getenv("AIR_DEPT_EMAIL", "")
MARITIME_DEPT_EMAIL: str = os.getenv("MARITIME_DEPT_EMAIL", "")
ROAD_DEPT_EMAIL: str = os.getenv("ROAD_DEPT_EMAIL", "")
OVERSEAS_DEPT_EMAIL: str = os.getenv("OVERSEAS_DEPT_EMAIL", "")

# ── Behaviour tuning ────────────────────────────────────
POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# ── Introduction email template ──────────────────────────
INTRODUCTION_EMAIL_SUBJECT: str = os.getenv(
    "INTRODUCTION_EMAIL_SUBJECT",
    "Welcome – Adamar Air & Sea Services Co. Ltd."
)

# The introduction body is loaded from templates/introduction.txt so it's
# easy to edit without touching code.  Falls back to env var if the file
# is missing (e.g. in a minimal deployment).
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_INTRO_FILE = os.path.join(_TEMPLATE_DIR, "introduction.txt")

try:
    with open(_INTRO_FILE, "r", encoding="utf-8") as f:
        INTRODUCTION_EMAIL_BODY: str = f.read()
except FileNotFoundError:
    INTRODUCTION_EMAIL_BODY: str = os.getenv(
        "INTRODUCTION_EMAIL_BODY",
        "[PLACEHOLDER] — templates/introduction.txt not found."
    )
