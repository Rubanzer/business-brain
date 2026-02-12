"""Vercel serverless entry point — re-exports the FastAPI app."""

import sys
from pathlib import Path

# Ensure src/ and project root are on sys.path for imports
_root = Path(__file__).resolve().parent.parent
for p in [str(_root / "src"), str(_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from business_brain.action.api import app  # noqa: E402, F401

# Vercel requires the app variable to be named `app` or `handler`
handler = app
