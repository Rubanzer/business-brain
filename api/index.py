"""Vercel serverless entry point — re-exports the FastAPI app."""

import sys
from pathlib import Path

# Ensure src/ and project root are on sys.path for imports
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from business_brain.action.api import app  # noqa: E402, F401
