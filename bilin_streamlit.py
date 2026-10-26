"""Compatibility entry point for the existing Streamlit Cloud deployment.

The app lives in app.py; this file only runs it, so the deployed app keeps its URL.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
