"""
Single-command startup: launches the Streamlit UI, which is a self-
contained evaluator experience (upload, ingest, ask, inspect citations
and evaluation log — no terminal use needed after this).

Run:  python run.py
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "app" / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
