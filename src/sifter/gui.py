"""Console launcher for SIFTER's local Streamlit interface."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the bundled application on the local loopback interface."""
    repository_app = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    bundled_app = Path(__file__).with_name("_streamlit_app.py")
    app_path = repository_app if repository_app.is_file() else bundled_app
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            "localhost",
        ],
        check=True,
    )
