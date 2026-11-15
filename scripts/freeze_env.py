"""Writes the exact package versions of the environment it runs in (the lockfile asked for in the outside review of
15 November 2026). Run inside the cloud image the experiments use:

  modal run cloud/analysis_app.py --commit <sha> --run freeze_env --args "{out}/requirements-lock.txt" --out env --cpu
"""
import platform
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    out = Path((argv or sys.argv[1:])[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    pkgs = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True,
                          text=True, check=True).stdout
    out.write_text(f"# Python {platform.python_version()}, {platform.platform()}\n" + pkgs)
    print(out.read_text()[:300])


if __name__ == "__main__":
    main()
