"""Repository paths. Everything is resolved from the repository root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_MANIFESTS = REPO_ROOT / "data" / "manifests"
RESULTS = REPO_ROOT / "results"
