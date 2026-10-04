"""The record written next to every result.

A result is only reportable if its manifest shows: a clean git commit, the
exact config, the checksums of the data it read, the seed, and the installed
package versions. With those five things the run can be repeated exactly.
"""
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from rwm.utils.hashing import sha256_obj
from rwm.utils.paths import REPO_ROOT


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit or None, "dirty": dirty if commit else None}


def installed_packages() -> list[str]:
    return sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in metadata.distributions()
        if d.metadata["Name"]
    )


def build_manifest(config: dict, data_files: list[dict]) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "config": config,
        "config_sha256": sha256_obj(config),
        "seed": config.get("seed"),
        "data_files": data_files,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": installed_packages(),
    }
