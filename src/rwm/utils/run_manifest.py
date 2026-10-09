"""The record written next to every result.

A result is only reportable if its manifest shows: a clean git commit, the
exact config, the checksums of the data it read, the seed, and the installed
package versions. With those five things the run can be repeated exactly.
The manifest also records the machine (processor, cores, memory, GPU and its
memory, CUDA version) and, once the run ends, how long each window took to fit
and forecast, so run times can be reported.
"""
import os
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


def compute_device() -> str:
    """The processor or GPU that deep models will run on."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return "cpu"


def hardware() -> dict:
    """The machine the run used: processor, cores, memory and GPU."""
    info = {"cpu": platform.processor() or platform.machine(), "logical_cores": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["memory_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    return info


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
        "device": compute_device(),
        "hardware": hardware(),
        "packages": installed_packages(),
    }
