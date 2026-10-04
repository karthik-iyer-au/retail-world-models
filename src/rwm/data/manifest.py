"""Record and check the checksums of raw data files.

  python -m rwm.data.manifest register m5   # after placing files in data/raw/m5/
  python -m rwm.data.manifest verify m5

`register` writes data/manifests/<dataset>.json, which is committed. From then
on anyone can confirm they hold byte-identical files.
"""
import json
import sys
from pathlib import Path

from rwm.utils.hashing import sha256_file
from rwm.utils.paths import DATA_MANIFESTS, DATA_RAW


def describe(paths: list[Path], root: Path) -> list[dict]:
    return [
        {
            "file": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(paths)
    ]


def manifest_path(dataset: str) -> Path:
    return DATA_MANIFESTS / f"{dataset}.json"


def register(dataset: str) -> Path:
    root = DATA_RAW / dataset
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no files under {root}")
    out = manifest_path(dataset)
    out.write_text(json.dumps({"dataset": dataset, "files": describe(files, root)}, indent=2) + "\n")
    return out


def verify(dataset: str, only_present: bool = False, files: list[Path] | None = None) -> list[dict]:
    """Returns the entries checked. Raises if any file differs or is missing.
    With `only_present`, registered files that are absent are skipped.
    With `files`, exactly those files are checked (the ones a run reads), and
    each must be registered."""
    path = manifest_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: make data-register DATASET={dataset}")
    root = DATA_RAW / dataset
    entries = json.loads(path.read_text())["files"]
    if files is not None:
        wanted = {str(Path(f).resolve().relative_to(root.resolve())) for f in files}
        unknown = wanted - {e["file"] for e in entries}
        if unknown:
            raise ValueError(f"not registered in {path}: {sorted(unknown)}")
        entries = [e for e in entries if e["file"] in wanted]
    checked = []
    for e in entries:
        p = root / e["file"]
        if not p.exists():
            if only_present:
                continue
            raise FileNotFoundError(f"{p} is listed in the manifest but missing")
        if sha256_file(p) != e["sha256"]:
            raise ValueError(f"{p} does not match the registered checksum")
        checked.append(e)
    return checked


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("register", "verify"):
        sys.exit("usage: python -m rwm.data.manifest [register|verify] <dataset>")
    if sys.argv[1] == "register":
        print(f"wrote {register(sys.argv[2])}")
    else:
        print(f"{len(verify(sys.argv[2]))} files match")
