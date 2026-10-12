"""Gather the manifest and scores of runs quoted in the paper into one zip file,
to be unpacked into results/promoted/ and committed.

  python scripts/promote.py --results <results folder> --out <zip file> [--pattern GLOB ...]

Run where the run folders are (Colab, on Drive). Only manifest.json and
metrics.json are taken; forecasts and checkpoints stay outside git, and
metrics.json holds each checkpoint's checksum. Unfinished runs (no
metrics.json) and partway saves are skipped.
"""
import argparse
import zipfile
from pathlib import Path

PATTERNS = [
    "*_m5_test_*", "*_m5_confirm_*", "*_m5_validation_timesfm*", "*_m5_validation_graph*",
    "*_m5_validation_world_model_roll_norm*", "*_m5_validation_lift*",
    "*_frat_final_*", "*_dominicks_*_final_*",
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pattern", nargs="*", default=PATTERNS)
    args = p.parse_args(argv)
    results = Path(args.results)
    found = sorted({d for pat in args.pattern for d in results.glob(pat)
                    if d.is_dir() and (d / "metrics.json").exists() and (d / "manifest.json").exists()})
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in found:
            for f in ("manifest.json", "metrics.json"):
                z.write(d / f, f"{d.name}/{f}")
    print(f"{len(found)} runs written to {args.out}")


if __name__ == "__main__":
    main()
