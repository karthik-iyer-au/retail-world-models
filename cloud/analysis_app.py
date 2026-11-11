"""Scores saved runs where the data are: the tuned-model analysis of the final runs (entry of 10 November 2026),
the ensemble rebuilt from the tuned world model, the tuned LightGBM and TimesFM with the plan.

  modal run cloud/analysis_app.py --commit <sha> --keys dominicks_che frat ...

Links every run folder of rwm-results/final and rwm-results/cleanrev into one folder and runs
scripts/revision_analysis.py --tuned on it, writing rwm-results/analysis_tuned/<parts>.csv.
An ephemeral app: it deploys nothing and touches no running loop.
"""
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_app import CHECKOUT, DATA, RESULTS, _checkout, data_volume, image, results_volume  # noqa: E402

app = modal.App("rwm-analysis")
# the container needs modal_app.py beside this file, for the import above
image = image.add_local_file(Path(__file__).resolve().parent / "modal_app.py", "/root/modal_app.py")


@app.function(image=image, volumes={DATA: data_volume, RESULTS: results_volume}, cpu=8, memory=96 * 1024, timeout=8 * 3600)
def analyse(commit: str, keys: list[str], parts: list[str], boot: int) -> str:
    _checkout(commit)
    runs = Path("/tmp/runs")
    runs.mkdir(exist_ok=True)
    for src in ("final", "cleanrev"):
        for d in Path(RESULTS, src).iterdir():
            if (d / "forecasts.csv").exists() and (d / "manifest.json").exists() and "_piece_" not in d.name:
                (runs / d.name).symlink_to(d)
    out = Path(RESULTS, "analysis_tuned")
    cmd = [sys.executable, "scripts/revision_analysis.py", "--tuned", "--results", str(runs), "--out", str(out),
           "--parts", *parts, "--boot", str(boot)] + (["--keys", *keys] if keys else [])
    done = subprocess.run(cmd, cwd=CHECKOUT, capture_output=True, text=True)
    results_volume.commit()
    return done.stdout[-4000:] + done.stderr[-4000:]


@app.local_entrypoint()
def main(commit: str, keys: str = "", parts: str = "scores,paired,mcs", boot: int = 1000):
    print(analyse.remote(commit, [k for k in keys.split(",") if k], parts.split(","), boot))
