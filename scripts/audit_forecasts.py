"""Check that every scored run has exactly one finite forecast for every expected series, date and window
(15 November 2026, after an outside code review: the scorer used to read a missing or non-finite forecast as zero and
let a duplicate overwrite).

  python scripts/audit_forecasts.py --root /vol/results --tops final h52 hist cleanrev rev ftransfer pooled --out <folder>

For each run folder with forecasts.csv: the data frame of its config (loaded once per data set and settings), and per
window the expected keys (every series and test date where the data has a row). Reports missing keys, extra keys (series or
dates not expected), duplicate keys and non-finite forecasts, and whether metrics.json was written without
forecasts.csv. Writes audit.csv, one row per run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data.schema import DATE, SERIES
from rwm.evaluation import compare as C
from rwm.utils.hashing import sha256_obj


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--tops", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, rows = {}, []
    for top in a.tops:
        base = Path(a.root) / top
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            m = d / "manifest.json"
            if not m.exists() or "_piece_" in d.name:
                continue
            row = dict(top=top, run=d.name, metrics=(d / "metrics.json").exists(), forecasts=(d / "forecasts.csv").exists())
            if not row["forecasts"]:
                rows.append({**row, "status": "no forecasts" if row["metrics"] else "unfinished"})
                continue
            try:
                cfg = json.loads(m.read_text())["config"]
                key = sha256_obj({"dataset": cfg["dataset"], "evaluation": cfg["evaluation"]})
                if key not in frames:
                    frames[key] = C.frame(cfg)
                fr = frames[key]
                f = pd.read_csv(d / "forecasts.csv", usecols=["origin", SERIES, DATE, "forecast"])
                f[DATE] = pd.to_datetime(f[DATE])
                f[SERIES] = f[SERIES].astype(str)
                names = set(fr.names)
                miss = extra = dup = bad = 0
                for sp in fr.splits:
                    g = f[f["origin"] == sp.origin]
                    t0 = int(np.searchsorted(fr.dates, np.datetime64(sp.train_end))) + 1
                    need = fr.present[:, t0:t0 + len(sp.test_dates)]  # rows the data has: each needs a forecast
                    want = int(need.sum())
                    k = g[[SERIES, DATE]]
                    dup += int(k.duplicated().sum())
                    got = k.drop_duplicates()
                    ok = got[SERIES].isin(names) & got[DATE].isin(pd.DatetimeIndex(sp.test_dates))
                    extra += int((~ok).sum())
                    r = fr.names.get_indexer(got.loc[ok, SERIES])
                    c = pd.DatetimeIndex(sp.test_dates).get_indexer(got.loc[ok, DATE])
                    have = np.zeros_like(need)
                    have[r, c] = True
                    miss += int((need & ~have).sum())
                    bad += int((~np.isfinite(g["forecast"].to_numpy(dtype=float))).sum())
                other = int((~f["origin"].isin([s.origin for s in fr.splits])).sum())
                row.update(config=cfg["name"], missing=miss, extra=extra + other, duplicate=dup, nonfinite=bad,
                           status="ok" if miss == extra == dup == bad == other == 0 else "FAULT")
            except Exception as e:  # noqa: BLE001 - report and go on
                row.update(status=f"error: {type(e).__name__}: {str(e)[:200]}")
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(out / "audit.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "audit.csv", index=False)


if __name__ == "__main__":
    main()
