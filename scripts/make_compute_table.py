"""Wall-clock time and spend of paper 1's revision (plan, compute records of 7 November 2026).

  python scripts/make_compute_table.py --spend    # writes results/analysis_tuned/spend_summary.csv from the billing
                                                  # snapshots in results/compute (kept out of git)
  python scripts/make_compute_table.py            # writes paper/generated/tab_compute_revision.tex and compute_values.tex

Run time comes from each run's manifest (results/analysis_tuned/compute_runs.csv, scripts/compute_records.py); spend
from the summary, which holds totals only.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "analysis_tuned"
GEN = ROOT / "paper" / "generated"
STUDY = [  # (label, test on the top folder and run name)
    ("Searches of both models' settings", lambda top, run: top == "research"),
    ("Final runs on the rolling windows", lambda top, run: top == "final"),
    ("Transfer and new-retailer tests", lambda top, run: top == "ftransfer"),
    ("Training on other retailers' data", lambda top, run: top == "pooled"),
    ("Comparison models on cleaned prices", lambda top, run: top == "cleanrev"),
]


def spend() -> None:
    m = json.loads((ROOT / "results" / "compute" / "modal_billing_daily.json").read_text())
    v = json.loads((ROOT / "results" / "compute" / "vast_invoices.json").read_text())
    rows = [{"provider": "Modal", "item": "all of this project's apps", "usd": sum(float(r["cost"]) for r in m),
             "hours": None}]
    for kind in ("GPU", "storage", "download", "upload"):
        sel = [x for x in v if x.get("type") == "charge" and re.search(rf"\b{kind}\b", x["description"])]
        rows.append({"provider": "Vast", "item": kind, "usd": sum(float(x["amount"]) for x in sel),
                     "hours": sum(float(x["quantity"]) for x in sel) if kind == "GPU" else None})
    pd.DataFrame(rows).to_csv(A / "spend_summary.csv", index=False)
    print(pd.DataFrame(rows))


def device(d: str) -> str:
    d = str(d)
    if "4090" in d:
        return "RTX 4090"
    if "5090" in d:
        return "RTX 5090"
    if "cuda" in d:
        return d.split("NVIDIA ")[-1].replace("GeForce ", "")
    return "CPU"


def table() -> None:
    runs = pd.read_csv(A / "compute_runs.csv")
    runs["hours"] = runs.wall_seconds.fillna(0) / 3600
    runs["dev"] = runs.device.map(device)
    lines, vals = [], {}
    total_h = total_n = 0
    for label, test in STUDY:
        g = runs[[test(t, r) for t, r in zip(runs.top, runs.run)]]
        if g.empty:
            continue
        main = ["RTX 4090", "RTX 5090", "L4", "CPU"]
        counts = g.dev.value_counts()
        parts = [f"{k} ({counts[k]})" for k in main if k in counts]
        other = counts[[k for k in counts.index if k not in main]].sum()
        devs = ", ".join(parts + ([f"other GPUs ({other})"] if other else []))
        lines.append(f"{label} & {len(g):,} & {g.hours.sum():,.0f} & {devs} \\\\")
        total_h += g.hours.sum()
        total_n += len(g)
    sp = pd.read_csv(A / "spend_summary.csv")
    vast = sp[sp.provider == "Vast"].usd.sum()
    modal = sp[sp.provider == "Modal"].usd.sum()
    vals.update(computeRevRuns=f"{total_n:,}", computeRevHours=f"{total_h:,.0f}",
                computeVastUsd=f"{vast:,.0f}", computeModalUsd=f"{modal:,.0f}", computeTotalUsd=f"{vast + modal:,.0f}",
                computeVastGpuHours=f"{sp[(sp.provider == 'Vast') & (sp.item == 'GPU')].hours.sum():,.0f}",
                computeVastGpuUsd=f"{sp[(sp.provider == 'Vast') & (sp.item == 'GPU')].usd.sum():,.0f}")
    tab = r"""\begin{table}[t]
\centering
\caption{\textbf{Run time of the \revn{}, from each run's manifest.} Runs counts every run folder, the pieces of
split runs included; hours is their summed wall-clock time; machines are the GPUs or CPU hosts the runs used, with the
number of runs on each. A run resumed from a partway save records only its last stretch, so the hours are a floor;
the machine time paid for is in the text.}
\label{tab:compute-revision}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{0.36\columnwidth}rrp{0.32\columnwidth}}
\toprule
\textbf{Study} & \textbf{Runs} & \textbf{Hours} & \textbf{Machines} \\
\midrule
""" + "\n".join(lines) + "\n" + rf"""\midrule
All & {total_n:,} & {total_h:,.0f} & \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (GEN / "tab_compute_revision.tex").write_text(tab)
    (GEN / "compute_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()))
    print(tab, vals)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spend", action="store_true")
    a = ap.parse_args()
    spend() if a.spend else table()
