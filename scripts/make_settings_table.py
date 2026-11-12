"""Table of each data set's tuned world model and tuned LightGBM, read from the final configs
(configs/experiments/final_<data>_<model>.yaml, written by scripts/make_final_configs.py).

  python scripts/make_settings_table.py     # writes paper/generated/tab_final_settings.tex
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"m5": "M5", "favorita": "Favorita", "frat": "Frat", "dominicks_ana": "Analgesics",
        "dominicks_bjc": "Bottled juice", "dominicks_che": "Cheese", "dominicks_fre": "Frozen entrees",
        "dominicks_sdr": "Soft drinks", "dominicks_sna": "Snacks"}
LOSS = {None: "Tweedie", "negative_binomial": "Negative binomial", "hurdle_nb": "Hurdle negative binomial"}


def load(key: str, model: str) -> tuple[dict, dict, list[str]]:
    path = ROOT / "configs" / "experiments" / f"final_{key}_{model}.yaml"
    head = path.read_text().splitlines()[0]
    kept = re.search(r"kept: ([^)]*)\)", head)
    kept = [] if not kept or kept.group(1) == "nothing" else [k.split(":")[1].replace("_", " ") for k in kept.group(1).split(", ")]
    cfg = yaml.safe_load(path.read_text())
    p = cfg["model"]["params"]
    return p, (p["params"] if cfg["model"]["name"] == "seed_average" else p), kept


def main() -> None:
    rows = []
    for key, name in KEYS.items():
        p, w, _ = load(key, "world_model")
        copies = p.get("members", 1) if "members" in p else 1
        pre = f", pretrained {w['pretrain_steps']:,} steps" if w.get("pretrain_steps") else ""
        wm = (f"{copies} & {w['d_model']}$\\times${w['layers']} & {w['steps']:,}{pre} & {w['train_periods']} & "
              f"{LOSS[w.get('likelihood')]}")
        q, g, kept = load(key, "lightgbm")
        lcopies = q.get("members", 1) if "members" in q else 1
        obj = "Poisson" if "objective poisson" in kept else "Tweedie"
        lgb = f"{lcopies} & {g.get('rounds', '--')} & {g.get('train_periods', '--')} & {obj}"
        rows.append(f"{name} & {wm} & {lgb} \\\\")
    tab = r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{The tuned settings used for every result of Section~\ref{sec:results}, read from the final configurations.
Copies: forecasts averaged over that many fits with different seeds. Size: state width by encoder layers. Steps:
training steps (and pretraining steps on the latent objective, where used). Periods: training periods read. The
changes each search kept are listed in Table~\ref{tab:loop}.}
\label{tab:final-settings}
\begin{tabular}{l ccccc cccc}
\toprule
 & \multicolumn{5}{c}{World model} & \multicolumn{4}{c}{LightGBM}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-10}
Data set & Copies & Size & Steps & Periods & Loss & Copies & Rounds & Periods & Objective\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table*}
"""
    (ROOT / "paper" / "generated" / "tab_final_settings.tex").write_text(tab)
    print(tab)


if __name__ == "__main__":
    main()
