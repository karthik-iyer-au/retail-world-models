"""Paper 1's table of what each data set holds and what the runs used, from results/data_spans/data_spans.csv
(scripts/data_spans.py). Writes paper/generated/tab_data_spans.tex."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"m5": "M5", "frat": "Frat", "favorita": "Favorita"}


def month(s: str) -> str:
    return pd.Timestamp(s).strftime("%b %Y")


def main():
    d = pd.read_csv(ROOT / "results" / "data_spans" / "data_spans.csv").set_index("key")
    dom = d[d.index.str.startswith("dominicks")]
    rows = []
    for key in ["m5", "dominicks", "frat", "favorita"]:
        if key == "dominicks":
            r = dom.iloc[0].copy()
            r["first_train"] = min(dom.first_train)
            r["first_test"] = min(dom.first_test)
            name, unit = "Dominick's", "weeks"
        else:
            r, name = d.loc[key], NAMES[key]
            unit = "days" if key in ("m5", "favorita") else "weeks"
        held = f"{month(r.first)} to {month(r.last)}"
        if key == "favorita":
            held = f"{month(d.loc['favorita_from_2013'].first)} to {month(r.last)}"
        if key == "m5":  # the competition's final 28 days, in the organisers' test file (rwm.data.m5, include_test)
            held = f"{month(r.first)} to Jun 2016"
        loaded = f"{month(r.first)} to {month(r.last)}"
        train = f"{int(r.train_periods)} {unit}, from {month(r.first_train)}"
        test = f"{int(r.windows)} of {int(r.horizon)} {unit}, {month(r.first_test)} to {month(r.last_test)}"
        out = f"{int(r.holdout)} {unit} from {month(r.holdout_from)}" if r.holdout else "none"
        rows.append(f"{name} & {held} & {loaded} & {train} & {test} & {out} \\\\")
    tab = (r"""\begin{table*}[t]
\centering
\caption{\textbf{What each data set holds and what the runs use.} The source data, the part the paper loads, the
training length of the final runs and the earliest period it reaches, the rolling test windows, and the final periods
held out of every run. Dominick's: the six categories share their dates; training counts weeks with data, so it starts earlier than the
calendar would give where a category has weeks without data, and the table gives the earliest start of the six.
M5's last 28 days are in the organisers' test file and are not loaded. Everything loaded before the training
start is not trained on by the final runs.}
\label{tab:data-spans}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\textwidth}\begin{tabular}{llllll}
\toprule
Data & Source holds & Loaded & Training (final runs) & Test windows & Held out \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\end{adjustbox}
\end{table*}
""")
    (ROOT / "paper" / "generated" / "tab_data_spans.tex").write_text(tab)
    print(tab)


if __name__ == "__main__":
    main()
