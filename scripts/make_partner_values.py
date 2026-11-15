"""Which second model the average needs (fresh-reader item 4, 15 November 2026): item-by-store WRMSSE, mean over the
rolling windows, of the average of the two tuned models, the three-model ensemble (with TimesFM given the plan) and
LightGBM averaged with TimesFM given the plan (no world model). From results/analysis_tuned/scores.csv and
ablation/scores.csv. Writes paper/generated/partner_values.tex."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORDS = "none one two three four five six seven eight nine".split()


def main():
    s = pd.read_csv(ROOT / "results/analysis_tuned/scores.csv")
    if "set" in s:
        s = s[s["set"] == "rolling"]
    b = s[s.level == "item_idxstore_id"].groupby(["key", "model"]).wrmsse.mean().unstack()
    a = pd.read_csv(ROOT / "results/analysis_tuned/ablation/scores.csv")
    n = a[a.level == "item_idxstore_id"].groupby(["key", "model"]).wrmsse.mean().unstack()
    avg, three, nowm = b["ensemble_2"].round(3), b["ensemble"].round(3), n["without_world_model"].round(3)
    v = {"ptThreeBelowAvg": WORDS[int((three < avg).sum())], "ptThreeAboveAvg": WORDS[int((three > avg).sum())],
         "ptNoWmAtOrBelowAvg": WORDS[int((nowm <= avg).sum())], "ptNoWmAboveAvg": WORDS[int((nowm > avg).sum())],
         "ptSets": WORDS[len(avg)]}
    (ROOT / "paper/generated/partner_values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in v.items()))
    print(v)


if __name__ == "__main__":
    main()
