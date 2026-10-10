import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("compare_forecasts", Path(__file__).parents[1] / "scripts" / "compare_forecasts.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


def _run(folder: Path, forecast) -> Path:
    folder.mkdir()
    ids = ["FOODS_3_001_CA_1", "HOBBIES_1_002_TX_2"]
    dates = pd.date_range("2016-05-23", periods=14).strftime("%Y-%m-%d")
    rows = [(s, d, 2.0) for s in ids for d in dates]
    df = pd.DataFrame(rows, columns=["series_id", "date", "units"])
    df["forecast"] = forecast(df)
    df["origin"] = 0
    df.to_csv(folder / "forecasts.csv", index=False)
    return folder


def test_bias_and_closer_share_by_store(tmp_path, capsys):
    over = _run(tmp_path / "a1", lambda d: d["units"] * 1.1)
    over2 = _run(tmp_path / "a2", lambda d: d["units"] * 1.3)  # seeds average to 20% over
    exact_ca = _run(tmp_path / "b", lambda d: np.where(d["series_id"].str.endswith("CA_1"), 2.0, 1.0))
    cf.main(["--a", str(over), str(over2), "--b", str(exact_ca), "--names", "A", "B", "--out", str(tmp_path / "out")])
    t = pd.read_csv(tmp_path / "out" / "by_store.csv", index_col=0)
    assert np.isclose(t.loc["CA_1", "A bias %"], 20.0)
    assert np.isclose(t.loc["TX_2", "B bias %"], -50.0)
    assert t.loc["CA_1", "pairs where A is closer %"] == 0
    assert t.loc["TX_2", "pairs where A is closer %"] == 100
    d = pd.read_csv(tmp_path / "out" / "by_department.csv", index_col=0)
    assert set(d.index) == {"FOODS_3", "HOBBIES_1"}
    assert "Company total, day by day" in capsys.readouterr().out


def test_summed_error_lets_opposite_errors_cancel(tmp_path):
    # A is 1 unit high on one item and 1 unit low on the other: the store total is exact.
    a = _run(tmp_path / "a", lambda d: np.where(d["series_id"].str.startswith("FOODS"), 3.0, 1.0))
    b = _run(tmp_path / "b", lambda d: d["units"])
    cf.main(["--a", str(a), "--b", str(b), "--out", str(tmp_path / "out")])
    t = pd.read_csv(tmp_path / "out" / "by_all.csv", index_col=0)
    assert t.loc["all", "A summed error %"] == 0
    assert t.loc["all", "A item error %"] == 50


def test_bands_can_come_from_an_earlier_window(tmp_path):
    a = _run(tmp_path / "a", lambda d: d["units"])
    b = _run(tmp_path / "b", lambda d: d["units"])
    early = _run(tmp_path / "early", lambda d: d["units"])
    f = pd.read_csv(early / "forecasts.csv")
    f.loc[f["series_id"].str.startswith("FOODS"), "units"] = 20.0  # sold fast before, 2 a day now
    f.to_csv(early / "forecasts.csv", index=False)
    cf.main(["--a", str(a), "--b", str(b), "--band-from", str(early), "--out", str(tmp_path / "out")])
    t = pd.read_csv(tmp_path / "out" / "by_sales_band.csv", index_col=0)
    assert set(t.index) == {"1 to 3", "over 10"}
