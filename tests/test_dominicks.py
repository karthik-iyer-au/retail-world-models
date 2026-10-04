import zipfile

import numpy as np
import pandas as pd
import pytest

from rwm.data import load_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.utils.paths import DATA_RAW

MOVEMENT = """STORE,UPC,WEEK,MOVE,QTY,PRICE,SALE,PROFIT,OK,PRICE_HEX,PROFIT_HEX
76,1192603016,2,0,1,0,,0,1,0,0
76,1192603016,1,4,1,2.00,,25,1,0,0
76,1192603016,3,9,3,5.00,B,10,1,0,0
76,1192603016,4,7,1,2.00,,25,0,0,0
5,1192603016,1,1,1,3.00,S,50,1,0,0
5,300,1,2,1,1.00,,20,1,0,0
"""


@pytest.fixture
def small(tmp_path):
    with zipfile.ZipFile(tmp_path / "wxyz.zip", "w") as z:
        z.writestr("wxyz.csv", MOVEMENT)
    (tmp_path / "upcxyz.csv").write_text(
        "COM_CODE,UPC,DESCRIP,SIZE,CASE,NITEM\n953,1192603016,ITEM A,16 CT,6,1\n954,300,ITEM B,8 CT,6,2\n"
    )
    return load_dataset("dominicks", categories=["xyz"], root=str(tmp_path))


def test_loader_follows_the_manual(small):
    p = small.panel
    assert small.levers == ["price", "promo", "cost"]
    assert len(p) == 5  # the week flagged OK = 0 is dropped
    assert p[SERIES].astype(str).tolist() == [
        "000000000300_005", "001192603016_005",
        "001192603016_076", "001192603016_076", "001192603016_076",
    ]
    s = p[p[SERIES] == "001192603016_076"]
    assert s[DATE].dt.strftime("%Y-%m-%d").tolist() == ["1989-09-14", "1989-09-21", "1989-09-28"]
    assert s[UNITS].tolist() == [4, 0, 9]
    assert s["promo"].tolist() == [0, 0, 1] and s["promo_type"].tolist() == ["none", "none", "B"]
    # week 2 has no price on file; week 3 is a bundle of 3 for 5.00 at 10% margin
    np.testing.assert_allclose(s[PRICE], [2.0, np.nan, 5 / 3], rtol=1e-6)
    np.testing.assert_allclose(s["cost"], [1.5, np.nan, 1.5], rtol=1e-6)
    assert p["cat_id"].eq("xyz").all() and set(p["com_code"]) == {953, 954}
    assert p[STORE].astype(str).unique().tolist() == ["005", "076"]
    assert p[ITEM].astype(str).nunique() == 2


@pytest.mark.skipif(not (DATA_RAW / "dominicks" / "wana_csv.zip").exists(), reason="Dominick's files not present")
def test_real_category_matches_the_raw_file():
    p = load_dataset("dominicks", categories=["ana"]).panel
    raw = pd.read_csv(DATA_RAW / "dominicks" / "wana_csv.zip", usecols=["MOVE", "QTY", "PRICE", "OK"])
    raw = raw[raw["OK"] == 1]
    assert len(p) == len(raw)
    assert p[UNITS].astype("float64").sum() == raw["MOVE"].sum()
    revenue = (p[PRICE].fillna(0).astype("float64") * p[UNITS]).sum()
    assert revenue == pytest.approx((raw["PRICE"] * raw["MOVE"] / raw["QTY"]).sum(), rel=1e-6)
