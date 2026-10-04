import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from rwm.data import load_dataset
from rwm.data.breakfast_at_the_frat import ZIP_NAME
from rwm.data.schema import DATE, PRICE, SERIES, UNITS
from rwm.utils.paths import DATA_RAW


def _workbook() -> bytes:
    wb = Workbook()
    sheets = {
        "dh Transaction Data": [
            ["WEEK_END_DATE", "STORE_NUM", "UPC", "UNITS", "VISITS", "HHS", "SPEND", "PRICE", "BASE_PRICE", "FEATURE", "DISPLAY", "TPR_ONLY"],
            [dt.datetime(2009, 1, 21), 367, 1111009477, 5, 5, 5, 6.95, 1.39, 1.57, 0, 0, 1],
            [dt.datetime(2009, 1, 14), 367, 1111009477, 13, 13, 13, 20.41, 1.57, 1.57, 0, 0, 0],
            [dt.datetime(2009, 1, 14), 23, 7797502248, 9, 9, 9, 18.0, 2.0, 2.5, 1, 1, 0],
        ],
        "dh Store Lookup": [
            ["STORE_ID", "STORE_NAME", "ADDRESS_CITY_NAME", "ADDRESS_STATE_PROV_CODE", "MSA_CODE", "SEG_VALUE_NAME", "PARKING_SPACE_QTY", "SALES_AREA_SIZE_NUM", "AVG_WEEKLY_BASKETS"],
            [367, "A", "X", "OH", 1, "VALUE", None, 100, 10.0],
            [23, "B", "Y", "TX", 2, "MAINSTREAM", None, 200, 20.0],
            [23, "B", "Y", "TX", 2, "UPSCALE", None, 200, 20.0],
        ],
        "dh Products Lookup": [
            ["UPC", "DESCRIPTION", "MANUFACTURER", "CATEGORY", "SUB_CATEGORY", "PRODUCT_SIZE"],
            [1111009477, "PL MINI TWIST PRETZELS", "PRIVATE LABEL", "BAG SNACKS", "PRETZELS", "15 OZ"],
            [7797502248, "CEREAL", "MAKER", "COLD CEREAL", "ALL FAMILY CEREAL", "12 OZ"],
        ],
    }
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        ws.append([None, None, None, "Breakfast at the Frat: A Time Series Analysis"])
        for row in rows:
            ws.append(row)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def test_loader_follows_the_user_guide(tmp_path):
    with zipfile.ZipFile(tmp_path / ZIP_NAME, "w") as z:
        z.writestr("dunnhumby_Breakfast-at-the-Frat/dunnhumby - Breakfast at the Frat.xlsx", _workbook())
    ds = load_dataset("breakfast_at_the_frat", root=str(tmp_path))
    p = ds.panel
    assert ds.levers == ["price", "promo", "display", "feature"]
    assert p[SERIES].astype(str).tolist() == ["01111009477_00367", "01111009477_00367", "07797502248_00023"]
    assert p[DATE].dt.strftime("%Y-%m-%d").tolist() == ["2009-01-14", "2009-01-21", "2009-01-14"]
    assert p[UNITS].tolist() == [13, 5, 9]
    np.testing.assert_allclose(p[PRICE], [1.57, 1.39, 2.0], rtol=1e-6)
    np.testing.assert_allclose(p["base_price"], [1.57, 1.57, 2.5], rtol=1e-6)
    assert p["promo"].tolist() == [0, 1, 1]
    assert p["tpr_only"].tolist() == [0, 1, 0]
    assert p["display"].tolist() == [0, 0, 1] and p["feature"].tolist() == [0, 0, 1]
    assert p["cat_id"].astype(str).tolist() == ["BAG SNACKS", "BAG SNACKS", "COLD CEREAL"]
    assert p["store_segment"].astype(str).tolist() == ["VALUE", "VALUE", "MAINSTREAM"]  # first listing wins


@pytest.mark.skipif(not (DATA_RAW / "breakfast_at_the_frat" / ZIP_NAME).exists(), reason="file not present")
def test_real_file_matches_the_user_guide_counts():
    p = load_dataset("breakfast_at_the_frat").panel
    assert len(p) == 524950  # the record count stated in the user guide
    assert p[DATE].nunique() == 156 and p["store_id"].nunique() == 77 and p["item_id"].nunique() == 55
    assert p[UNITS].astype("float64").sum() == 10293354
    assert set(p["cat_id"].astype(str)) == {"BAG SNACKS", "COLD CEREAL", "FROZEN PIZZA", "ORAL HYGIENE PRODUCTS"}
