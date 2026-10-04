"""Breakfast at the Frat loader (dunnhumby source file).

Weekly sales by product and store for 77 stores, 55 products in four
categories (pretzels, cold cereal, frozen pizza, mouthwash), 156 weeks from
January 2009. Distributed by dunnhumby to registered users.

Reads the zip as downloaded: one workbook with a transaction sheet, a store
sheet and a product sheet.

What the data records, following the dunnhumby user guide:
- units: UNITS sold in the week.
- price: PRICE, the amount charged at shelf. base_price is the item's
  regular price, so price below base_price is a discount.
- display: 1 if the product was on in-store display.
- feature: 1 if the product was in the store circular (brochure).
- tpr_only: 1 if the price was reduced with a shelf tag only, with no display
  and no circular.
- promo: 1 if any of display, feature or tpr_only is set.
- The date is the week-ending date.

Not recorded: cost, shelf space.

Two stores are listed twice in the store sheet with different value
segments. The first listing is used.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rwm.data.registry import Dataset, register_dataset
from rwm.data.schema import DATE, ITEM, PRICE, SERIES, STORE, UNITS
from rwm.utils.paths import DATA_RAW

ZIP_NAME = "dunnhumby_Breakfast-at-the-Frat.zip"
HIERARCHY = [[], ["cat_id"], ["store_id"], ["cat_id", "store_id"], ["item_id"], ["item_id", "store_id"]]


@register_dataset("breakfast_at_the_frat")
def load_breakfast_at_the_frat(root: str | None = None) -> Dataset:
    path = (Path(root) if root else DATA_RAW / "breakfast_at_the_frat") / ZIP_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    with zipfile.ZipFile(path) as z:
        book = io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(".xlsx"))))
    d = pd.read_excel(book, sheet_name="dh Transaction Data", header=1, usecols="A:L")
    stores = pd.read_excel(book, sheet_name="dh Store Lookup", header=1, usecols="A:I")
    items = pd.read_excel(book, sheet_name="dh Products Lookup", header=1, usecols="A:F")
    stores = stores.drop_duplicates("STORE_ID").set_index("STORE_ID")
    items = items.drop_duplicates("UPC").set_index("UPC")

    # Fixed-width text keys so that text order equals numeric order.
    store = d["STORE_NUM"].map("{:05d}".format)
    item = d["UPC"].map("{:011d}".format)
    flags = d[["FEATURE", "DISPLAY", "TPR_ONLY"]].astype(np.int8)
    panel = pd.DataFrame(
        {
            SERIES: pd.Categorical(item + "_" + store),
            STORE: pd.Categorical(store),
            ITEM: pd.Categorical(item),
            DATE: pd.to_datetime(d["WEEK_END_DATE"]),
            UNITS: d["UNITS"].astype(np.float32),
            PRICE: d["PRICE"].astype(np.float32),
            "base_price": d["BASE_PRICE"].astype(np.float32),
            "promo": flags.max(axis=1),
            "display": flags["DISPLAY"],
            "feature": flags["FEATURE"],
            "tpr_only": flags["TPR_ONLY"],
            "cat_id": pd.Categorical(items["CATEGORY"].reindex(d["UPC"]).to_numpy()),
            "sub_category": pd.Categorical(items["SUB_CATEGORY"].reindex(d["UPC"]).to_numpy()),
            "manufacturer": pd.Categorical(items["MANUFACTURER"].reindex(d["UPC"]).to_numpy()),
            "state": pd.Categorical(stores["ADDRESS_STATE_PROV_CODE"].reindex(d["STORE_NUM"]).to_numpy()),
            "store_segment": pd.Categorical(stores["SEG_VALUE_NAME"].reindex(d["STORE_NUM"]).to_numpy()),
        }
    )
    panel = panel.sort_values([SERIES, DATE], kind="stable").reset_index(drop=True)
    return Dataset(
        "breakfast_at_the_frat",
        panel,
        ["price", "promo", "display", "feature"],
        files=[path],
        hierarchy=HIERARCHY,
    )
