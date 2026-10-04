"""Download the public data files and check them against the registered checksums.

  python -m rwm.data.fetch m5              # needs a Kaggle API token, see data/README.md
  python -m rwm.data.fetch m5_reference
  python -m rwm.data.fetch dominicks ana cer   # category codes; none means all

Files already in place are left alone. Every run ends by verifying the
files against data/manifests/, so a changed or damaged download is caught.
"""
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from rwm.data import manifest
from rwm.utils.paths import DATA_RAW

KAGGLE_COMPETITION = "m5-forecasting-accuracy"
KAGGLE_FILES = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]

# Files in the M5 organisers' public Google Drive folder, linked from
# github.com/Mcompetitions/M5-methods. Drive file id, then where it is saved.
ORGANISERS = {
    "m5": {"sales_test_evaluation.csv": "1Bj7Xj15yn4j6-BM_mpSmOVTV0c2ihEdo"},
    "m5_reference": {
        "weights_evaluation.csv": "1M37c__ioZChCYpY69oXSzBi3bKrKVEkN",
        "Estimate WRMSSE.R": "1K9nCkHB1n_OClVGSRmKvMqAGY9KhBXFf",
        "Dataset_ReadMe.txt": "1Go3OcBeVUonEU9yUkvfVqGCtHzGcSJvz",
        "Accuracy_readme.txt": "1e7ZsJUzc55nFI8JSQbVFy43RPeu_Yd6Z",
        "benchmark_sNaive.csv": "1Nmb5hE9qfla2tNYsfp-oA4GYnBG1ZoNo",
        "benchmark_ES_bu.csv": "1T2BRXdTm0wl_EtU0EkQafph2gGFAmVWn",
        "submission_0001_YJ_STU.csv": "1MEpzUZg3EY8OkrEDN1-kZ0-zxajqi1wc",
        "Accuracy_benchmarks_overall.csv": "1SJ9AAT0N1aM8hgsmxlO6ufRhU5YqCqLq",
        "Accuracy_benchmarks_results.csv": "1IPOGpfv-xIDpfXx_YKfq3w2zGjBJuqPf",
        "Accuracy_top50_overall.csv": "1UtgBVkliAiaEiN_e779Dy0wMYvBMUSu2",
        "Accuracy_top50_results.csv": "1g_0Wjv7s6W2i4UmzryWYR1CJSJ08Knyp",
    },
}

KILTS = "https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset"
KILTS_STORES = "https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count"


def _from_drive(files: dict[str, str], folder: Path) -> None:
    import gdown

    for name, file_id in files.items():
        if not (folder / name).exists():
            gdown.download(id=file_id, output=str(folder / name), quiet=True)


def _from_url(url: str, path: Path) -> None:
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(path, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)


def fetch_m5(folder: Path) -> None:
    if not all((folder / f).exists() for f in KAGGLE_FILES):
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(folder)],
            check=True,
        )
        archive = folder / f"{KAGGLE_COMPETITION}.zip"
        with zipfile.ZipFile(archive) as z:
            for f in KAGGLE_FILES:
                z.extract(f, folder)
        archive.unlink()
    _from_drive(ORGANISERS["m5"], folder)


def fetch_dominicks(folder: Path, codes: list[str]) -> None:
    listed = json.loads(manifest.manifest_path("dominicks").read_text())["files"]
    names = [e["file"] for e in listed]
    if codes:
        names = [n for n in names if any(n in (f"w{c}.zip", f"w{c}_csv.zip", f"upc{c}.csv") for c in codes)]
        names.append("demo_stata.zip")
    for name in names:
        if name.startswith("upc"):
            _from_url(f"{KILTS}/upc_csv-files/{name}", folder / name)
        elif name.startswith("w"):
            _from_url(f"{KILTS}/movement_csv-files/{name}", folder / name)
        elif name == "demo_stata.zip":
            _from_url(f"{KILTS_STORES}/{name}", folder / name)
        elif name.endswith(".pdf"):
            _from_url(f"{KILTS}/{name}", folder / name)


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("m5", "m5_reference", "dominicks"):
        sys.exit(__doc__)
    dataset, extra = argv[0], argv[1:]
    folder = DATA_RAW / dataset
    folder.mkdir(parents=True, exist_ok=True)
    if dataset == "m5":
        fetch_m5(folder)
    elif dataset == "m5_reference":
        _from_drive(ORGANISERS["m5_reference"], folder)
    else:
        fetch_dominicks(folder, extra)
    checked = manifest.verify(dataset, only_present=bool(extra))
    print(f"{dataset}: {len(checked)} files match the registered checksums")


if __name__ == "__main__":
    main(sys.argv[1:])
