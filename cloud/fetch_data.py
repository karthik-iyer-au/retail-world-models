"""Download a data set's raw files from the bridge (cloud/vast_app.py) in parts.

  BRIDGE_TOKEN=... python cloud/fetch_data.py <bridge url> <data set> <folder>

Used by the Vast and NUC workers. A Modal web request is cut off after about
150 seconds, and Favorita's train.csv (4.7 GB) took longer than that in one
request (2 November 2026: cut at 152 s on one machine, and at 1,775 s by a
restart of the bridge on another). So each file comes in 64 MB parts, six at a
time, each part retried. A file already in the folder at the right size is
kept; the run checks every file against its checksum anyway.
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PART = 64 << 20
AT_ONCE = 6


def get(url: str, token: str, tries: int = 8) -> bytes:
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # a cut connection or a restarting bridge: wait and ask again
            if t == tries - 1:
                raise
            print(f"retry {t + 1} of {url.rsplit('/', 1)[-1]}: {e}", flush=True)
            time.sleep(10 * (t + 1))
    raise RuntimeError("unreachable")


def main(bridge: str, dataset: str, folder: str) -> None:
    token = os.environ["BRIDGE_TOKEN"]
    dest = Path(folder)
    dest.mkdir(parents=True, exist_ok=True)
    for name in get(f"{bridge}/data/{dataset}", token).decode().split():
        size = int(get(f"{bridge}/size/{dataset}/{name}", token))
        out = dest / name
        if out.exists() and out.stat().st_size == size:
            continue
        tmp = dest / (name + ".part")
        with open(tmp, "wb") as f:
            f.truncate(size)

        def one(start: int) -> None:
            want = min(PART, size - start)
            for _ in range(3):
                b = get(f"{bridge}/part/{dataset}/{name}?start={start}&length={want}", token)
                if len(b) == want:
                    with open(tmp, "r+b") as f:
                        f.seek(start)
                        f.write(b)
                    return
            raise IOError(f"{name}: part at {start} came back short")

        started = time.time()
        with ThreadPoolExecutor(AT_ONCE) as pool:
            list(pool.map(one, range(0, size, PART)))
        tmp.rename(out)
        print(f"{name}: {size / 1e6:.0f} MB in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
