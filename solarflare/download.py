"""Download the Cleaned-SWANSF .pkl partitions.

The repository stores the actual arrays on Google Drive (the GitHub repo only
holds a download.txt pointer, code, and docs). This script:
  * clones/pulls the repo to read the current Drive link, and
  * uses `gdown` to pull the folder into data/swansf/.

If automated download fails (Drive quota / interstitial), it prints the manual
steps. Training then just needs the .pkl files sitting in data/swansf/.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

from .config import load_config

REPO = "https://github.com/samresume/Cleaned-SWANSF-Dataset"
RAW_DOWNLOAD_TXT = (
    "https://raw.githubusercontent.com/samresume/Cleaned-SWANSF-Dataset/main/download.txt"
)


def _drive_link(cfg) -> str | None:
    import requests
    try:
        txt = requests.get(RAW_DOWNLOAD_TXT, timeout=20).text
        m = re.search(r"https?://drive\.google\.com/\S+", txt)
        return m.group(0) if m else None
    except Exception:                              # noqa: BLE001
        return None


def main():
    cfg = load_config()
    dest = cfg["paths"]["data_dir"]
    os.makedirs(dest, exist_ok=True)

    link = _drive_link(cfg)
    print(f"Target directory : {dest}")
    print(f"Drive link found : {link or '(none — see manual steps below)'}")

    if link:
        try:
            import gdown
            print("Downloading with gdown ...")
            if "/folders/" in link:
                gdown.download_folder(url=link, output=dest, quiet=False, use_cookies=False)
            else:
                gdown.download(url=link, output=dest, quiet=False, fuzzy=True)
            print("Done. Verifying ...")
        except Exception as exc:                   # noqa: BLE001
            print(f"Automated download failed: {exc}")
            link = None

    pkls = [f for f in os.listdir(dest) if f.endswith(".pkl")] if os.path.isdir(dest) else []
    if pkls:
        print(f"Found {len(pkls)} .pkl files in {dest}. Ready to train.")
        return

    print("\n" + "=" * 64)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 64)
    print(f"1. Open {REPO}")
    print("2. Read download.txt -> open the Google Drive link.")
    print(f"3. Download all Partition*.pkl files into:\n     {dest}")
    print("4. Re-run training:  python -m solarflare.train")
    sys.exit(1)


if __name__ == "__main__":
    main()
