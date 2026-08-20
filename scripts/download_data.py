"""Download and unpack the APTOS 2019 dataset from Kaggle.

The signed Google-Cloud URL originally supplied with the project expired, so we
use the official Kaggle API instead. This requires a one-time setup:

  1. Create a free Kaggle account and, on the competition page, click
     "Join competition" / accept the rules:
     https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
  2. Account -> Settings -> API -> "Create New Token". This downloads
     ``kaggle.json``. Place it at:
         Windows:  C:\\Users\\<you>\\.kaggle\\kaggle.json
         Linux/Mac: ~/.kaggle/kaggle.json
  3. Run:  python scripts/download_data.py

The archive is ~10 GB; expect a few minutes on a fast connection.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR  # noqa: E402

COMPETITION = "aptos2019-blindness-detection"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{COMPETITION}.zip"

    if not zip_path.exists():
        print(f"Downloading {COMPETITION} to {DATA_DIR} ...")
        # Uses the kaggle CLI; will raise a clear error if kaggle.json is missing
        # or the competition rules have not been accepted.
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "competitions", "download",
                "-c", COMPETITION, "-p", str(DATA_DIR),
            ],
            check=True,
        )
    else:
        print(f"Archive already present: {zip_path}")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)

    n_imgs = len(list((DATA_DIR / "train_images").glob("*.png")))
    print(f"Done. Found {n_imgs} training images in {DATA_DIR / 'train_images'}.")
    if n_imgs == 0:
        print("WARNING: no PNGs found - check the archive contents.", file=sys.stderr)


if __name__ == "__main__":
    main()
