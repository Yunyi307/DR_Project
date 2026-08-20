"""Pre-render preprocessed fundus images to disk (one-time, per variant).

On-the-fly Ben Graham enhancement runs a Gaussian blur on full-resolution images
(APTOS images are up to ~3000px wide), which starves the GPU during training.
Rendering the crop + enhancement once to a fixed 512px cache turns each training
epoch into cheap file reads + augmentation, keeping the GPU fed. All downstream
experiments (CNN, ViT, Swin, ablations) reuse the same cache.

Variants:
  full : crop -> resize -> CLAHE -> Ben Graham -> circular mask  (main pipeline)
  raw  : crop -> resize -> circular mask                         (for the E4 ablation)

Usage:
    python scripts/cache_preprocess.py --variant full
    python scripts/cache_preprocess.py --variant raw
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, TRAIN_CSV, TRAIN_IMAGES_DIR  # noqa: E402
from src.preprocessing import (  # noqa: E402
    apply_clahe, ben_graham, circular_mask, crop_to_roi,
)

CACHE_ROOT = DATA_DIR.parent / "processed"


def process_one(id_code: str, variant: str, size: int, out_dir: Path) -> bool:
    src = TRAIN_IMAGES_DIR / f"{id_code}.png"
    dst = out_dir / f"{id_code}.png"
    if dst.exists():
        return True
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = crop_to_roi(rgb)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    if variant == "full":
        out = apply_clahe(out)
        out = ben_graham(out)
    out = circular_mask(out)
    cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "raw"], default="full")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = CACHE_ROOT / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(TRAIN_CSV)["id_code"].tolist()
    print(f"Caching {len(ids)} images -> {out_dir}  (variant={args.variant}, {args.size}px)")

    worker = partial(process_one, variant=args.variant, size=args.size, out_dir=out_dir)
    ok = 0
    with Pool(args.workers) as pool:
        for success in tqdm(pool.imap_unordered(worker, ids, chunksize=16), total=len(ids)):
            ok += int(success)
    print(f"Done: {ok}/{len(ids)} written to {out_dir}"
          + ("" if ok == len(ids) else "  (some images failed to read!)"))


if __name__ == "__main__":
    main()
