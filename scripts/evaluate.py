#!/usr/bin/env python3
"""
Evaluation script for the SceneTextRecognizer pipeline.

Computes two families of metrics:

1. Restoration quality  - SSIM and PSNR between each raw (defocus-blurred) image
                          and its Swin2SR-enhanced counterpart. The raw image is
                          bicubically upscaled to the enhanced resolution first so
                          the comparison is pixel-aligned.

2. Recognition quality  - token-level exact-match accuracy and character accuracy
                          between the TrOCR predictions and the focusStep ground
                          truth transcriptions.

Both are written to CSV under results/.

Usage
-----
    python scripts/evaluate.py \
        --raw-dir dataset_results_34/raw \
        --enhanced-dir dataset_results_34/enhanced \
        --pred-dir dataset_results_34/output \
        --gt-dir dataset/raw \
        --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# The enhanced images are 4x upscaled (~55 megapixels), well past Pillow's
# decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None

# SSIM is evaluated on tiles so peak memory stays bounded. HALO pixels are
# computed but discarded at every seam so the windowed statistics match a
# whole-image evaluation.
TILE = 1024
HALO = 32


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_raw_gray(path: os.PathLike) -> np.ndarray:
    """Load a 16-bit TIF (or any image) as a uint8 luminance array."""
    arr = np.array(Image.open(path))

    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)
    elif arr.dtype != np.uint8:
        span = arr.max() - arr.min() + 1e-8
        arr = ((arr - arr.min()) / span * 255).astype(np.uint8)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    return arr


def load_enhanced_gray(path: os.PathLike) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


# --------------------------------------------------------------------------- #
# Restoration metrics
# --------------------------------------------------------------------------- #
def tiled_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the whole image, computed tile by tile."""
    height, width = a.shape
    total, count = 0.0, 0

    for y in range(0, height, TILE):
        for x in range(0, width, TILE):
            y0, y1 = max(y - HALO, 0), min(y + TILE + HALO, height)
            x0, x1 = max(x - HALO, 0), min(x + TILE + HALO, width)

            tile_a = a[y0:y1, x0:x1].astype(np.float32)
            tile_b = b[y0:y1, x0:x1].astype(np.float32)
            if min(tile_a.shape) < 7:  # smaller than the SSIM window
                continue

            _, ssim_map = ssim(tile_a, tile_b, data_range=255, full=True)

            # Discard the halo so seams are not double counted.
            top = y - y0
            left = x - x0
            interior = ssim_map[
                top : top + min(TILE, height - y), left : left + min(TILE, width - x)
            ]
            total += float(interior.sum())
            count += interior.size

    return total / max(count, 1)


def exact_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR accumulated in int64 so no intermediate float image is needed."""
    diff = a.astype(np.int32) - b.astype(np.int32)
    mse = float(np.sum(diff.astype(np.int64) ** 2)) / diff.size
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def restoration_metrics(
    raw_dir: Path,
    enhanced_dir: Path,
    done: set[str] | None = None,
    time_budget: float | None = None,
) -> list[dict]:
    """SSIM / PSNR for every (raw, enhanced) pair matched by filename stem.

    Each pair costs ~3 s and ~500 MB of peak memory at 4x of a 2360x1460 page,
    so `done` (stems already measured) and `time_budget` (seconds) let a long
    run be split across several invocations.
    """
    rows: list[dict] = []
    done = done or set()
    started = time.monotonic()

    for fname in sorted(os.listdir(enhanced_dir)):
        if not fname.endswith("_enhanced.png"):
            continue

        stem = fname.replace("_enhanced.png", "")
        if stem in done:
            continue

        raw_path = raw_dir / f"{stem}.tif"
        if not raw_path.exists():
            continue

        if time_budget is not None and time.monotonic() - started > time_budget:
            print(f"  time budget reached, {len(rows)} pair(s) this pass")
            break

        enh = load_enhanced_gray(enhanced_dir / fname)
        raw = load_raw_gray(raw_path)

        # Upscale raw to the enhanced resolution so SSIM/PSNR are well defined.
        raw_up = cv2.resize(
            raw, (enh.shape[1], enh.shape[0]), interpolation=cv2.INTER_CUBIC
        )

        rows.append(
            {
                "image": stem,
                "focus_step": stem.split("_")[1],
                "font": stem.split("_")[2],
                "ssim": round(tiled_ssim(raw_up, enh), 4),
                "psnr_db": round(exact_psnr(raw_up, enh), 2),
            }
        )
        print(f"  [{len(rows):3d}] {stem}  ssim={rows[-1]['ssim']}  psnr={rows[-1]['psnr_db']}")

        del raw, raw_up, enh

    return rows


# --------------------------------------------------------------------------- #
# Recognition metrics
# --------------------------------------------------------------------------- #
def normalise(token: str) -> str:
    """Case- and punctuation-insensitive normalisation.

    trocr-base-printed is trained on printed English and does not reliably
    preserve case on random alphanumeric strings, so case is folded before
    comparison. Reported numbers are therefore case-insensitive.
    """
    return re.sub(r"[^a-z0-9]", "", token.lower())


def recognition_metrics(pred_dir: Path, gt_dir: Path) -> tuple[list[dict], dict]:
    """Token exact-match and character accuracy per image."""
    rows: list[dict] = []
    total_tokens = total_exact = 0
    matched_chars = total_chars = 0

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".txt"):
            continue

        gt_path = gt_dir / fname
        if not gt_path.exists():
            continue

        pred = [t for t in map(normalise, (pred_dir / fname).read_text().split()) if t]
        gt = [t for t in map(normalise, gt_path.read_text().split()) if t]

        n = exact = 0
        img_matched = img_chars = 0

        for p, g in zip(pred, gt):
            n += 1
            exact += p == g
            matcher = difflib.SequenceMatcher(None, p, g)
            img_matched += sum(b.size for b in matcher.get_matching_blocks())
            img_chars += len(g)

        if n == 0:
            continue

        total_tokens += n
        total_exact += exact
        matched_chars += img_matched
        total_chars += img_chars

        rows.append(
            {
                "image": fname[:-4],
                "tokens": n,
                "exact_match": exact,
                "token_accuracy": round(exact / n, 4),
                "char_accuracy": round(img_matched / max(img_chars, 1), 4),
            }
        )

    summary = {
        "images": len(rows),
        "tokens": total_tokens,
        "token_accuracy": round(total_exact / max(total_tokens, 1), 4),
        "char_accuracy": round(matched_chars / max(total_chars, 1), 4),
    }
    return rows, summary


# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict], append: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a" if append else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not (append and exists):
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="dataset_results_34/raw")
    parser.add_argument("--enhanced-dir", default="dataset_results_34/enhanced")
    parser.add_argument("--pred-dir", default="dataset_results_34/output")
    parser.add_argument("--gt-dir", default="dataset/raw")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip pairs already present in restoration_metrics.csv and append",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="stop the restoration pass after N seconds (use with --resume)",
    )
    parser.add_argument("--skip-restoration", action="store_true")
    parser.add_argument("--skip-recognition", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rest_csv = out_dir / "restoration_metrics.csv"

    if not args.skip_restoration:
        previous = read_csv(rest_csv) if args.resume else []
        done = {r["image"] for r in previous}

        fresh = restoration_metrics(
            Path(args.raw_dir),
            Path(args.enhanced_dir),
            done=done,
            time_budget=args.time_budget,
        )
        write_csv(rest_csv, fresh, append=args.resume)

        rest = previous + fresh
        if rest:
            ssim_vals = [float(r["ssim"]) for r in rest]
            psnr_vals = [float(r["psnr_db"]) for r in rest]
            print(f"Restoration  — {len(rest)} pairs measured so far")
            print(
                f"  SSIM  mean {np.mean(ssim_vals):.4f} "
                f"[{np.min(ssim_vals):.4f}, {np.max(ssim_vals):.4f}]"
            )
            print(
                f"  PSNR  mean {np.mean(psnr_vals):.2f} dB "
                f"[{np.min(psnr_vals):.2f}, {np.max(psnr_vals):.2f}]"
            )

    if not args.skip_recognition:
        rec, summary = recognition_metrics(Path(args.pred_dir), Path(args.gt_dir))
        write_csv(out_dir / "recognition_metrics.csv", rec)

        if rec:
            print(f"\nRecognition  — {summary['images']} images, {summary['tokens']} tokens")
            print(f"  Token exact-match : {summary['token_accuracy'] * 100:.1f}%")
            print(f"  Character accuracy: {summary['char_accuracy'] * 100:.1f}%")

    print(f"\nCSVs written to {out_dir}/")


if __name__ == "__main__":
    main()
