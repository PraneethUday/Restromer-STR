"""End-to-end pipeline: restore -> enhance -> detect -> recognise -> transcript.

    Restormer (deblur)  ->  Swin2SR (4x SR)  ->  EasyOCR (detect)  ->  TrOCR (read)

Each stage can be switched off, which is the point: the interesting question is
not whether the full stack works but how much each stage contributes. Running
with `--no-restore` and then with restoration enabled isolates the effect of the
deblurring model on downstream recognition accuracy.

Examples
--------
Full stack:
    python -m src.pipeline --input-dir dataset/raw --output-dir outputs/full \
        --restormer-weights checkpoints/restormer_focusstep_best.pth

Ablation, no deblurring (reproduces the Swin2SR-only baseline in docs/results.md):
    python -m src.pipeline --input-dir dataset/raw --output-dir outputs/no_restore \
        --no-restore
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.focusstep import IMAGE_EXTS, load_rgb

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--no-restore", action="store_true", help="skip Restormer")
    parser.add_argument("--no-enhance", action="store_true", help="skip Swin2SR")

    parser.add_argument("--restormer-weights", default=None)
    parser.add_argument("--restormer-tile", type=int, default=256)
    parser.add_argument("--restormer-overlap", type=int, default=32)

    parser.add_argument("--sr-tile", type=int, default=256)
    parser.add_argument("--sr-scale", type=int, default=4)

    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    restorer = None
    if not args.no_restore:
        import torch

        from src.models.restormer import Restormer, load_pretrained
        from src.restoration.restore import restore_image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = (
            load_pretrained(args.restormer_weights, device=device)
            if args.restormer_weights
            else Restormer().to(device).eval()
        )
        if not args.restormer_weights:
            print(
                "[warn] no --restormer-weights given; running an untrained "
                "Restormer. Results are diagnostic only."
            )

        def restorer(rgb: np.ndarray) -> np.ndarray:  # noqa: F811
            return restore_image(
                model,
                rgb,
                tile=args.restormer_tile,
                overlap=args.restormer_overlap,
                device=device,
            )

    enhancer = None
    if not args.no_enhance:
        from src.enhancement.swin2sr import Swin2SRConfig, Swin2SREnhancer

        enhancer = Swin2SREnhancer(
            Swin2SRConfig(scale=args.sr_scale, tile=args.sr_tile)
        )

    from src.recognition.ocr import TextRecognizer, draw_boxes

    reader = TextRecognizer()

    paths = [p for p in sorted(Path(args.input_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if args.limit:
        paths = paths[: args.limit]

    stages = [
        s
        for s, on in (
            ("restore", restorer is not None),
            ("enhance", enhancer is not None),
            ("detect+read", True),
        )
        if on
    ]
    print(f"{len(paths)} image(s)   stages: {' -> '.join(stages)}\n")

    manifest: list[dict] = []

    for i, path in enumerate(paths, 1):
        txt_path = text_dir / f"{path.stem}.txt"
        if txt_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(paths)}] skip {path.stem}")
            continue

        started = time.time()
        image = load_rgb(path)
        original_shape = image.shape[:2]

        if restorer is not None:
            image = restorer(image)
            if args.save_intermediate:
                (out_dir / "restored").mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(out_dir / "restored" / f"{path.stem}.png")

        if enhancer is not None:
            image = enhancer.enhance(image)
            if args.save_intermediate:
                (out_dir / "enhanced").mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(out_dir / "enhanced" / f"{path.stem}.png")

        boxes, lines = reader.read(image)
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        if args.save_overlay and boxes:
            (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
            Image.fromarray(draw_boxes(image, boxes, lines)).save(
                out_dir / "overlay" / f"{path.stem}.png"
            )

        elapsed = round(time.time() - started, 2)
        manifest.append(
            {
                "image": path.stem,
                "input_shape": list(original_shape),
                "final_shape": list(image.shape[:2]),
                "regions": len(boxes),
                "seconds": elapsed,
            }
        )
        print(f"  [{i}/{len(paths)}] {path.stem}: {len(boxes)} region(s), {elapsed}s")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nTranscripts in {text_dir}/   manifest at {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
