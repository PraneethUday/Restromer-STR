"""Stage 1 inference: run Restormer over full-resolution pages.

Document pages are far too large to push through the network in one piece, so
inference is tiled with an overlap. Tiles are blended with a linear ramp in the
overlap region; a hard seam would otherwise show up as a visible vertical or
horizontal line right where a glyph might sit.

Restormer also requires spatial dimensions divisible by 8 (three pixel-unshuffle
downsamples), which is handled by reflect-padding and cropping back.

Example
-------
    python -m src.restoration.restore \
        --input-dir dataset/focusStep/blurred \
        --output-dir outputs/restored \
        --weights checkpoints/restormer_focusstep_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.focusstep import IMAGE_EXTS, load_rgb
from src.models.restormer import Restormer, load_pretrained

Image.MAX_IMAGE_PIXELS = None

MULTIPLE = 8


def pad_to_multiple(x: torch.Tensor, multiple: int = MULTIPLE) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def _ramp(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    """Linear blend weights along one axis of a tile."""
    w = np.ones(length, dtype=np.float32)
    if overlap > 0:
        taper = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        if not at_start:
            w[:overlap] = taper
        if not at_end:
            w[-overlap:] = taper[::-1]
    return w


@torch.no_grad()
def restore_image(
    model: Restormer,
    image: np.ndarray,
    tile: int = 256,
    overlap: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Tiled, overlap-blended restoration of a uint8 RGB page."""
    height, width = image.shape[:2]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    stride = max(tile - overlap, 1)

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y1 = min(y + tile, height)
            x1 = min(x + tile, width)
            y0 = max(y1 - tile, 0)
            x0 = max(x1 - tile, 0)

            patch = tensor[:, :, y0:y1, x0:x1].to(device)
            patch, ph, pw = pad_to_multiple(patch)
            out = model(patch).clamp(0, 1)[:, :, :ph, :pw]

            out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

            wy = _ramp(y1 - y0, overlap, y0 == 0, y1 == height)
            wx = _ramp(x1 - x0, overlap, x0 == 0, x1 == width)
            tile_w = (wy[:, None] * wx[None, :])[:, :, None]

            accum[y0:y1, x0:x1] += out_np * tile_w
            weight[y0:y1, x0:x1] += tile_w

            if x1 == width:
                break
        if y1 == height:
            break

    blended = accum / np.maximum(weight, 1e-6)
    return (blended * 255.0).round().clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--weights",
        default=None,
        help="fine-tuned checkpoint; omit to run an untrained model (diagnostic only)",
    )
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   tile={args.tile} overlap={args.overlap}")

    model = (
        load_pretrained(args.weights, device=device)
        if args.weights
        else Restormer().to(device).eval()
    )

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} image(s) to restore")

    for i, path in enumerate(paths, 1):
        out_path = out_dir / f"{path.stem}_restored.png"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(paths)}] skip {path.stem}")
            continue

        restored = restore_image(
            model, load_rgb(path), tile=args.tile, overlap=args.overlap, device=device
        )
        Image.fromarray(restored).save(out_path)
        print(f"  [{i}/{len(paths)}] {path.stem} -> {out_path.name}")

    print(f"\nRestored pages in {out_dir}/")


if __name__ == "__main__":
    main()
