"""Paired dataset loader for the focusStep defocus-blur corpus.

Filenames follow

    focusStep_<level>_<font>_size_<pt>_sample_<idx>.tif

with a sibling `.txt` holding the ground-truth transcription.

An important constraint: the blurred folders do **not** contain the same page
re-blurred at each level. Each `focusStep_<level>` directory holds different
text content. Paired supervision is therefore only valid where an *exact*
filename match exists between the blurred directory and the focused reference
directory (`CAM01_focused`). Anything else silently pairs unrelated pages and
produces meaningless PSNR/SSIM. `FocusStepPairs` enforces the exact-match rule
and reports how many candidates it discarded.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

NAME_RE = re.compile(
    r"focusStep_(?P<level>\d+)_(?P<font>\w+?)_size_(?P<size>\d+)_sample_(?P<idx>\d+)"
)

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Sample:
    stem: str
    blurred: Path
    focused: Path
    level: int
    font: str

    @property
    def transcription(self) -> Path:
        return self.blurred.with_suffix(".txt")


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """16-bit TIFs are the norm in focusStep; normalise everything to uint8."""
    if arr.dtype == np.uint16:
        return (arr / 256).astype(np.uint8)
    if arr.dtype != np.uint8:
        span = arr.max() - arr.min() + 1e-8
        return ((arr - arr.min()) / span * 255).astype(np.uint8)
    return arr


def load_rgb(path: Path) -> np.ndarray:
    arr = to_uint8(np.array(Image.open(path)))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3]


def index_pairs(
    blurred_dir: Path,
    focused_dir: Path,
    levels: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> list[Sample]:
    """Build the exact-filename-match pair list."""
    pairs: list[Sample] = []
    unmatched = 0

    for path in sorted(blurred_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        meta = NAME_RE.match(path.stem)
        if meta is None:
            continue

        level = int(meta["level"])
        if levels is not None and level not in levels:
            continue

        focused = None
        for ext in IMAGE_EXTS:
            candidate = focused_dir / f"{path.stem}{ext}"
            if candidate.exists():
                focused = candidate
                break

        if focused is None:
            unmatched += 1
            continue

        pairs.append(
            Sample(
                stem=path.stem,
                blurred=path,
                focused=focused,
                level=level,
                font=meta["font"],
            )
        )

    if verbose:
        print(
            f"[focusstep] {len(pairs)} exact-match pair(s); "
            f"{unmatched} blurred image(s) had no focused counterpart and were dropped"
        )
    return pairs


class FocusStepPairs(Dataset):
    """Random crops of (blurred, focused) pairs for restoration training.

    Cropping rather than resizing is deliberate: defocus blur is a
    scale-dependent artifact, so downscaling a page changes the very degradation
    the model is meant to learn.
    """

    def __init__(
        self,
        blurred_dir: str | Path,
        focused_dir: str | Path,
        patch_size: int = 256,
        levels: tuple[int, ...] | None = None,
        augment: bool = True,
        crops_per_image: int = 4,
        seed: int = 0,
    ):
        self.samples = index_pairs(Path(blurred_dir), Path(focused_dir), levels)
        if not self.samples:
            raise RuntimeError(
                "No paired samples found. Check that blurred and focused "
                "directories share filenames - see docs/dataset.md."
            )
        self.patch_size = patch_size
        self.augment = augment
        self.crops_per_image = crops_per_image
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples) * self.crops_per_image

    def _crop(self, blur: np.ndarray, sharp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.patch_size
        h, w = blur.shape[:2]

        if h < p or w < p:
            pad_h, pad_w = max(0, p - h), max(0, p - w)
            blur = np.pad(blur, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            sharp = np.pad(sharp, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = blur.shape[:2]

        y = self.rng.randint(0, h - p)
        x = self.rng.randint(0, w - p)
        return blur[y : y + p, x : x + p], sharp[y : y + p, x : x + p]

    def _augment(self, blur: np.ndarray, sharp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            blur, sharp = blur[:, ::-1], sharp[:, ::-1]
        k = self.rng.randint(0, 3)
        if k:
            blur, sharp = np.rot90(blur, k), np.rot90(sharp, k)
        return np.ascontiguousarray(blur), np.ascontiguousarray(sharp)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index % len(self.samples)]

        blur = load_rgb(sample.blurred)
        sharp = load_rgb(sample.focused)

        # Blurred and focused captures can differ by a pixel or two at the edge.
        h = min(blur.shape[0], sharp.shape[0])
        w = min(blur.shape[1], sharp.shape[1])
        blur, sharp = blur[:h, :w], sharp[:h, :w]

        blur, sharp = self._crop(blur, sharp)
        if self.augment:
            blur, sharp = self._augment(blur, sharp)

        return {
            "blurred": torch.from_numpy(blur).permute(2, 0, 1).float() / 255.0,
            "focused": torch.from_numpy(sharp).permute(2, 0, 1).float() / 255.0,
            "stem": sample.stem,
            "level": sample.level,
        }
