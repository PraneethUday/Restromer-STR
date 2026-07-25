"""Stage 2: Swin2SR 4x super-resolution.

Restormer removes the defocus blur but does not add resolution, and TrOCR wants
text that is physically large in pixels. Swin2SR at 4x supplies that.

Loaded from HuggingFace (`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr`) rather
than the original release, which avoids the `basicsr` / `realesrgan` dependency
tangle that made Real-ESRGAN unworkable in this project.

Tiling is mandatory here: a 4x upscale of a 2360x1460 page is ~55 megapixels, so
the full tensor never fits comfortably in memory. Tiles are padded to a multiple
of 8 (the window requirement) before being fed to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MODEL = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"


@dataclass
class Swin2SRConfig:
    model_id: str = DEFAULT_MODEL
    scale: int = 4
    tile: int = 256          # 256 is safe on a 15 GB GPU; drop to 128 if OOM
    device: str | None = None
    fp16: bool = True


class Swin2SREnhancer:
    """4x super-resolution with memory-bounded tiling."""

    def __init__(self, config: Swin2SRConfig | None = None):
        self.config = config or Swin2SRConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        use_fp16 = self.config.fp16 and self.device == "cuda"
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.processor = Swin2SRImageProcessor()
        self.model = (
            Swin2SRForImageSuperResolution.from_pretrained(
                self.config.model_id, torch_dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )
        print(f"[swin2sr] {self.config.model_id} on {self.device} ({self.dtype})")

    @torch.no_grad()
    def _upscale_tile(self, tile: np.ndarray) -> np.ndarray:
        pad_h = (8 - tile.shape[0] % 8) % 8
        pad_w = (8 - tile.shape[1] % 8) % 8
        if pad_h or pad_w:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        inputs = self.processor(images=Image.fromarray(tile), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        out = self.model(pixel_values=pixel_values).reconstruction
        out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)

        # Trim the region that corresponds to the reflect padding.
        s = self.config.scale
        keep_h = out.shape[0] - pad_h * s
        keep_w = out.shape[1] - pad_w * s
        return out[:keep_h, :keep_w]

    @torch.no_grad()
    def enhance(self, rgb: np.ndarray) -> np.ndarray:
        """Upscale a uint8 RGB image by `scale`, tile by tile."""
        height, width = rgb.shape[:2]
        s = self.config.scale
        tile = self.config.tile

        output = np.zeros((height * s, width * s, 3), dtype=np.uint8)

        for y in range(0, height, tile):
            for x in range(0, width, tile):
                y1 = min(y + tile, height)
                x1 = min(x + tile, width)

                up = self._upscale_tile(rgb[y:y1, x:x1])
                output[y * s : y1 * s, x * s : x1 * s] = up[
                    : (y1 - y) * s, : (x1 - x) * s
                ]

                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return output


def bicubic_baseline(rgb: np.ndarray, scale: int = 4) -> np.ndarray:
    """Reference upscale. Any SR result should beat this to be worth its cost."""
    return cv2.resize(
        rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_CUBIC
    )
