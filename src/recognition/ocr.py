"""Stage 3: text detection and recognition.

Detection and recognition are split, which is the standard STR decomposition:

* **EasyOCR** is loaded with `recognizer=False`, so only its CRAFT-style
  detector runs. Its own recogniser is weaker than TrOCR on printed text, and
  loading it would waste memory.
* **TrOCR** (`microsoft/trocr-base-printed`) reads each detected crop. It is an
  encoder-decoder ViT+RoBERTa model, so it decodes a whole line at once rather
  than per character - no CTC alignment needed.

A caveat worth stating plainly: trocr-base-printed is trained on printed English
and carries a language-model prior in its decoder. The focusStep transcriptions
are random alphanumeric strings, which that prior actively fights. It also does
not reliably preserve case. Evaluation is therefore case-insensitive, and a
character-level accuracy is reported alongside exact match - see docs/results.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class OCRConfig:
    trocr_model: str = "microsoft/trocr-base-printed"
    languages: list[str] = field(default_factory=lambda: ["en"])
    device: str | None = None
    fp16: bool = True
    min_box_area: int = 64
    pad: int = 4
    max_new_tokens: int = 32
    num_beams: int = 1


class TextRecognizer:
    """EasyOCR detection followed by TrOCR recognition."""

    def __init__(self, config: OCRConfig | None = None):
        import easyocr  # imported lazily: pulls in its own heavy deps

        self.config = config or OCRConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = self.config.fp16 and self.device == "cuda"
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.detector = easyocr.Reader(
            self.config.languages, gpu=self.device == "cuda", recognizer=False
        )
        self.processor = TrOCRProcessor.from_pretrained(self.config.trocr_model)
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(
                self.config.trocr_model, torch_dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )
        print(f"[ocr] detector + {self.config.trocr_model} on {self.device}")

    # ------------------------------------------------------------------ #
    def detect(self, rgb: np.ndarray) -> list[Box]:
        """Axis-aligned boxes, sorted top-to-bottom then left-to-right."""
        height, width = rgb.shape[:2]
        raw = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # EasyOCR returns ([horizontal_boxes], [free_form_polys]) per image.
        horizontal = raw[0][0] if raw and raw[0] else []

        boxes: list[Box] = []
        pad = self.config.pad
        for x_min, x_max, y_min, y_max in horizontal:
            x1 = max(int(x_min) - pad, 0)
            x2 = min(int(x_max) + pad, width)
            y1 = max(int(y_min) - pad, 0)
            y2 = min(int(y_max) + pad, height)
            if (x2 - x1) * (y2 - y1) >= self.config.min_box_area:
                boxes.append((x1, y1, x2, y2))

        # Group into reading order with a row tolerance of half a box height.
        if boxes:
            tol = max(1, int(np.median([b[3] - b[1] for b in boxes]) * 0.5))
            boxes.sort(key=lambda b: (b[1] // tol, b[0]))
        return boxes

    @torch.no_grad()
    def recognise(self, rgb: np.ndarray, boxes: list[Box]) -> list[str]:
        """Read each crop. Batched so one forward pass covers the whole page."""
        if not boxes:
            return []

        crops = [Image.fromarray(rgb[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes]
        pixel_values = self.processor(images=crops, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        ids = self.model.generate(
            pixel_values,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
        )
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def read(self, rgb: np.ndarray) -> tuple[list[Box], list[str]]:
        boxes = self.detect(rgb)
        return boxes, self.recognise(rgb, boxes)


def draw_boxes(rgb: np.ndarray, boxes: list[Box], texts: list[str] | None = None) -> np.ndarray:
    """Overlay detections for visual inspection."""
    canvas = rgb.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if texts and i < len(texts):
            cv2.putText(
                canvas,
                texts[i],
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
    return canvas
