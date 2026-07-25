# Methodology

## The problem

A defocused camera acts as a low-pass filter. On natural scenes that costs you
texture. On text it costs you the thing that carries all the information: the
sharp transition at a glyph boundary. Once neighbouring strokes blur into each
other, no amount of contrast adjustment recovers the character — the distinction
between `B` and `8`, or `l` and `1`, lives entirely in high-frequency content
the blur has removed.

So the pipeline is built around one claim: on blurred documents, restoration is
not pre-processing, it is the substantive step. Everything downstream inherits
whatever the restoration stage did or failed to do.

---

## Stage 1 — Restoration with Restormer

### Why a transformer, and why this one

Convolutional restoration networks have a fixed receptive field per layer.
Defocus blur has a spatial extent that can exceed a character's width, so
recovering a stroke means aggregating evidence from further away than a small
kernel reaches. Attention gives that directly.

The obstacle is cost: standard spatial self-attention is O((HW)²), which is
hopeless on a 2360×1460 page. Restormer's contribution is to sidestep this:

- **MDTA** (Multi-Dconv Head Transposed Attention) computes attention across the
  *channel* dimension instead of the spatial one. The attention matrix is
  C×C rather than HW×HW, so cost becomes linear in pixel count. Depth-wise
  convolutions on Q, K and V restore the local spatial context that the
  transposition gives up.
- **GDFN** (Gated-Dconv Feed-Forward Network) splits the projected features into
  two depth-wise branches and gates one by the GELU of the other, letting the
  block suppress less informative features before they propagate.

The encoder-decoder is a 4-level U-Net with `num_blocks=(4,6,6,8)` and
`heads=(1,2,4,8)` at `dim=48`. Downsampling uses pixel-unshuffle rather than
strided convolution or pooling — resolution is traded for channels losslessly,
which matters when the signal you need is a thin edge. The network predicts a
residual added back to the input, so it learns the correction rather than
re-synthesising the page.

Implementation: `src/models/restormer.py`. Module names deliberately mirror the
official release so `single_image_defocus_deblurring.pth` loads without key
remapping.

### Why fine-tuning is not optional

The pretrained defocus checkpoint is trained on DPDD — dual-pixel natural
scenes. Applied unchanged to focusStep pages it produced **no PSNR improvement
and in fact a slight degradation**.

This is a domain mismatch, not a bug. Natural images have roughly 1/f spectra
with locally smooth regions. A text page is nearly bimodal in intensity, with
dense high-frequency edges spread uniformly across the frame. A model whose
learned priors say "smooth region with occasional edge" will regularise glyph
strokes toward the background.

The conclusion that shaped the project: **pretrained restoration weights are a
starting point for optimisation, not a drop-in component.**

### Training setup

`src/restoration/finetune_restormer.py`.

| Choice | Value | Reasoning |
|---|---|---|
| Loss | L1 | Penalises edge error more evenly than L2, which averages across a blurred stroke and is happy with a soft answer |
| Optimiser | AdamW, cosine decay | Standard for transformer fine-tuning |
| LR | 2e-5 | Two orders below from-scratch training: adapting a converged model, not searching |
| Patches | 256×256 random crops | Blur is scale-dependent, so resizing a page would alter the very degradation being learned. Cropping preserves it |
| Augmentation | Flips + 90° rotations | Safe here; no colour or blur augmentation, which would corrupt the target degradation |
| `--freeze-encoder` | optional | The encoder's low-level edge features already transfer. Adapting decoder + refinement only is faster and overfits less on a small corpus |
| Grad clipping | 1.0 | Fine-tuning at small batch size on high-variance crops |

Validation tracks L1 and PSNR against held-out crops; the best-PSNR checkpoint
is kept separately from the last.

### Full-page inference

`src/restoration/restore.py`. Pages are tiled with overlap and blended with a
linear ramp across the overlap region. A hard tile boundary would leave a visible
seam, and a seam falling across a glyph is a recognition error. Tiles are
reflect-padded to a multiple of 8 for the three pixel-unshuffle stages, then
cropped back.

---

## Stage 2 — Super-resolution with Swin2SR

Deblurring recovers edges but adds no resolution, and TrOCR's encoder wants text
that is physically large in pixels. Swin2SR at 4× supplies that.

`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr`, loaded from HuggingFace. The
HuggingFace route rather than the original release is a practical choice: it
avoids the `basicsr` / `realesrgan` dependency conflicts that made Real-ESRGAN
unusable in this project.

A 4× upscale of a 2360×1460 page is ~55 megapixels, so tiling is mandatory
(`src/enhancement/swin2sr.py`, `tile=256`; drop to 128 if a GPU still OOMs).
Tiles are reflect-padded to a multiple of 8 for the window attention and the
padded region trimmed at 4× scale afterwards.

`bicubic_baseline()` is provided in the same module for a reason: a learned
upscaler that does not beat bicubic is not earning its runtime.

---

## Stage 3 — Detection, and Stage 4 — Recognition

`src/recognition/ocr.py`. Detection and recognition are separated, the standard
STR decomposition:

**Detection** — EasyOCR loaded with `recognizer=False`, so only its
CRAFT-style detector runs. Its bundled recogniser is weaker than TrOCR on
printed text and loading it would waste memory. Boxes are padded slightly,
filtered by minimum area, then sorted into reading order using a row tolerance
of half the median box height — sorting purely by `y` scrambles words that sit
on the same line at slightly different heights.

**Recognition** — `microsoft/trocr-base-printed`, a ViT encoder with a RoBERTa
decoder. It decodes a whole line autoregressively, so unlike a CTC model there
is no per-frame alignment to get wrong. All crops from a page are batched into
one forward pass.

The architectural alternatives considered are the two reference papers:
**RARE** (Shi et al. 2016) adds a spatial transformer to rectify irregular text
before a sequence recogniser, and **attention-based CRNN** (Alshawi et al. 2024)
uses squeeze-and-excitation with CTC. Rectification is the one worth revisiting:
focusStep pages are axis-aligned so an STN buys nothing here, but it would
matter on photographed pages.

### The known weakness

TrOCR's decoder carries a language-model prior. focusStep ground truth is
*random* alphanumeric strings. The prior therefore actively fights the task —
the decoder wants to produce plausible English, and `UbuJkRjtQY` is not
plausible English. It also does not reliably preserve case on such strings.

Both consequences are handled in the evaluation rather than hidden:
comparison is case-insensitive, and character accuracy is reported alongside
exact match. See [results.md](results.md).

---

## Evaluation

`scripts/evaluate.py`, two independent families of metric.

**Restoration — SSIM and PSNR.** The raw image is bicubically upscaled to the
enhanced resolution first, so the comparison is pixel-aligned.

Two implementation notes. SSIM is computed tile-by-tile with a 32 px halo that
is computed then discarded at every seam, because a 55-megapixel float64 SSIM map
exhausts memory; this reproduces the whole-image values exactly (verified
against the notebook's original figures, agreeing to four decimal places). PSNR
is accumulated in int64, which is exact and needs no float image at all.

**Recognition — token exact-match and character accuracy.** Tokens are
normalised to lowercase alphanumerics. Character accuracy comes from
`difflib.SequenceMatcher` matching-block totals over ground-truth length, which
credits partial reads that exact match discards entirely.

The restoration pass supports `--resume` and `--time-budget` so a long run can
be split across invocations — each pair costs roughly 3 s and 500 MB peak.

### The measurement that needs fixing

The SSIM/PSNR reported here compare the **enhanced output against the bicubically
upscaled blurred input** — not against a sharp reference. They therefore measure
*how much Swin2SR changed the image*, not how close it got to the truth. A high
value can mean "faithfully preserved the blur".

The correct evaluation pairs each blurred page against its `CAM01_focused`
counterpart by exact filename. The loader in `src/data/focusstep.py` already
does this; wiring it into the evaluation path is the outstanding work, and it is
why [results.md](results.md) leans on the recognition numbers rather than the
restoration ones.
