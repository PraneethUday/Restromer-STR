# Dataset — focusStep

Synthetically defocused document pages. Each page is a block of random
alphanumeric strings rendered in a known font, then captured or simulated at a
series of focus settings.

## Naming

```
focusStep_<level>_<font>_size_<pt>_sample_<idx>.tif   image
focusStep_<level>_<font>_size_<pt>_sample_<idx>.txt   ground-truth transcription
```

| Field | Values |
|---|---|
| `level` | 0–4, increasing defocus (0 = sharpest) |
| `font` | `timesR`, `verdanaRef` |
| `size` | 30 pt |
| `idx` | 0001–0020, 20 samples per font per level |

Images are **2360 × 1460, 16-bit grayscale TIF** (`I;16` in Pillow). Everything
in `src/data/focusstep.py` normalises to 8-bit RGB on load — `to_uint8` divides
by 256 for `uint16` and min-max stretches anything else.

Ground truth is three whitespace-separated 10-character alphanumeric tokens per
page, e.g.

```
UbuJkRjtQY
VFhh3nrV98
fMEmU8cgR4
```

That the tokens are random is deliberate for measuring raw character fidelity,
but it works against any recogniser carrying a language-model prior. See
[results.md](results.md).

## The pairing constraint

**The blurred directories do not contain the same page re-blurred at each
level.** Each `focusStep_<level>` group holds *different text content*. So a
paired PSNR/SSIM evaluation is only valid where an exact filename match exists
between a blurred image and its focused reference in `CAM01_focused`.

Pairing by index, or by position in a sorted listing, silently matches unrelated
pages and produces numbers that look plausible and mean nothing. This is the
single easiest way to get this project wrong.

`index_pairs()` in `src/data/focusstep.py` enforces exact-stem matching and
prints how many blurred images it discarded for lack of a counterpart. If that
count is high, the directory layout is wrong.

## Expected layout

For training the restoration stage:

```
dataset/
└── focusStep/
    ├── blurred/                 focusStep_1..4_*.tif  + .txt
    └── CAM01_focused/           focusStep_*_*.tif     sharp references
```

For running the recognition pipeline on the flat working set used here:

```
dataset/
├── raw/                         200 images + 200 .txt ground truth  (~1.3 GB)
├── enhanced/                    Swin2SR output, written by the pipeline
└── output/                      predicted transcriptions, one .txt per page
```

The evaluation run reported in [results.md](results.md) used a separate
subset directory:

```
dataset_results_34/
├── raw/                         56 blurred images, levels 2–4     (371 MB)
├── enhanced/                    56 × 9440×5840 PNG, 4× upscaled   (475 MB)
└── output/                      56 predicted transcriptions
```

## Size and version control

`dataset/` (1.3 GB) and `dataset_results_*/` (849 MB) are gitignored. GitHub
rejects single files over 100 MB and warns well before a repository reaches this
size; the 4× upscaled PNGs alone are ~8.5 MB each.

What *is* committed: per-image metric CSVs under `results/`, which are a few
kilobytes and let anyone re-derive every number in the docs without the imagery.
