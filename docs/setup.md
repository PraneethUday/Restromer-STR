# Setup

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended. Restormer fine-tuning is impractical on
  CPU, and a 4× Swin2SR pass over a full page takes minutes rather than seconds.
- ~4 GB disk for model weights, plus whatever the dataset needs (~2 GB for the
  working set used here).

## Install

```bash
git clone https://github.com/PraneethUday/Restromer-STR.git
cd Restromer-STR

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it first from
<https://pytorch.org/get-started/locally/> and then run the requirements file —
`torch` will already be satisfied.

`easyocr` pulls a large dependency tree. If it conflicts with your environment,
note that it is only needed for detection; `src/enhancement/swin2sr.py` and the
restoration stage do not import it.

## Pretrained weights

The restoration stage fine-tunes from the official Restormer defocus checkpoint.
Download `single_image_defocus_deblurring.pth` from the
[Restormer release page](https://github.com/swz30/Restormer) and place it at:

```
weights/single_image_defocus_deblurring.pth
```

Swin2SR and TrOCR download automatically from HuggingFace on first use and cache
under `~/.cache/huggingface`.

`weights/`, `checkpoints/` and `*.pth` are gitignored.

## Dataset

Not distributed with the repository. See [dataset.md](dataset.md) for the
naming scheme and the directory layout the code expects. The short version:

```
dataset/
├── focusStep/
│   ├── blurred/            for training the restoration stage
│   └── CAM01_focused/      sharp references, matched by exact filename
└── raw/                    flat working set for the recognition pipeline
```

---

## Running the stages

### 1. Fine-tune Restormer

```bash
python -m src.restoration.finetune_restormer \
    --blurred-dir dataset/focusStep/blurred \
    --focused-dir dataset/focusStep/CAM01_focused \
    --pretrained weights/single_image_defocus_deblurring.pth \
    --epochs 30 --batch-size 4 --patch-size 256 \
    --freeze-encoder
```

Useful flags:

| Flag | Effect |
|---|---|
| `--freeze-encoder` | Adapt decoder + refinement only. Faster, less overfitting on a small corpus |
| `--levels 1 2 3` | Restrict training to specific blur levels |
| `--crops-per-image N` | Random crops drawn per source page per epoch |
| `--lr` | Default 2e-5. Raise only if training from scratch |

Writes `restormer_focusstep_best.pth`, `restormer_focusstep_last.pth` and
`history.json` to `--out-dir` (default `checkpoints/`).

If the pair count printed at startup is 0, or the "no focused counterpart" count
is high, your directory layout is wrong — re-read [dataset.md](dataset.md).

### 2. Restore pages

```bash
python -m src.restoration.restore \
    --input-dir dataset/raw \
    --output-dir outputs/restored \
    --weights checkpoints/restormer_focusstep_best.pth \
    --tile 256 --overlap 32
```

Lower `--tile` if you hit OOM. Raise `--overlap` if you see tile seams.

### 3. Full pipeline

```bash
python -m src.pipeline \
    --input-dir dataset/raw \
    --output-dir outputs/full \
    --restormer-weights checkpoints/restormer_focusstep_best.pth \
    --save-intermediate --save-overlay
```

Produces `outputs/full/text/<page>.txt` plus a `manifest.json` recording region
counts and per-page timing. `--save-overlay` writes detection boxes drawn on the
page, which is the fastest way to tell a detection failure from a recognition
failure.

Ablations:

```bash
python -m src.pipeline --input-dir ... --output-dir ... --no-restore   # skip Restormer
python -m src.pipeline --input-dir ... --output-dir ... --no-enhance   # skip Swin2SR
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --pred-dir outputs/full/text \
    --gt-dir dataset/raw \
    --out-dir results
```

The restoration pass is slow (~3 s and ~500 MB peak per 4× page). Split it:

```bash
python scripts/evaluate.py --resume --time-budget 30 --skip-recognition
```

Re-run until the pair count stops rising. Recognition scoring alone is fast:

```bash
python scripts/evaluate.py --skip-restoration
```

---

## Troubleshooting

**CUDA out of memory during Swin2SR.** Lower `--sr-tile` to 128. The tile size
governs peak memory almost entirely.

**`Image size exceeds limit` / DecompressionBombWarning.** Already handled —
every module that opens an image sets `Image.MAX_IMAGE_PIXELS = None`. If you see
it, you are loading through your own code path.

**Restormer checkpoint reports missing or unexpected keys.** `load_pretrained`
prints counts rather than failing. A handful of `unexpected` keys is normal for
BasicSR-style archives; a large `missing` count means the architecture config
does not match the checkpoint — check `dim`, `num_blocks` and `heads`.

**Evaluation gets killed.** Out of memory on the 55-megapixel SSIM. Use
`--resume --time-budget`; the tiled implementation is already memory-bounded, but
the process still holds two full-page arrays.

**Zero pairs found in training.** Blurred and focused filenames must match
*exactly*. See the pairing constraint in [dataset.md](dataset.md).
