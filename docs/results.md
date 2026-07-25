# Results

All figures below were produced by `scripts/evaluate.py` on the 56-page subset in
`dataset_results_34/`, and are reproducible from the committed CSVs in
`results/`.

```bash
python scripts/evaluate.py \
    --raw-dir dataset_results_34/raw \
    --enhanced-dir dataset_results_34/enhanced \
    --pred-dir dataset_results_34/output \
    --gt-dir dataset/raw
```

**Configuration.** Swin2SR 4× → EasyOCR detection → `trocr-base-printed`.
No restoration stage. 56 pages, blur levels 2–4, fonts `timesR` (32) and
`verdanaRef` (24). Inputs 2360×1460 16-bit TIF, outputs 9440×5840 RGB.

---

## 1. Restoration quality

| | SSIM | PSNR (dB) |
|---|---|---|
| Mean | **0.7320** | **31.77** |
| Min | 0.7166 | 31.09 |
| Max | 0.7566 | 32.43 |

By blur level:

| Level | n | SSIM | PSNR (dB) |
|---|---|---|---|
| 2 | 4 | 0.7301 | 31.14 |
| 3 | 40 | 0.7317 | 31.75 |
| 4 | 12 | 0.7336 | 32.04 |

By font:

| Font | n | SSIM | PSNR (dB) |
|---|---|---|---|
| `timesR` | 32 | 0.7357 | 31.95 |
| `verdanaRef` | 24 | 0.7270 | 31.53 |

### Read these numbers carefully

They are the reason this section comes with a warning rather than a conclusion.

**What they actually measure.** The comparison is the enhanced output against the
*bicubically upscaled blurred input* — not against a sharp reference. So they
quantify how much Swin2SR changed the image, not how close it got to the truth.
A high SSIM here is as consistent with "faithfully preserved the blur" as with
"restored the page".

**Why the trend is backwards.** SSIM and PSNR both *rise* slightly as blur
increases (level 2 → 4: 0.7301 → 0.7336, 31.14 → 32.04 dB). Under a fidelity
interpretation that is nonsense. Under the interpretation above it is exactly
what you would expect: a blurrier input has less high-frequency content for the
super-resolver to alter, so output and input agree more closely. The metric is
rewarding inaction.

**Why `timesR` scores higher than `verdanaRef`.** Same mechanism. Times is a
serif face with thin strokes and more fine detail; Verdana is a wide sans with
heavier, more uniform strokes that the upscaler modifies more. This says
something about the two fonts' interaction with the model, not about which is
read more accurately — and indeed the recognition numbers do not follow the same
ordering.

The fix is known and the code for it already exists: pair each blurred page with
its `CAM01_focused` counterpart by exact filename via `index_pairs()` in
`src/data/focusstep.py`. Wiring that into the evaluation path is the outstanding
work. Until then, treat this table as a sanity check on the SR stage, not as
evidence of restoration quality.

---

## 2. Recognition quality

196 tokens across 56 pages. Case-insensitive, alphanumerics only.

| Metric | Value |
|---|---|
| Token exact-match | **51.0%** |
| Character accuracy | **83.3%** |

By blur level:

| Level | Images | Tokens | Token exact-match | Character accuracy |
|---|---|---|---|---|
| 2 | 4 | 13 | 53.8% | 89.3% |
| 3 | 40 | 144 | **61.8%** | 87.5% |
| 4 | 12 | 39 | **10.3%** | 60.1% |

### This is the informative table

Recognition holds up through level 3 and then collapses. Exact match falls 61.8%
→ 10.3%; character accuracy falls 87.5% → 60.1%. The gap between those two
declines is itself diagnostic: at level 4 the model is still getting roughly
three characters in five right, so it is not producing noise — it is producing
*near misses*. And because a token is 10 characters, 60% character accuracy makes
an exact match almost arithmetically impossible. One error anywhere in the string
loses the token.

This is where the case for the restoration stage lives. The recogniser has not
stopped functioning at level 4; it is being starved of edge information. That is
precisely the deficit a fine-tuned deblurring model addresses, and it predicts
that Restormer should buy the most at level 4 and close to nothing at level 2 —
a testable claim, and the next experiment to run.

The level 2 row is weaker evidence than it looks: 4 images and 13 tokens. Its
apparently lower exact-match than level 3 is within noise at that sample size.

### Confounds worth naming

- **The language-model prior.** `trocr-base-printed` has a RoBERTa decoder
  trained on printed English text. Ground truth is random alphanumeric strings.
  The decoder's prior actively works against the task, so these figures
  understate what the same pipeline would achieve on natural text. A
  character-level or CTC-based recogniser would likely score better here while
  scoring worse on real documents.
- **Case.** TrOCR does not reliably preserve case on random strings, so all
  comparison is case-folded. A case-sensitive score would be substantially lower
  and would mostly be measuring an artifact.
- **Detection is not scored separately.** A missed region and a misread region
  both surface as recognition error. Token counts vary per page (196 tokens for
  what should be 168 if every page yielded exactly 3), which means the detector
  is over-segmenting some lines. Isolating detection recall from recognition
  accuracy is unfinished.

---

## 3. What is not measured yet

| Gap | Why it matters |
|---|---|
| Restoration against `CAM01_focused` | Current SSIM/PSNR cannot distinguish restoration from preservation |
| Restormer in the loop | The central hypothesis of the project is untested end-to-end in this repo |
| Ablation: with vs without deblurring | The one comparison that would justify or kill the restoration stage. `src/pipeline.py --no-restore` exists for exactly this |
| Levels 0–1 | Only levels 2–4 evaluated; the low-blur end would establish the ceiling |
| Detection recall in isolation | Currently entangled with recognition error |

The pipeline flags to run the third row already exist. That is the next
experiment, and it is the one that decides whether this project's premise holds.

---

## Reproducing

```bash
python scripts/evaluate.py --resume --time-budget 30    # restoration, resumable
python scripts/evaluate.py --skip-restoration           # recognition only, fast
```

Outputs:

- `results/restoration_metrics.csv` — per-image SSIM, PSNR, blur level, font
- `results/recognition_metrics.csv` — per-image token count, exact matches,
  token accuracy, character accuracy

The tiled SSIM implementation was verified against the original notebook's
whole-image computation and agrees to four decimal places on every overlapping
image.
