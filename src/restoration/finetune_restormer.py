"""Fine-tune Restormer on the focusStep document corpus.

Why fine-tune at all: the official `single_image_defocus_deblurring.pth`
checkpoint is trained on the DPDD natural-scene set. Applied unchanged to text
pages it gives no benefit and in fact slightly *degrades* PSNR - the statistics
of dense high-frequency glyph edges are not the statistics of natural scenes.
Fine-tuning on paired focusStep data is what makes the restoration stage useful.

Training recipe
---------------
* L1 (Charbonnier-free) loss on 256px random crops - L1 preserves edges better
  than L2 on text, which matters far more here than in natural-image restoration.
* AdamW with cosine decay, low LR (2e-5) because we are adapting a converged
  model rather than training from scratch.
* Optional encoder freeze: the encoder already extracts good low-level edge
  features, so adapting only the decoder + refinement is faster and less prone
  to overfitting on a small corpus.
* Mixed precision on CUDA.

Example
-------
    python -m src.restoration.finetune_restormer \
        --blurred-dir dataset/focusStep/blurred \
        --focused-dir dataset/focusStep/CAM01_focused \
        --pretrained weights/single_image_defocus_deblurring.pth \
        --epochs 30 --batch-size 4 --freeze-encoder
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.focusstep import FocusStepPairs
from src.models.restormer import Restormer, load_pretrained


def psnr_from_mse(mse: torch.Tensor) -> float:
    if mse.item() <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse.item())


def build_model(args, device: str) -> Restormer:
    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained}")
        model = load_pretrained(args.pretrained, device=device)
    else:
        print("Training from random initialisation")
        model = Restormer().to(device)

    if args.freeze_encoder:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(("patch_embed", "encoder_level", "down")):
                param.requires_grad = False
                frozen += param.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"Froze encoder: {frozen / 1e6:.2f}M / {total / 1e6:.2f}M params")

    return model.train()


@torch.no_grad()
def validate(model: Restormer, loader: DataLoader, device: str) -> dict:
    model.eval()
    l1 = nn.L1Loss()
    total_l1 = total_mse = 0.0
    batches = 0

    for batch in loader:
        blur = batch["blurred"].to(device)
        sharp = batch["focused"].to(device)
        out = model(blur).clamp(0, 1)

        total_l1 += l1(out, sharp).item()
        total_mse += ((out - sharp) ** 2).mean().item()
        batches += 1

    model.train()
    if batches == 0:
        return {}
    mean_mse = total_mse / batches
    return {
        "val_l1": total_l1 / batches,
        "val_psnr": 10.0 * math.log10(1.0 / mean_mse) if mean_mse > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blurred-dir", required=True)
    parser.add_argument("--focused-dir", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crops-per-image", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=None,
        help="restrict to these focus steps, e.g. --levels 1 2 3",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FocusStepPairs(
        args.blurred_dir,
        args.focused_dir,
        patch_size=args.patch_size,
        levels=tuple(args.levels) if args.levels else None,
        crops_per_image=args.crops_per_image,
        seed=args.seed,
    )

    n_val = max(1, int(len(dataset) * args.val_fraction))
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train crops: {len(train_set)}   Val crops: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = build_model(args, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history: list[dict] = []
    best_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, 1):
            blur = batch["blurred"].to(device, non_blocking=True)
            sharp = batch["focused"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = criterion(model(blur), sharp)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimiser)
            scaler.update()

            running += loss.item()
            seen += 1
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} l1={running / seen:.5f}")

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_l1": running / max(seen, 1),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - started, 1),
        }
        record.update(validate(model, val_loader, device))
        history.append(record)
        print(f"epoch {epoch}: {record}")

        torch.save({"params": model.state_dict()}, out_dir / "restormer_focusstep_last.pth")
        if record.get("val_psnr", -1) > best_psnr:
            best_psnr = record["val_psnr"]
            torch.save({"params": model.state_dict()}, out_dir / "restormer_focusstep_best.pth")
            print(f"  new best val PSNR {best_psnr:.2f} dB")

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nDone. Best val PSNR {best_psnr:.2f} dB. Checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()
