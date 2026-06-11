"""Self-supervised pre-training of ResNetAutoencoder2D on BraTS PNG data.

Mirrors the notebook-based 3D autoencoder training but for 2D:
  - Input: 5-channel stacked quantile slices (5, 240, 240) per BraTS patient
  - Task: reconstruct the input via MSELoss
  - No outcome labels needed — all BraTS patients are used

Weights are saved to:
    baseline/autoencoder/models/encoder_2d/{model_name}_{epochs}e_{lr}lr.pt

CLI:
    python -m baseline.autoencoder.train_autoencoder_2d \\
        --model_name resnet18 --epochs 150 --lr 0.0004 --batch_size 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

FYP_ROOT = Path(__file__).resolve().parents[2]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

from datasets.brats_png_os import BraTSPNGDatasetWithOS
from utils.utils import seed_all
from baseline.autoencoder.models_2d import ResNetAutoencoder2D

DATA_ROOT = FYP_ROOT.parent
DEFAULT_DATA_DIR = str(DATA_ROOT / "Processed_BraTS2020_TrainingData_PNG")
ENCODER_DIR = FYP_ROOT / "baseline" / "autoencoder" / "models" / "encoder_2d"


def result_path_for(cfg: dict) -> Path:
    """Return the expected weight file path for this config."""
    return ENCODER_DIR / f"{cfg['model_name']}_{cfg['epochs']}e_{cfg['lr']}lr.pt"


def run_pretraining(cfg: dict) -> None:
    seed_all(cfg["seed"], deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU model:", torch.cuda.get_device_name(0))

    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    save_path = ENCODER_DIR / f"{cfg['model_name']}_{cfg['epochs']}e_{cfg['lr']}lr.pt"
    print(f"Weights will be saved to: {save_path}")

    data_dir = cfg.get("data_dir") or DEFAULT_DATA_DIR

    print("Loading BraTS PNG dataset (all patients, no label filtering)...")
    dataset = BraTSPNGDatasetWithOS(
        root_dir=data_dir,
        score_file=None,  # use all patients; label is ignored
        max_patients=cfg.get("max_patients"),
    )
    print(f"Total patients: {len(dataset)}")

    # 80/20 train/val split by patient index (reproducible)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(dataset)).tolist()
    split = int(0.8 * len(dataset))
    train_idx, val_idx = indices[:split], indices[split:]
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True,
    )

    model = ResNetAutoencoder2D(
        model_name=cfg["model_name"], in_channels=cfg["in_channels"]
    ).to(device)
    print(f"Model: ResNetAutoencoder2D({cfg['model_name']}, in_channels={cfg['in_channels']})")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")
        for batch in pbar:
            imgs = batch[0].to(device)  # (B, in_channels, H, W); label (batch[1]) ignored
            optimizer.zero_grad()
            x_hat, _ = model(imgs)
            loss = criterion(x_hat, imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / max(1, pbar.n):.5f}"})

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch[0].to(device)
                x_hat, _ = model(imgs)
                val_loss += criterion(x_hat, imgs).item()
        val_loss /= max(1, len(val_loader))

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved new best (val_loss={best_val_loss:.5f}) → {save_path.name}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{cfg['patience']} epoch(s). Best={best_val_loss:.5f}")
            if epochs_no_improve >= cfg["patience"]:
                print("Early stopping triggered.")
                break

    print(f"\nPre-training complete. Best val_loss={best_val_loss:.5f}")
    print(f"Weights saved to: {save_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D autoencoder pre-training on BraTS PNG.")
    p.add_argument("--model_name", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--in_channels", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pretraining(vars(args))
