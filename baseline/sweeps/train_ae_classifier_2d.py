"""Headless 2D autoencoder-classifier fine-tuning trainer (BrainWear PNG).

Sses:
  - ResNetAutoencoder2D / ResNetClassifier2D  (from baseline/autoencoder/models_2d.py)
  - BrainWearPNGDataset                       (5-channel stacked PNG slices)
  - encoder_2d/ for weight resolution          (set by train_autoencoder_2d.py)
  - classifier_2d/ for output                  (separate from 3D classifier/ runs)

CLI:
    python -m baseline.sweeps.train_ae_classifier_2d \\
        --ae_weights resnet18_150e_0.0004lr.pt --epochs 40 --num_classes 3 --folds 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

FYP_ROOT = Path(__file__).resolve().parents[2]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

from datasets.brainwear_png import BrainWearPNGDataset
from utils.utils import seed_all
from baseline.autoencoder.models_2d import ResNetAutoencoder2D, ResNetClassifier2D
from baseline.sweeps.metrics import print_metrics

DATA_ROOT = FYP_ROOT.parent
MODEL_DIR = FYP_ROOT / "baseline" / "autoencoder" / "models"
ENCODER_DIR = MODEL_DIR / "encoder_2d"
DEFAULT_DATA_DIR = str(DATA_ROOT / "Processed_Brainwear_PNG")
DEFAULT_SCORE_FILE = str(DATA_ROOT / "eortc_scores.csv")


def resolve_ae_weights(ae_weights: str) -> Path:
    """Accept either a bare filename (resolved under encoder_2d/) or a full path."""
    p = Path(ae_weights)
    return p if p.is_absolute() or p.exists() else ENCODER_DIR / ae_weights


def make_run_name(cfg: dict) -> str:
    model = Path(cfg["ae_weights"]).stem.split("_")[0]
    name = (
        f"{model}_{cfg['epochs']}e_{cfg['lr']}lr_d{cfg['dropout']}_"
        f"{cfg['folds']}k_{cfg['num_classes']}c_gradual_unfreeze_weighted_cost"
    )
    if cfg["quantile"]:
        name += "_quantile"
    if cfg.get("regression"):
        name += "_regression"
    score = cfg.get("score_name", "QL2")
    if score != "QL2":
        name += f"_score{score}"
    return name


def run_dir_for(cfg: dict) -> Path:
    return MODEL_DIR / "classifier_2d" / make_run_name(cfg)


def run_cv(cfg: dict) -> dict:
    """Fine-tune the 2D classifier under k-fold CV; write artefacts; return summary."""
    seed_all(cfg["seed"], deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ae_weights = resolve_ae_weights(cfg["ae_weights"])
    if not ae_weights.exists():
        raise FileNotFoundError(f"Autoencoder weights not found: {ae_weights}")
    model_name = ae_weights.stem.split("_")[0]
    cfg = {**cfg, "ae_weights": str(ae_weights), "model_name": model_name}

    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    print("Initialising BrainWear PNG Dataset")
    full_dataset = BrainWearPNGDataset(
        root_dir=cfg["data_dir"],
        score_file=cfg["score_file"],
        num_bins=cfg["num_classes"],
        quantile_bins=cfg["quantile"],
        regression=cfg.get("regression", False),
        max_patients=cfg.get("max_patients"),
        score_name=cfg.get("score_name", "QL2"),
    )
    total_size = len(full_dataset)

    print("Loading pre-trained 2D autoencoder")
    autoencoder = ResNetAutoencoder2D(model_name=model_name, in_channels=cfg["in_channels"])
    autoencoder.load_state_dict(torch.load(ae_weights, map_location=device), strict=False)

    labels = full_dataset.get_class_labels()
    kfold = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=42)
    unfreeze_epoch = int(cfg["epochs"] * 0.8)
    print(f"layer4 will unfreeze at epoch {unfreeze_epoch}")

    cv_results = {}
    oof_targets, oof_preds = [], []
    fold_balanced_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(total_size), labels)):
        print(f"\n================ Fold {fold + 1}/{cfg['folds']} ================")
        print(f"  train dist: {dict(sorted(Counter(labels[i] for i in train_idx).items()))}")
        print(f"  val   dist: {dict(sorted(Counter(labels[i] for i in val_idx).items()))}")

        train_labels_arr = np.array([labels[i] for i in train_idx])
        if cfg.get("regression"):
            regression_criterion = nn.MSELoss()
            classification_loss = None
        else:
            present_classes = np.unique(train_labels_arr)
            raw_weights_present = compute_class_weight(
                class_weight="balanced",
                classes=present_classes,
                y=train_labels_arr,
            )
            full_weights = np.ones(cfg["num_classes"], dtype=np.float64)
            for cls, w in zip(present_classes, raw_weights_present):
                full_weights[cls] = w
            class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)
            classification_loss = nn.CrossEntropyLoss(weight=class_weights)
            regression_criterion = None

        train_loader = DataLoader(full_dataset, batch_size=cfg["batch_size"],
                                  sampler=SubsetRandomSampler(train_idx), num_workers=4, pin_memory=True)
        val_loader = DataLoader(full_dataset, batch_size=cfg["batch_size"],
                                sampler=SubsetRandomSampler(val_idx), num_workers=4, pin_memory=True)

        model = ResNetClassifier2D(
            autoencoder=autoencoder,
            num_classes=1 if cfg.get("regression") else cfg["num_classes"],
            dropout=cfg["dropout"],
            unfreeze_encoder=False,
        ).to(device)
        model.set_encoder_frozen(True)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        save_location = run_dir / f"best_fold_{fold + 1}.pt"
        train_acc = 0.0

        for epoch in range(cfg["epochs"]):
            if epoch == unfreeze_epoch:
                print(f"  [Epoch {epoch + 1}] Unfreezing encoder layer4 (discriminative LR)")
                for name, param in model.encoder.named_parameters():
                    param.requires_grad = "layer4" in name
                model._encoder_frozen = False
                layer4_params = [p for n, p in model.encoder.named_parameters() if "layer4" in n and p.requires_grad]
                head_params = [p for p in model.classifier.parameters() if p.requires_grad]
                optimizer = optim.AdamW(
                    [
                        {"params": layer4_params, "lr": cfg["lr"] * 0.1},
                        {"params": head_params, "lr": cfg["lr"]},
                    ],
                    weight_decay=cfg["weight_decay"],
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
                epochs_no_improve = 0

            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False)
            for imgs, batch_labels in pbar:
                imgs, batch_labels = imgs.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                if cfg.get("regression"):
                    loss = regression_criterion(logits.squeeze(1), batch_labels.float())
                else:
                    loss = classification_loss(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                if cfg.get("regression"):
                    predicted = torch.tensor([full_dataset._score_to_class(p) for p in logits.squeeze(1).detach().cpu().numpy()], device=device)
                    true_class = torch.tensor([full_dataset._score_to_class(s) for s in batch_labels.cpu().numpy()], device=device)
                    train_correct += (predicted == true_class).sum().item()
                else:
                    _, predicted = torch.max(logits.data, 1)
                    train_correct += (predicted == batch_labels).sum().item()
                train_total += batch_labels.size(0)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, batch_labels in val_loader:
                    imgs, batch_labels = imgs.to(device), batch_labels.to(device)
                    logits = model(imgs)
                    if cfg.get("regression"):
                        val_loss += regression_criterion(logits.squeeze(1), batch_labels.float()).item()
                        predicted = torch.tensor([full_dataset._score_to_class(p) for p in logits.squeeze(1).cpu().numpy()], device=device)
                        true_class = torch.tensor([full_dataset._score_to_class(s) for s in batch_labels.cpu().numpy()], device=device)
                        val_correct += (predicted == true_class).sum().item()
                    else:
                        val_loss += classification_loss(logits, batch_labels).item()
                        _, predicted = torch.max(logits.data, 1)
                        val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            scheduler.step(avg_val_loss)
            print(f"F{fold+1} Epoch [{epoch+1}/{cfg['epochs']}] | Train {avg_train_loss:.4f}/{train_acc:.1f}% | Val {avg_val_loss:.4f}/{val_acc:.1f}%")

            if avg_val_loss < best_val_loss:
                best_val_loss, best_val_acc, best_epoch = avg_val_loss, val_acc, epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_location)
                print(f"  New best @ epoch {best_epoch} (val_loss={best_val_loss:.4f}, acc={best_val_acc:.1f}%)")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg["patience"]:
                    print(f"  Early stopping. Best val_loss={best_val_loss:.4f}")
                    break

        if save_location.exists():
            model.load_state_dict(torch.load(save_location, map_location=device))
        model.eval()
        fold_targets, fold_preds = [], []
        with torch.no_grad():
            for imgs, batch_labels in val_loader:
                logits = model(imgs.to(device))
                if cfg.get("regression"):
                    preds_raw = logits.squeeze(1).cpu().numpy()
                    fold_targets.extend([full_dataset._score_to_class(s) for s in batch_labels.numpy()])
                    fold_preds.extend([full_dataset._score_to_class(p) for p in preds_raw])
                else:
                    preds = torch.argmax(logits, dim=1)
                    fold_targets.extend(batch_labels.numpy())
                    fold_preds.extend(preds.cpu().numpy())

        fold_eval = print_metrics(
            np.array(fold_targets), np.array(fold_preds),
            strategy="categorical", header=f"FOLD {fold+1} VALIDATION RESULTS",
            num_labels=cfg["num_classes"], include_balanced=True,
        )
        fold_balanced_accs.append(fold_eval["balanced_accuracy"])
        oof_targets.extend(fold_targets)
        oof_preds.extend(fold_preds)

        cv_results[f"fold_{fold+1}"] = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_balanced_acc": fold_eval["balanced_accuracy"],
            "final_train_acc": train_acc,
        }
        print(f"Fold {fold+1} complete. Best Val Acc: {best_val_acc:.2f}% | Balanced Acc: {fold_eval['balanced_accuracy']:.4f}")

    avg_cv_loss = sum(f["best_val_loss"] for f in cv_results.values()) / cfg["folds"]
    avg_cv_acc = sum(f["best_val_acc"] for f in cv_results.values()) / cfg["folds"]
    avg_balanced = float(np.mean(fold_balanced_accs))
    oof_eval = print_metrics(
        np.array(oof_targets), np.array(oof_preds),
        strategy="categorical", header="OUT-OF-FOLD RESULTS",
        num_labels=cfg["num_classes"], include_balanced=True,
    )

    print("\n============ Cross-Validation Complete ============")
    print(f"Average CV Val Accuracy:   {avg_cv_acc:.2f}%")
    print(f"Average CV Balanced Acc:   {avg_balanced:.4f}")
    print(f"Out-of-fold Balanced Acc:  {oof_eval['balanced_accuracy']:.4f}")

    run_details = {
        "arguments": cfg,
        "cross_validation_metrics": {
            "average_val_loss": avg_cv_loss,
            "average_val_acc": avg_cv_acc,
            "average_balanced_acc": avg_balanced,
            "oof_balanced_acc": oof_eval["balanced_accuracy"],
        },
        "oof_results": oof_eval,
        "fold_details": cv_results,
    }
    json_path = run_dir / "cv_run_details.json"
    with open(json_path, "w") as f:
        json.dump(run_details, f, indent=4)
    print(f"Details saved to {json_path}")
    return run_details


def cfg_from_args(args: argparse.Namespace) -> dict:
    cfg = vars(args).copy()
    if cfg.get("data_dir") is None:
        cfg["data_dir"] = DEFAULT_DATA_DIR
    if cfg.get("score_file") is None:
        cfg["score_file"] = DEFAULT_SCORE_FILE
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D autoencoder-classifier fine-tuning CV trainer.")
    p.add_argument("--ae_weights", required=True, help="Encoder .pt: bare filename (under encoder_2d/) or path.")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--score_file", default=None)
    p.add_argument("--in_channels", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--quantile", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regression", action="store_true")
    p.add_argument("--score_name", default="QL2", help="EORTC outcome scale to predict (e.g. QL2, FA, BNVD).")
    return p


if __name__ == "__main__":
    run_cv(cfg_from_args(build_parser().parse_args()))
