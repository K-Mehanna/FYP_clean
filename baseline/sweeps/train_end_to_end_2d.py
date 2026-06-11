"""End-to-end 2D ResNet cross-validation trainer for PNG datasets.

Uses torchvision 2D ResNets loads from the PNG dataset classes
(``BrainWearPNGDataset`` / ``BraTSPNGDatasetWithOS``).

The first conv layer is replaced to accept ``in_channels`` input channels
(default 5 — one per quantile slice stacked as a channel).

Run directory layout and ``results.json`` schema are identical to the 3D
trainer so the aggregator treats both uniformly.

CLI:
    python -m baseline.sweeps.train_end_to_end_2d --dataset brainwear_png \\
        --model resnet18 --epochs 30 --n_splits 5 --num_bins 3 --quantile
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

FYP_ROOT = Path(__file__).resolve().parents[2]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

from datasets.brainwear_png import BrainWearPNGDataset
from datasets.brats_png_os import BraTSPNGDatasetWithOS
from utils.utils import seed_all
from baseline.sweeps.metrics import print_metrics

DATA_ROOT = FYP_ROOT.parent
DEFAULTS = {
    "brainwear_png": {
        "data_dir": str(DATA_ROOT / "Processed_Brainwear_PNG"),
        "csv_path": str(DATA_ROOT / "eortc_scores.csv"),
    },
    "brats_png": {
        "data_dir": str(DATA_ROOT / "Processed_BraTS2020_TrainingData_PNG"),
        "csv_path": str(DATA_ROOT / "BraTS_OS.csv"),
    },
}

_ORDINAL_LABEL_SPACE = 5


def _build_model(cfg: dict, num_outputs: int) -> nn.Module:
    if cfg["model"] == "resnet18":
        model = tvm.resnet18(weights=None)
    else:
        model = tvm.resnet50(weights=None)
    in_ch = cfg["in_channels"]
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model


def build_dataset(dataset_name: str, cfg: dict):
    common = dict(
        root_dir=cfg["data_dir"],
        max_patients=cfg.get("max_patients"),
        shuffle=False,
        score_file=cfg["csv_path"],
        quantile_bins=cfg["quantile"],
        num_bins=cfg["num_bins"],
        regression=(cfg["strategy"] == "regression"),
    )
    if dataset_name == "brainwear_png":
        return BrainWearPNGDataset(**common, score_name=cfg.get("score_name", "QL2"))
    if dataset_name == "brats_png":
        return BraTSPNGDatasetWithOS(**common, allowed_patients=cfg.get("allowed_patients"))
    raise ValueError(f"Unknown dataset '{dataset_name}' (expected 'brainwear_png' or 'brats_png')")


def make_run_name(cfg: dict) -> str:
    name = (
        f"e{cfg['epochs']}_b{cfg['batch_size']}_{cfg['model']}_"
        f"{cfg['strategy'][:3]}_lr{cfg['lr']}_cv{cfg['n_splits']}_decay{cfg['weight_decay']}"
    )
    if not cfg["quantile"]:
        name += "_no_quant"
    if cfg.get("num_classes_sweep"):
        name += f"_nb{cfg['num_bins']}"
    score = cfg.get("score_name", "QL2")
    if score != "QL2":
        name += f"_score{score}"
    if cfg.get("holdout_frac") is not None:
        name += f"_ho{cfg['holdout_frac']}"
    return name


def run_dir_for(cfg: dict) -> Path:
    return (
        FYP_ROOT
        / "baseline"
        / "end-to-end"
        / "runs"
        / cfg["dataset"]
        / make_run_name(cfg)
    )


def run_cv(cfg: dict) -> dict:
    seed_all(cfg["seed"], deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU model:", torch.cuda.get_device_name(0))

    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    # Exclude non-serialisable fields (e.g. allowed_patients set) from args.json.
    _serialisable_cfg = {k: v for k, v in cfg.items() if not isinstance(v, (set, frozenset))}
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(_serialisable_cfg, f, indent=2)

    # ── Holdout split ────────────────────────────────────────────────────────
    import os as _os
    holdout_frac = cfg.get("holdout_frac")
    holdout_folders: set | None = None
    n_holdout = 0
    if holdout_frac:
        _all_sorted = sorted(
            f for f in _os.listdir(cfg["data_dir"])
            if _os.path.isdir(_os.path.join(cfg["data_dir"], f))
        )
        n_holdout = max(1, int(len(_all_sorted) * holdout_frac))
        holdout_folders = set(_all_sorted[:n_holdout])
        cfg = {**cfg, "allowed_patients": set(_all_sorted[n_holdout:])}
        print(f"Holdout: first {n_holdout}/{len(_all_sorted)} patients reserved for test evaluation")
    # ────────────────────────────────────────────────────────────────────────

    print("Loading Dataset...")
    dataset = build_dataset(cfg["dataset"], cfg)
    all_idx = np.arange(len(dataset))
    all_labels = np.array(dataset.get_class_labels())
    unique_labels, class_counts = np.unique(all_labels, return_counts=True)

    if cfg["n_splits"] == 1:
        train_idx_np, val_idx_np = train_test_split(
            all_idx, test_size=0.3, stratify=all_labels, random_state=cfg["fold_seed"]
        )
        fold_splits = [(train_idx_np, val_idx_np)]
        print(f"Single train/val split: {len(train_idx_np)} train, {len(val_idx_np)} val")
    else:
        if int(class_counts.min()) < cfg["n_splits"]:
            raise ValueError(
                "Cannot run stratified k-fold: smallest class has "
                f"{int(class_counts.min())} samples but n_splits={cfg['n_splits']}. "
                f"Class counts: {dict(zip(unique_labels.tolist(), class_counts.tolist()))}"
            )
        skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["fold_seed"])
        fold_splits = list(skf.split(all_idx, all_labels))
        print(f"Using {cfg['n_splits']}-fold stratified cross-validation")

    if cfg["strategy"] == "ordinal":
        num_outputs = 4
        label_space = _ORDINAL_LABEL_SPACE
    elif cfg["strategy"] == "regression":
        num_outputs = 1
        label_space = cfg["num_bins"]
    else:
        num_outputs = cfg["num_bins"]
        label_space = cfg["num_bins"]

    # ── Holdout evaluation dataset (built once, used after every fold) ───────
    holdout_loader = None
    if holdout_folders:
        _ho_cfg = {**cfg, "max_patients": None, "allowed_patients": holdout_folders}
        holdout_dataset = build_dataset(cfg["dataset"], _ho_cfg)
        holdout_loader = DataLoader(holdout_dataset, batch_size=cfg["batch_size"], shuffle=False)
        print(f"Holdout evaluation set: {len(holdout_dataset)} patients, {len(holdout_dataset)} samples")
    # ────────────────────────────────────────────────────────────────────────

    holdout_fold_f1s: list[float] = []
    fold_metrics = []
    oof_targets, oof_preds = [], []
    cv_start = time.time()

    for fold_idx, (train_idx_np, val_idx_np) in enumerate(fold_splits, start=1):
        train_idx, val_idx = train_idx_np.tolist(), val_idx_np.tolist()
        print(f"\n========== Fold {fold_idx}/{cfg['n_splits']} ==========")
        print(f"Train set: {len(train_idx)} | Val set: {len(val_idx)}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg["batch_size"], shuffle=False)

        model = _build_model(cfg, num_outputs).to(device)

        if cfg["strategy"] == "ordinal":
            criterion = nn.BCEWithLogitsLoss()
        elif cfg["strategy"] == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(cfg["scheduler_patience"]))

        best_ckpt_path = run_dir / f"best_model_fold{fold_idx}.pt"
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        epochs_ran = 0
        stopped_early = False

        for epoch in range(cfg["epochs"]):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{cfg['epochs']}")
            for batch in pbar:
                x, y = batch[0].to(device), batch[-1].to(device)
                optimizer.zero_grad()
                logits = model(x)
                if cfg["strategy"] == "ordinal":
                    targets = torch.stack([y >= k for k in range(1, 5)], dim=1).float()
                    loss = criterion(logits, targets)
                elif cfg["strategy"] == "regression":
                    loss = criterion(logits.squeeze(1), y.float())
                else:
                    loss = criterion(logits, y.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{running_loss / max(1, pbar.n):.4f}"})

            train_loss = running_loss / max(1, len(train_loader))

            model.eval()
            val_running_loss = 0.0
            val_targets, val_preds = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(device), batch[-1].to(device)
                    logits = model(x)
                    if cfg["strategy"] == "ordinal":
                        targets = torch.stack([y >= k for k in range(1, 5)], dim=1).float()
                        vloss = criterion(logits, targets)
                        preds = (logits > 0).int().sum(dim=1)
                        val_targets.extend(y.cpu().numpy())
                        val_preds.extend(preds.cpu().numpy())
                    elif cfg["strategy"] == "regression":
                        vloss = criterion(logits.squeeze(1), y.float())
                        preds_raw = logits.squeeze(1).cpu().numpy()
                        val_targets.extend([dataset._score_to_class(s) for s in y.cpu().numpy()])
                        val_preds.extend([dataset._score_to_class(p) for p in preds_raw])
                    else:
                        vloss = criterion(logits, y.long())
                        preds = torch.argmax(logits, dim=1)
                        val_targets.extend(y.cpu().numpy())
                        val_preds.extend(preds.cpu().numpy())
                    val_running_loss += vloss.item()

            val_epoch_loss = val_running_loss / max(1, len(val_loader))
            val_acc = accuracy_score(np.array(val_targets), np.array(val_preds))
            epochs_ran = epoch + 1
            print(f"Fold {fold_idx} Epoch {epoch+1}: train_loss={train_loss:.4f} | val_loss={val_epoch_loss:.4f} | val_acc={val_acc:.4f}")

            if (best_val_loss - val_epoch_loss) > cfg["min_delta"]:
                best_val_loss = val_epoch_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": cfg,
                        "num_outputs": num_outputs,
                    },
                    best_ckpt_path,
                )
                print(f"Saved new best fold checkpoint at epoch {best_epoch} (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{cfg['patience']} epoch(s). Best={best_val_loss:.4f} @ {best_epoch}.")

            if epochs_no_improve >= cfg["patience"]:
                stopped_early = True
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
            scheduler.step(val_epoch_loss)

        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(f"Loaded best checkpoint (epoch {best_ckpt['best_epoch']}, val_loss={best_ckpt['best_val_loss']:.4f}).")

        model.eval()
        fold_targets, fold_preds = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx} Validation Eval"):
                x = batch[0].to(device)
                y = batch[-1]
                logits = model(x)
                if cfg["strategy"] == "ordinal":
                    preds = (logits > 0).int().sum(dim=1)
                    fold_targets.extend(y.numpy())
                    fold_preds.extend(preds.cpu().numpy())
                elif cfg["strategy"] == "regression":
                    preds_raw = logits.squeeze(1).cpu().numpy()
                    fold_targets.extend([dataset._score_to_class(s) for s in y.numpy()])
                    fold_preds.extend([dataset._score_to_class(p) for p in preds_raw])
                else:
                    preds = torch.argmax(logits, dim=1)
                    fold_targets.extend(y.numpy())
                    fold_preds.extend(preds.cpu().numpy())

        # ── Evaluate on held-out test patients ───────────────────────────────
        if holdout_loader is not None:
            from sklearn.metrics import f1_score as _f1_score
            model.eval()
            ho_targets_fold: list = []
            ho_preds_fold: list = []
            with torch.no_grad():
                for batch in holdout_loader:
                    x, y = batch[0].to(device), batch[-1]
                    logits = model(x)
                    if cfg["strategy"] == "ordinal":
                        preds = (logits > 0).int().sum(dim=1)
                        ho_targets_fold.extend(y.numpy())
                        ho_preds_fold.extend(preds.cpu().numpy())
                    elif cfg["strategy"] == "regression":
                        preds_raw = logits.squeeze(1).cpu().numpy()
                        ho_targets_fold.extend([dataset._score_to_class(s) for s in y.numpy()])
                        ho_preds_fold.extend([dataset._score_to_class(p) for p in preds_raw])
                    else:
                        preds = torch.argmax(logits, dim=1)
                        ho_targets_fold.extend(y.numpy())
                        ho_preds_fold.extend(preds.cpu().numpy())
            fold_ho_f1 = float(_f1_score(np.array(ho_targets_fold), np.array(ho_preds_fold),
                                         average="macro", zero_division=0))
            holdout_fold_f1s.append(fold_ho_f1)
            print(f"Fold {fold_idx} holdout macro-F1: {fold_ho_f1:.4f}")
        # ────────────────────────────────────────────────────────────────────

        fold_results = print_metrics(
            np.array(fold_targets), np.array(fold_preds),
            cfg["strategy"], header=f"FOLD {fold_idx} VALIDATION RESULTS",
            num_labels=label_space,
        )
        fold_results.update(
            {
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "epochs_ran": int(epochs_ran),
                "stopped_early": bool(stopped_early),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )
        fold_metrics.append(fold_results)
        oof_targets.extend(fold_targets)
        oof_preds.extend(fold_preds)

    cv_elapsed = time.time() - cv_start
    print(f"Cross-validation completed in {cv_elapsed/60:.2f} minutes")

    fold_acc = np.array([fm["accuracy"] for fm in fold_metrics], dtype=np.float64)
    fold_loss = np.array([fm["best_val_loss"] for fm in fold_metrics], dtype=np.float64)
    fold_epochs = np.array([fm["epochs_ran"] for fm in fold_metrics], dtype=np.float64)

    oof_results = print_metrics(
        np.array(oof_targets), np.array(oof_preds),
        cfg["strategy"], header="OUT-OF-FOLD RESULTS",
        num_labels=label_space, include_balanced=True,
    )

    summary = {
        "strategy": cfg["strategy"],
        "model": cfg["model"],
        "n_splits": cfg["n_splits"],
        "seed": cfg["seed"],
        "fold_seed": cfg["fold_seed"],
        "num_samples": len(dataset),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "best_val_loss_mean": float(np.mean(fold_loss)),
        "best_val_loss_std": float(np.std(fold_loss)),
        "epochs_ran_mean": float(np.mean(fold_epochs)),
        "epochs_ran_std": float(np.std(fold_epochs)),
        "cv_elapsed_minutes": float(cv_elapsed / 60.0),
    }
    if holdout_fold_f1s:
        summary["holdout_frac"] = holdout_frac
        summary["holdout_n_patients"] = n_holdout
        summary["holdout_cv_f1"] = holdout_fold_f1s
        summary["holdout_cv_mean_f1"] = float(np.mean(holdout_fold_f1s))
        summary["holdout_cv_std_f1"] = float(np.std(holdout_fold_f1s))
        print(f"Holdout macro-F1: {summary['holdout_cv_mean_f1']:.4f} ± {summary['holdout_cv_std_f1']:.4f}")
    print(
        f"CV summary: accuracy={summary['accuracy_mean']:.4f}+/-{summary['accuracy_std']:.4f}, "
        f"best_val_loss={summary['best_val_loss_mean']:.4f}+/-{summary['best_val_loss_std']:.4f}, "
        f"oof_balanced_acc={oof_results['balanced_accuracy']:.4f}"
    )

    results = {"summary": summary, "oof_results": oof_results, "fold_metrics": fold_metrics}
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {run_dir / 'results.json'}")
    return results


def cfg_from_args(args: argparse.Namespace) -> dict:
    cfg = vars(args).copy()
    if cfg.get("data_dir") is None:
        cfg["data_dir"] = DEFAULTS[cfg["dataset"]]["data_dir"]
    if cfg.get("csv_path") is None:
        cfg["csv_path"] = DEFAULTS[cfg["dataset"]]["csv_path"]
    if cfg.get("fold_seed") is None:
        cfg["fold_seed"] = cfg["seed"]
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end 2D ResNet CV trainer (PNG datasets).")
    p.add_argument("--dataset", choices=["brainwear_png", "brats_png"], required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--csv_path", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold_seed", type=int, default=None)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--strategy", choices=["ordinal", "categorical", "regression"], default="categorical")
    p.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--quantile", action="store_true")
    p.add_argument("--num_bins", type=int, default=5)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--holdout_frac", type=float, default=None,
                   help="Hold out the first N%% of sorted patients as test set; train on the rest.")
    p.add_argument("--in_channels", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--scheduler_patience", type=int, default=4)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--score_name", default="QL2", help="EORTC outcome scale to predict (e.g. QL2, FA, BNVD).")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.n_splits < 1:
        raise ValueError("--n_splits must be >= 1.")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1.")
    run_cv(cfg_from_args(args))
