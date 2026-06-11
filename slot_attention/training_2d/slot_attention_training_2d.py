"""
2D PNG slot-attention training script.

Trains SlotClassifier2D on axial PNG slices from the BraTS2020 dataset
"""

import os
import sys
import time
import math
import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, random_split, Subset

import wandb

root_path = Path("/path/to/BrainWear_Kareem")
project_root = root_path/"FYP"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Added to path: {project_root}")

from datasets.brats2020_png import BraTS2020PNGDataset, batch_seg_to_slot_targets_2d
from slot_attention.training_2d.slot_attention_2d import SlotClassifier2D
from utils.utils import (
    compute_cost_matrix, compute_cost_matrix_2d, compute_cost_matrix_2d_spatial,
    compute_cost_matrix_2d_centroid, compute_cost_matrix_2d_class_only,
    hungarian_algorithm, seed_all, seed_worker, get_slot_attention_scheduler,
)

def init_params(p: nn.Module) -> None:
    if isinstance(p, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(p.weight)
        if p.bias is not None:
            p.bias.data.fill_(0)

def slot_attention_loss(
    slot_objects: Tensor,
    labels: Tensor,
    num_slots: int,
    recon: Tensor,
    img: Tensor,
    masks: Tensor,
    seg: Tensor,
    classification_weight: float,
    mask_weight: float,
    bce_mask: bool = True,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.5,
    focal_alpha_per_class: Tensor | None = None,
    weakly_supervised: bool = False,
    matching: str = "spatial_dice",
    weak_coord_weight: float = 0.0,
    mask_entropy_weight: float = 0.0,
    use_mlp_coord_vol: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, tuple[float, int]]]:
    B = seg.shape[0]

    class_ids = torch.arange(num_slots, device=seg.device)
    gt_seg = (seg[:, None, None, :, :] == class_ids[None, :, None, None, None]).float()

    with torch.no_grad():
        if matching == "spatial_dice":
            # Full GT mask Dice + class BCE supplement. Default supervised behaviour.
            # When masks are diffuse early in training, Dice is uniform across slots;
            # the 0.5*BCE supplement keeps matching class-coherent in that regime.
            cost_matrix = compute_cost_matrix_2d_spatial(
                slot_objects, labels, num_slots, coord_dim=2,
                pred_masks=masks, target_masks=gt_seg.squeeze(2)
            )
            cp = torch.clamp(torch.sigmoid(slot_objects[..., :4]), 1e-7, 1.0 - 1e-7)
            gt_cls = torch.clamp(labels[..., :4], 0.0, 1.0)
            bce_cost = -(
                gt_cls.unsqueeze(1) * torch.log(cp.unsqueeze(2)) +
                (1 - gt_cls.unsqueeze(1)) * torch.log(1 - cp.unsqueeze(2))
            ).mean(dim=-1)
            cost_matrix = cost_matrix + 0.5 * bce_cost
        elif matching == "centroid":
            # Class BCE + predicted-mask centroid vs GT centroid L2.
            # No raw GT mask pixels used; GT centroids are derived summary statistics.
            cost_matrix = compute_cost_matrix_2d_centroid(
                slot_objects, labels, num_slots, pred_masks=masks
            )
        else:  # class_only
            # Pure class BCE — no GT mask pixels, no spatial statistics at all.
            cost_matrix = compute_cost_matrix_2d_class_only(
                slot_objects, labels, num_slots
            )
        _, indices = hungarian_algorithm(cost_matrix)

    batch_idx = torch.arange(B, device=seg.device)[:, None] # (B, 1)
    
    pred_idx = indices[:, 0]  # Slots
    gt_idx = indices[:, 1]    # Ground Truth Labels

    matched_preds = slot_objects[batch_idx, pred_idx]
    matched_labels = labels[batch_idx, gt_idx]
    
    matched_pred_masks = masks.squeeze(2)[batch_idx, pred_idx]
    matched_gt_masks = gt_seg.squeeze(2)[batch_idx, gt_idx]

    # Loss calculations (Mean instead of Sum)
    # 1D Classifier Loss with sigmoid
    pred_class = torch.clamp(torch.sigmoid(matched_preds[..., :4]), 1e-7, 1.0 - 1e-7)
    target_class = torch.clamp(matched_labels[..., :4], 0.0, 1.0)
    if focal_gamma > 0.0:
        # Focal BCE with optional alpha weighting.
        # gamma suppresses easy examples; alpha addresses class-prior imbalance
        # (background appears in every slice, ET/NCR may be absent).
        bce_elem = -(target_class * torch.log(pred_class) + (1 - target_class) * torch.log(1 - pred_class))
        pt = target_class * pred_class + (1 - target_class) * (1 - pred_class)
        if focal_alpha_per_class is not None:
            # Per-class alpha: shape (1, 1, 4) broadcasts over (B, K, 4).
            # Reduces background self-identification weight and increases tumour
            # slot "not-background" discrimination weight.
            alpha_vec = focal_alpha_per_class.to(pred_class.device).view(1, 1, -1)
            alpha_t = target_class * alpha_vec + (1 - target_class) * (1 - alpha_vec)
        else:
            alpha_t = target_class * focal_alpha + (1 - target_class) * (1 - focal_alpha)
        class_loss = (alpha_t * (1 - pt).pow(focal_gamma) * bce_elem).mean()
    else:
        class_loss = F.binary_cross_entropy(pred_class, target_class)

    # Volume and centroid: either derived from mask spatial properties
    # or predicted by the MLP heads on the slot vectors (use_mlp_coord_vol=True).
    # Mask-derived values always computed; used as loss/metric source in the default path.
    H_m, W_m = matched_pred_masks.shape[2:]
    mask_vol = matched_pred_masks.sum(dim=(2, 3)) / (H_m * W_m)  # (B, K)
    target_vol = matched_labels[..., 6:].squeeze(-1)              # (B, K)

    target_coords = matched_labels[..., 4:6]                              # (B, K, 2)
    grid_y = torch.linspace(0, 1, H_m, device=matched_pred_masks.device).view(1, 1, H_m, 1)
    grid_x = torch.linspace(0, 1, W_m, device=matched_pred_masks.device).view(1, 1, 1, W_m)
    mask_sum = matched_pred_masks.sum(dim=(2, 3)).clamp(min=1e-6)         # (B, K)
    mask_cy = (matched_pred_masks * grid_y).sum(dim=(2, 3)) / mask_sum   # (B, K)
    mask_cx = (matched_pred_masks * grid_x).sum(dim=(2, 3)) / mask_sum   # (B, K)
    mask_coords = torch.stack([mask_cy, mask_cx], dim=-1)                 # (B, K, 2)

    if use_mlp_coord_vol:
        # Gradient flows through slot vectors only - masks remain unsupervised.
        # output layout: [..., :4]=class, [..., 4:6]=coords, [..., 6:]=volume
        pred_vol = torch.sigmoid(matched_preds[..., 6:]).squeeze(-1)  # (B, K)
        pred_coords = torch.sigmoid(matched_preds[..., 4:6])          # (B, K, 2)
    else:
        pred_vol = mask_vol
        pred_coords = mask_coords

    is_present = (target_vol > 0.0)                                              # (B, K)
    vol_loss = F.mse_loss(pred_vol, target_vol)
    coord_err = F.mse_loss(pred_coords, target_coords, reduction="none")         # (B, K, 2)
    if is_present.any():
        coord_loss = (coord_err * is_present.unsqueeze(-1).float()).sum() / (is_present.float().sum() * 2 + 1e-6)
    else:
        coord_loss = coord_err.mean()

    if weakly_supervised:
        if use_mlp_coord_vol:
            # Coord+vol always in loss: provides the only gradient path for spatial MLP
            # heads when masks are unsupervised and recon bypass makes ∂recon/∂masks ≈ 0.
            classifier_loss = class_loss + coord_loss + vol_loss
        else:
            # Mask-derived path: coord+vol optionally anchored via weak_coord_weight.
            coord_anchor = weak_coord_weight * (coord_loss + vol_loss) if weak_coord_weight > 0.0 else coord_loss.new_zeros(1)
            classifier_loss = class_loss + coord_anchor
    else:
        classifier_loss = class_loss + coord_loss + vol_loss

    # 2D Dice + BCE Mask Loss. BCE penalises diffuse masks that cheat soft-Dice
    # by spreading probability uniformly (good soft-Dice, near-zero hard-Dice).
    smooth = 1.0
    intersection = (matched_pred_masks * matched_gt_masks).sum(dim=(2, 3))
    denominator = matched_pred_masks.sum(dim=(2, 3)) + matched_gt_masks.sum(dim=(2, 3))
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    mask_dice_loss = (1.0 - dice_score).mean()

    if bce_mask:
        mask_bce_loss = F.binary_cross_entropy(
            matched_pred_masks.clamp(1e-6, 1.0 - 1e-6),
            matched_gt_masks.clamp(1e-6, 1.0 - 1e-6),
        )
        mask_dice_loss = 0.5 * mask_dice_loss + 0.5 * mask_bce_loss

    # Image reconstruction loss
    recon_loss = F.mse_loss(recon, img)

    if weakly_supervised:
        loss = classification_weight * classifier_loss + recon_loss
    else:
        loss = classification_weight * classifier_loss + recon_loss + mask_weight * mask_dice_loss

    # Entropy + coverage-floor regularizer (anti-uniform, anti-winner-take-all).
    # Per-pixel entropy minimisation sharpens slot assignments (pushes away from 0.2).
    # Coverage floor penalises slots whose mean coverage falls below 1/(2K),
    # preventing winner-take-all collapse when background recon dominance is strong.
    if mask_entropy_weight > 0.0:
        K_slots = masks.shape[1]
        per_pixel_H = -(masks * torch.log(masks + 1e-8)).sum(dim=1).mean()
        slot_cov = masks.squeeze(2).mean(dim=(2, 3))
        min_cov = 1.0 / (2.0 * K_slots)
        coverage_floor = F.relu(min_cov - slot_cov).pow(2).sum(dim=1).mean()
        loss = loss + mask_entropy_weight * (per_pixel_H + coverage_floor)

    # ---- Slot-quality metrics (no grad). These are what AA-CBR ultimately
    # consumes: hard segmentation Dice (-> centroid/volume), class accuracy,
    # volume MAE and centroid error. Returned as {name: (sum, count)} so the
    # epoch loop can aggregate correctly over present classes only.
    with torch.no_grad():
        # Hard slot assignment per pixel from the softmax masks
        hard = masks.squeeze(2).argmax(dim=1)  # (B, H, W)
        pred_bin = (hard.unsqueeze(1) == pred_idx[:, :, None, None]).float()  # (B, K, H, W)

        inter_h = (pred_bin * matched_gt_masks).sum(dim=(2, 3))
        denom_h = pred_bin.sum(dim=(2, 3)) + matched_gt_masks.sum(dim=(2, 3))
        dice_h = (2.0 * inter_h + smooth) / (denom_h + smooth)  # (B, K)

        present = matched_gt_masks.sum(dim=(2, 3)) > 0  # (B, K) class has pixels in GT
        is_tumour = (gt_idx >= 1) & (gt_idx <= 3)
        tumour_valid = present & is_tumour

        # Classifier accuracy on present real classes (0..3)
        pred_cls = matched_preds[..., :4].argmax(dim=-1)  # (B, K)
        cls_valid = present & (gt_idx <= 3)
        cls_correct = (pred_cls == gt_idx) & cls_valid

        # Volume MAE and centroid L2: report from whichever source is being optimised.
        vol_abs_err = (pred_vol - target_vol).abs()                              # (B, K)
        centroid_l2 = (pred_coords - target_coords).pow(2).sum(dim=-1).sqrt()   # (B, K)

        def _sc(t: Tensor, mask: Tensor) -> tuple[float, int]:
            n = int(mask.sum().item())
            s = float(t[mask].sum().item()) if n > 0 else 0.0
            return s, n

        metrics: dict[str, tuple[float, int]] = {
            "mean_tumour_dice": _sc(dice_h, tumour_valid),
            "classifier_acc": _sc(cls_correct.float(), cls_valid),
            "volume_mae": _sc(vol_abs_err, tumour_valid),
            "centroid_l2": _sc(centroid_l2, tumour_valid),
        }
        for c in (1, 2, 3):
            metrics[f"dice_class_{c}"] = _sc(dice_h, present & (gt_idx == c))

    return loss, classifier_loss, recon_loss, mask_dice_loss, metrics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    num_slots: int = 5,
    classification_weight: float = 0.7,
    mask_weight: float = 0.1,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    bce_mask: bool = True,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.5,
    focal_alpha_per_class: Tensor | None = None,
    grad_clip: float = 1.0,
    weakly_supervised: bool = False,
    matching: str = "spatial_dice",
    weak_coord_weight: float = 0.0,
    mask_entropy_weight: float = 0.0,
    use_mlp_coord_vol: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss, total_class_loss, total_recon_loss, total_mask_loss, count = 0.0, 0.0, 0.0, 0.0, 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    skipped_batches = 0

    num_batches = len(dataloader)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y_label = batch_seg_to_slot_targets_2d(y, num_slots).float().to(device)
        y = y.to(device)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast('cuda', enabled=(training and scaler is not None)):
                recon_combined, _, masks, _, y_hat = model(x)

            y_hat = y_hat.float()
            recon_combined = recon_combined.float()
            masks = masks.float()

            loss, class_loss, recon_loss, mask_loss, batch_metrics = slot_attention_loss(
                y_hat, y_label, num_slots, recon_combined, x, masks, y, classification_weight, mask_weight,
                bce_mask=bce_mask, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
                focal_alpha_per_class=focal_alpha_per_class,
                weakly_supervised=weakly_supervised,
                matching=matching,
                weak_coord_weight=weak_coord_weight,
                mask_entropy_weight=mask_entropy_weight,
                use_mlp_coord_vol=use_mlp_coord_vol,
            )

        if not torch.isfinite(loss):
            skipped_batches += 1
            if training:
                optimizer.zero_grad(set_to_none=True)
            continue

        if training:
            scaled_loss = loss / accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_last_batch = (step + 1) == num_batches
            if (step + 1) % accumulation_steps == 0 or is_last_batch:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                if scaler is not None:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    step_was_skipped = scale_after < scale_before
                    if scheduler is not None and not step_was_skipped:
                        scheduler.step()
                else:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                optimizer.zero_grad(set_to_none=True)

        count += x.shape[0]
        total_loss += loss.item() * x.shape[0]
        total_class_loss += class_loss.item() * x.shape[0]
        total_recon_loss += recon_loss.item() * x.shape[0]
        total_mask_loss += mask_loss.item() * x.shape[0]

        for name, (s, n) in batch_metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + s
            metric_counts[name] = metric_counts.get(name, 0) + n

    if skipped_batches:
        print(f"  WARNING: skipped {skipped_batches}/{num_batches} batches with non-finite loss")

    if count == 0:
        return {"loss": float("nan"), "classifier_loss": float("nan"), "recon_loss": float("nan"), "mask_loss": float("nan")}

    results = {
        "loss": total_loss / count,
        "classifier_loss": total_class_loss / count,
        "recon_loss": total_recon_loss / count,
        "mask_loss": total_mask_loss / count,
    }
    for name, s in metric_sums.items():
        n = metric_counts.get(name, 0)
        results[name] = s / n if n > 0 else float("nan")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_h", type=int, default=240)
    parser.add_argument("--input_w", type=int, default=240)
    parser.add_argument("--in_channels", type=int, default=1)
    # model
    parser.add_argument("--num_slots", type=int, default=5)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--slot_dim", type=int, default=64)
    parser.add_argument("--routing_iters", type=int, default=3)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--encoder_depth", type=int, default=4,
                        help="Stride-2 encoder layers: 4=15×15 slot grid (default), 3=30×30 slot grid.")
    parser.add_argument("--enc3_init_skip", action="store_true",
                        help="Add enc3 skip connection at initial 30×30 decode resolution (depth=3 only).")
    parser.add_argument("--use_mask_pool_classifier", action="store_true",
                        help="Replace slot-vector class head with mask-pooled enc3 feature classifier.")
    # loss
    parser.add_argument("--classification_weight", type=float, default=2.0)
    parser.add_argument("--mask_weight", type=float, default=0.1)
    # training
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset to first N samples (None = use all)")
    parser.add_argument("--no_val_split", action="store_true")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Fraction of patients held out entirely for final evaluation (default 0.15). "
                             "Set to 0.0 to use all patients for training.")
    parser.add_argument("--acc_steps", type=int, default=1)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_classification_weight_annealing", action="store_true")
    parser.add_argument("--anneal_epochs", type=int, default=None,
                        help="Epochs over which classification weight anneals from 0 to full. "
                             "Default: epochs//3. Overrides --no_classification_weight_annealing.")
    parser.add_argument("--no_bce_mask", action="store_true",
                        help="Use Dice-only mask loss (no BCE term). Default adds BCE to penalise diffuse masks.")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal loss exponent for class BCE (0 = standard BCE, 2 = standard focal loss).")
    parser.add_argument("--focal_alpha", type=float, default=0.5,
                        help="Per-positive-class alpha weight for focal BCE (0.5 = uniform, 0.75 = tumour-boosted). "
                             "Only applied when --focal_gamma > 0.")
    parser.add_argument("--focal_alpha_bg", type=float, default=None,
                        help="Alpha for background class (dim 0) in focal BCE. When set, uses per-class alpha "
                             "[focal_alpha_bg, focal_alpha, focal_alpha, focal_alpha] instead of scalar alpha.")
    parser.add_argument("--focal_gamma_start", type=float, default=None,
                        help="Initial focal gamma for annealing. If set, gamma anneals linearly from this "
                             "value to --focal_gamma over --focal_gamma_anneal_epochs epochs.")
    parser.add_argument("--focal_gamma_anneal_epochs", type=int, default=500,
                        help="Epochs to anneal focal_gamma from focal_gamma_start to focal_gamma (default 500).")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (default 1.0). Lower values reduce recon-loss spikes.")
    parser.add_argument("--weakly_supervised", action="store_true",
                        help="Weak regime: loss = CW·class_loss + recon only. Centroid/volume/Dice logged but not optimised.")
    parser.add_argument("--matching", type=str, default="spatial_dice",
                        choices=["spatial_dice", "centroid", "class_only"],
                        help="Hungarian matching cost: spatial_dice (full GT mask Dice+BCE), "
                             "centroid (class BCE + pred centroid vs GT centroid L2, no raw mask), "
                             "class_only (class BCE only, no spatial signal).")
    parser.add_argument("--weak_coord_weight", type=float, default=0.0,
                        help="Weight for coord+vol anchor in weak loss (0 = disabled). "
                             "Provides a light spatial anchor to break uniform-mask collapse.")
    parser.add_argument("--mask_entropy_weight", type=float, default=0.0,
                        help="Weight for entropy-min + coverage-floor regularizer (anti-WTA). "
                             "0 = disabled. Typical value 0.1 for weak regime.")
    parser.add_argument("--use_mlp_coord_vol", action="store_true",
                        help="Predict centroid and volume from MLP heads (slot vectors) instead of mask spatial "
                             "properties. In weak mode, coord+vol are always included in classifier_loss so the "
                             "slot vectors learn spatial properties even when masks are unsupervised.")
    # Kept for backward compatibility with older slurm scripts; superseded by --matching spatial_dice (default).
    parser.add_argument("--weak_keep_gt_matching", action="store_true")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Fraction of patients held out as test set (currently unused, kept for compat).")
    parser.add_argument("--early_stop_patience", type=int, default=15,
                        help="Stop if val mean tumour Dice does not improve for this many validations (each = 4 epochs). Large value disables.")
    args = parser.parse_known_args()[0]

    focal_alpha_per_class: Tensor | None = None
    if args.focal_alpha_bg is not None:
        focal_alpha_per_class = torch.tensor(
            [args.focal_alpha_bg, args.focal_alpha, args.focal_alpha, args.focal_alpha]
        )

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "This code requires CUDA support for GPU acceleration."
    print("GPU model:", torch.cuda.get_device_name(0))

    free_vram, total_vram = torch.cuda.mem_get_info()
    print(f"Total VRAM: {total_vram / 1024**3:.2f} GB")
    print(f"Free VRAM:  {free_vram / 1024**3:.2f} GB")

    cwd = Path.cwd()
    print("Current working directory:", cwd)
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    seed_all(args.seed, deterministic=args.deterministic)
    model_dir = "/path/to/BrainWear_Kareem/FYP/slot_attention/training_2d/models/checkpoints"
    checkpoint_dir = (
        f"{model_dir}/brats_png_{args.exp_name}/"
        if args.checkpoint_dir is None
        else args.checkpoint_dir
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Two instances over the same (deterministically ordered) samples: the train
    # instance applies flips/affine augmentation (is_train=True), the val instance
    # does not. They enumerate identically, so split indices align across both.
    train_full = BraTS2020PNGDataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        is_train=True,
    )
    val_full = BraTS2020PNGDataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        is_train=False,
    )

    if args.no_val_split:
        # Overfit path: no augmentation, single dataset used for training.
        datasets = {"train": val_full, "val": None}
    else:
        # Group sample indices by patient
        patient_to_indices: dict[str, list[int]] = {}
        for i, (t2_path, _) in enumerate(val_full.samples):
            patient_id = Path(t2_path).parent.name
            patient_to_indices.setdefault(patient_id, []).append(i)

        # Split patients (not slices): hold out the first test_split fraction sequentially,
        # then apply a seeded 80/20 random train/val split within the remaining pool.
        patients = sorted(patient_to_indices.keys())
        n_test = int(args.test_split * len(patients))
        test_patients = patients[:n_test]
        train_pool    = patients[n_test:]

        if test_patients:
            print(f"Held-out test patients ({len(test_patients)}): {test_patients}")

        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(train_pool), generator=rng).tolist()
        n_train = int(0.8 * len(train_pool))
        train_indices = [i for p in [train_pool[j] for j in perm[:n_train]] for i in patient_to_indices[p]]
        val_indices   = [i for p in [train_pool[j] for j in perm[n_train:]] for i in patient_to_indices[p]]

        datasets = {
            "train": Subset(train_full, train_indices),
            "val": Subset(val_full, val_indices)
        }

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    if args.deterministic:
        kwargs["worker_init_fn"] = partial(seed_worker, base_seed=args.seed, rank=0)

    loader_generator = None
    if args.deterministic:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)

    dataloaders = {
        split: DataLoader(
            datasets[split],
            shuffle=(split == "train"),
            drop_last=(split == "train" and not args.no_val_split),
            generator=loader_generator,
            **kwargs,
        )
        for split in ["train", "val"]
        if datasets[split] is not None
    }

    model = SlotClassifier2D(
        in_shape=(args.in_channels, args.input_h, args.input_w),
        width=args.width,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        routing_iters=args.routing_iters,
        temperature=args.temp,
        encoder_depth=args.encoder_depth,
        enc3_init_skip=args.enc3_init_skip,
        use_mask_pool_classifier=args.use_mask_pool_classifier,
    )
    if args.deterministic:
        model.set_deterministic_slot_init(args.seed)
    model.to(device)

    # model = torch.compile(model)
    # model = torch.compile(model, backend="aot_eager")


    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        fused=True
    )

    accumulation_steps = args.acc_steps
    scaler = None if args.no_amp else torch.amp.GradScaler('cuda')

    batches_per_epoch = len(dataloaders["train"])
    effective_steps_per_epoch = math.ceil(batches_per_epoch / args.acc_steps)
    total_steps = args.epochs * effective_steps_per_epoch
    warmup_steps = int(total_steps * 0.05)
    decay_steps = total_steps - warmup_steps

    print(f"Validation split enabled: {not args.no_val_split}")
    print(f"Steps per epoch: {effective_steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} ({(warmup_steps/total_steps)*100:.1f}%)")
    print(f"Decay steps: {decay_steps}\n")

    scheduler = get_slot_attention_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        decay_rate=0.1,
        decay_steps=decay_steps,
    )

    checkpoint_store_path = os.path.join(checkpoint_dir, "ckpt.pt")
    if args.anneal_epochs is not None:
        anneal_epochs = max(1, args.anneal_epochs)
    elif args.no_classification_weight_annealing:
        anneal_epochs = 1
    else:
        anneal_epochs = max(1, args.epochs // 3)
    last_epoch = 1
    best_loss = 1e3
    best_dice = -1.0          # checkpoint selection metric for the val path
    evals_since_improve = 0   # early-stopping counter (counts validations)
    patience = args.early_stop_patience
    prev_val_loss = 20
    if args.use_checkpoint is not None:
        checkpoint_load_path = Path(args.use_checkpoint).expanduser()
        if checkpoint_load_path.is_dir():
            checkpoint_load_path = checkpoint_load_path / "ckpt.pt"
        print(f"Loading checkpoint from {checkpoint_load_path}...")
        assert checkpoint_load_path.exists(), "Checkpoint file doesn't exist"
        checkpoint = torch.load(checkpoint_load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"Resuming from Epoch {last_epoch - 1} with Best Loss: {best_loss:.4f}")
    else:
        model.apply(init_params)

    print(f"#params: {sum(p.numel() for p in model.parameters()):,}")
    for k in sorted(vars(args)):
        print(f"--{k}={vars(args)[k]}")

    print("\nRunning sanity check...")
    _ = run_epoch(
        model, dataloaders["train"],
        num_slots=args.num_slots, mask_weight=args.mask_weight,
        bce_mask=not args.no_bce_mask, focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha, focal_alpha_per_class=focal_alpha_per_class,
        weakly_supervised=args.weakly_supervised, matching=args.matching,
        weak_coord_weight=args.weak_coord_weight, mask_entropy_weight=args.mask_entropy_weight,
        use_mlp_coord_vol=args.use_mlp_coord_vol,
    )
    prev_val_metrics = {
        "loss": prev_val_loss, "classifier_loss": 0.0, "recon_loss": 0.0, "mask_loss": 0.0,
        "mean_tumour_dice": 0.0, "classifier_acc": 0.0, "volume_mae": 0.0, "centroid_l2": 0.0,
        "dice_class_1": 0.0, "dice_class_2": 0.0, "dice_class_3": 0.0,
    }

    run = wandb.init(
        entity="km1422-imperial-college-london",
        project="jupyterhub-training",
        settings=wandb.Settings(console="auto"),
        resume="never",
        config={
            "architecture": "Slot attention 2D",
            "dataset": "BraTS2020 PNG",
            "name": args.exp_name,
            "epochs": args.epochs,
            "max_samples": args.max_samples,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "acc_steps": args.acc_steps,
            "routing_iters": args.routing_iters,
            "classification_weight": args.classification_weight,
            "mask_weight": args.mask_weight,
            "use_checkpoint": args.use_checkpoint,
            "amp": not args.no_amp,
            "no_val_split": args.no_val_split,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "focal_alpha_bg": args.focal_alpha_bg,
            "grad_clip": args.grad_clip,
            "encoder_depth": args.encoder_depth,
            "enc3_init_skip": args.enc3_init_skip,
            "use_mask_pool_classifier": args.use_mask_pool_classifier,
            "focal_gamma_start": args.focal_gamma_start,
            "focal_gamma_anneal_epochs": args.focal_gamma_anneal_epochs,
            "world_size": 1,
            "temp": args.temp,
            "input_h": args.input_h,
            "input_w": args.input_w,
            "weakly_supervised": args.weakly_supervised,
            "matching": args.matching,
            "weak_coord_weight": args.weak_coord_weight,
            "mask_entropy_weight": args.mask_entropy_weight,
            "use_mlp_coord_vol": args.use_mlp_coord_vol,
        },
    )

    for epoch in range(last_epoch, args.epochs + 1):
        current_classification_weight = min(args.classification_weight, args.classification_weight * (epoch / anneal_epochs))

        if args.focal_gamma_start is not None:
            t = min(1.0, epoch / max(1, args.focal_gamma_anneal_epochs))
            current_focal_gamma = args.focal_gamma_start + (args.focal_gamma - args.focal_gamma_start) * t
        else:
            current_focal_gamma = args.focal_gamma

        epoch_start = time.time()
        train_metrics = run_epoch(
            model, dataloaders["train"], optimizer, scheduler,
            args.num_slots, current_classification_weight, args.mask_weight, scaler=scaler,
            accumulation_steps=accumulation_steps, bce_mask=not args.no_bce_mask,
            focal_gamma=current_focal_gamma, focal_alpha=args.focal_alpha,
            focal_alpha_per_class=focal_alpha_per_class, grad_clip=args.grad_clip,
            weakly_supervised=args.weakly_supervised, matching=args.matching,
            weak_coord_weight=args.weak_coord_weight, mask_entropy_weight=args.mask_entropy_weight,
            use_mlp_coord_vol=args.use_mlp_coord_vol,
        )

        if "val" in dataloaders and epoch % 4 == 0:
            val_metrics = run_epoch(
                model, dataloaders["val"],
                num_slots=args.num_slots, classification_weight=current_classification_weight, mask_weight=args.mask_weight,
                bce_mask=not args.no_bce_mask, focal_gamma=current_focal_gamma, focal_alpha=args.focal_alpha,
                focal_alpha_per_class=focal_alpha_per_class,
                weakly_supervised=args.weakly_supervised, matching=args.matching,
                weak_coord_weight=args.weak_coord_weight, mask_entropy_weight=args.mask_entropy_weight,
                use_mlp_coord_vol=args.use_mlp_coord_vol,
            )
            prev_val_metrics = val_metrics

            # Select on mean tumour Dice (what AA-CBR's centroid/volume depend on),
            # not on total loss which is dominated by reconstruction.
            current_dice = val_metrics["mean_tumour_dice"]
            improved = (current_dice == current_dice) and current_dice > best_dice  # NaN-safe
            if improved and epoch >= anneal_epochs:
                best_dice = current_dice
                best_loss = val_metrics["loss"]
                evals_since_improve = 0
                save_dict = {
                    "hyperparameters": vars(args),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "best_dice": best_dice,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                }
                print(f"Epoch {epoch}: New best tumour Dice {best_dice:.4f} (loss {best_loss:.4f}) - checkpoint saved.")
                torch.save(save_dict, checkpoint_store_path)
            elif epoch >= anneal_epochs:
                evals_since_improve += 1
                if evals_since_improve >= patience:
                    print(f"Epoch {epoch}: early stopping after {patience} validations without Dice improvement (best {best_dice:.4f}).")
                    break
        elif "val" not in dataloaders and epoch % 4 == 0:
            if train_metrics["loss"] < best_loss and epoch >= anneal_epochs:
                best_loss = train_metrics["loss"]
                save_dict = {
                    "hyperparameters": vars(args),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                }
                print(f"Epoch {epoch}: New best loss {best_loss:.5f} - checkpoint saved.")
                torch.save(save_dict, checkpoint_store_path)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        current_wd = optimizer.param_groups[0]["weight_decay"]

        run.log({
            "train/loss": train_metrics["loss"],
            "train/classifier_loss": train_metrics["classifier_loss"],
            "train/recon_loss": train_metrics["recon_loss"],
            "train/mask_loss": train_metrics["mask_loss"],

            "train/mean_tumour_dice": train_metrics.get("mean_tumour_dice", float("nan")),
            "train/classifier_acc": train_metrics.get("classifier_acc", float("nan")),
            "train/volume_mae": train_metrics.get("volume_mae", float("nan")),
            "train/centroid_l2": train_metrics.get("centroid_l2", float("nan")),
            "train/dice_class_1": train_metrics.get("dice_class_1", float("nan")),
            "train/dice_class_2": train_metrics.get("dice_class_2", float("nan")),
            "train/dice_class_3": train_metrics.get("dice_class_3", float("nan")),

            "val/loss": prev_val_metrics["loss"],
            "val/classifier_loss": prev_val_metrics["classifier_loss"],
            "val/recon_loss": prev_val_metrics["recon_loss"],
            "val/mask_loss": prev_val_metrics["mask_loss"],

            "val/mean_tumour_dice": prev_val_metrics.get("mean_tumour_dice", float("nan")),
            "val/classifier_acc": prev_val_metrics.get("classifier_acc", float("nan")),
            "val/volume_mae": prev_val_metrics.get("volume_mae", float("nan")),
            "val/centroid_l2": prev_val_metrics.get("centroid_l2", float("nan")),
            "val/dice_class_1": prev_val_metrics.get("dice_class_1", float("nan")),
            "val/dice_class_2": prev_val_metrics.get("dice_class_2", float("nan")),
            "val/dice_class_3": prev_val_metrics.get("dice_class_3", float("nan")),

            "optim/learning_rate": current_lr,
            "optim/weight_decay": current_wd,
            "optim/temp": args.temp,
            "optim/classification_weight": current_classification_weight,
            "optim/focal_gamma": current_focal_gamma,

            "train_loss": train_metrics["loss"],
            "val_loss": prev_val_metrics["loss"],
            "epoch_time": epoch_time,
        })

    run.finish()