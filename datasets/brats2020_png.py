import os
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

# BraTS segmentation label values mapped to contiguous class indices
BRATS_LABEL_TO_INDEX = {0: 0, 1: 1, 2: 2, 4: 3}
QUANTILES = ["q25", "q38", "q50", "q62", "q75"]


class BraTS2020PNGDataset(Dataset):
    """Dataset of 2D axial slices extracted from BraTS2020 as PNG files.

    Each sample is one (T2 slice, segmentation mask) pair.  Five quantile
    slices (q25 / q38 / q50 / q62 / q75) per patient are included. Slices
    with no tumour present (all-background segmentation) are excluded,
    giving roughly 369 x 4-5 usable samples depending on tumour extent.

    Args:
        data_dir: Root directory containing one sub-folder per patient, each
            holding files named {patient_id}_t2_{q}.png and
            {patient_id}_seg_{q}.png.
        max_samples: If set, only use the first N samples.
        shuffle: If True, shuffle the sample list before applying max_samples.
    """

    def __init__(self, data_dir: str, max_samples: int | None = None, shuffle: bool = False, is_train: bool = False):
        self.samples: list[tuple[str, str]] = []
        self.is_train = is_train

        for patient_id in sorted(os.listdir(data_dir)):
            patient_path = os.path.join(data_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue
            for q in QUANTILES:
                t2_path = os.path.join(patient_path, f"{patient_id}_t2_{q}.png")
                seg_path = os.path.join(patient_path, f"{patient_id}_seg_{q}.png")
                if not (os.path.exists(t2_path) and os.path.exists(seg_path)):
                    continue
                # Exclude slices where no tumour class is present
                seg_arr = np.array(Image.open(seg_path), dtype=np.uint8)
                if not np.any(seg_arr > 0):
                    continue
                self.samples.append((t2_path, seg_path))

        if shuffle:
            random.shuffle(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        t2_path, seg_path = self.samples[idx]

        # T2: already z-score normalised then min-max scaled to [0, 255] uint8
        t2_np = np.array(Image.open(t2_path), dtype=np.float32) / 255.0
        t2 = torch.from_numpy(t2_np).unsqueeze(0)  # (1, H, W)

        # Seg: uint8 with raw BraTS labels {0, 1, 2, 4}; map 4 → 3
        seg_np = np.array(Image.open(seg_path), dtype=np.int64)
        seg_mapped = np.zeros_like(seg_np)
        for raw_label, class_idx in BRATS_LABEL_TO_INDEX.items():
            seg_mapped[seg_np == raw_label] = class_idx
        seg = torch.from_numpy(seg_mapped)  # (H, W)

        if self.is_train:
            # Random Flips
            if random.random() > 0.5:
                t2 = TF.hflip(t2)
                seg = TF.hflip(seg)
            if random.random() > 0.5:
                t2 = TF.vflip(t2)
                seg = TF.vflip(seg)

            # Random Affine (Rotation, Translation, Scaling)
            if random.random() > 0.5:
                # Generate random parameters once
                angle = random.uniform(-15.0, 15.0)  # +/- 15 degrees
                translate_x = int(random.uniform(-0.05, 0.05) * t2.shape[2]) # 5% shift
                translate_y = int(random.uniform(-0.05, 0.05) * t2.shape[1])
                scale = random.uniform(0.9, 1.1)     # +/- 10% zoom
                
                # Apply to T2 (Bilinear)
                t2 = TF.affine(
                    t2, angle=angle, translate=[translate_x, translate_y], 
                    scale=scale, shear=0.0, 
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                
                # Apply to Mask 
                seg = TF.affine(
                    seg.unsqueeze(0), angle=angle, translate=[translate_x, translate_y], 
                    scale=scale, shear=0.0, 
                    interpolation=TF.InterpolationMode.NEAREST
                ).squeeze(0)

        return t2, seg

def seg_mask_2d_to_slot_targets(seg: Tensor, num_slots: int = 4) -> Tensor:
    H, W = seg.shape
    total_pixels = H * W
    num_features = 7  # 4 class + 2 coords + 1 volume
    targets = torch.zeros(num_slots, num_features)

    for class_idx in range(min(4, num_slots)):
        mask = seg == class_idx

        # Only set one-hot class bit and spatial attributes if the class physically exists.
        # Absent classes remain all-zeros: a clean "null object" signal.
        if mask.any():
            targets[class_idx, class_idx] = 1.0
            ys, xs = torch.where(mask)
            targets[class_idx, 4] = ys.float().mean() / H
            targets[class_idx, 5] = xs.float().mean() / W
            targets[class_idx, 6] = mask.sum().float() / total_pixels

    return targets


def batch_seg_to_slot_targets_2d(seg_batch: Tensor, num_slots: int) -> Tensor:
    """Apply seg_mask_2d_to_slot_targets to each sample in a batch.

    Args:
        seg_batch: Integer tensor of shape (B, H, W).
        num_slots: Number of slots per sample.

    Returns:
        Tensor of shape (B, num_slots, 7).
    """
    return torch.stack([
        seg_mask_2d_to_slot_targets(seg_batch[i], num_slots)
        for i in range(seg_batch.shape[0])
    ])
