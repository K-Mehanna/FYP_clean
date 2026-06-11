import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

QUANTILES = ["q25", "q38", "q50", "q62", "q75"]


class BraTSPNGDatasetWithOS(Dataset):
    """BraTS2020 dataset loaded from 2D PNG T2 slices with overall-survival labels.

    Each patient contributes 5 quantile axial T2 slices stacked into a single
    (5, H, W) tensor so the sample count and CV splits are identical to the
    3D NIfTI baseline. Segmentation masks are not used (matching the 3D
    end-to-end trainer which only uses `batch[0]` = T2 and `batch[-1]` = label).

    PNG structure:
        <root_dir>/BraTS20_Training_xxx/{id}_t2_q{25,38,50,62,75}.png

    Args:
        root_dir: Directory containing one sub-folder per patient.
        score_file: Path to BraTS_OS.csv (Brats20ID, Age, Survival_days, ...).
        num_bins: Number of class bins for discretising survival days.
        quantile_bins: If True, use equal-frequency bins; else equal-width.
        regression: If True, return raw float scores instead of class indices.
        max_patients: Truncate to first N patients (for debugging).
        shuffle: Shuffle patient list before truncation.
        target_size: Optional (H, W) to resize slices. None keeps originals.
        augment: If True, apply random flips and affine transforms.
    """

    def __init__(
        self,
        root_dir: str,
        score_file: str | None = None,
        num_bins: int = 5,
        quantile_bins: bool = False,
        regression: bool = False,
        max_patients: int | None = None,
        shuffle: bool = False,
        target_size: tuple[int, int] | None = None,
        augment: bool = False,
        allowed_patients: set | None = None,
    ):
        self.root_dir = root_dir
        self.num_bins = num_bins
        self.quantile_bins = quantile_bins
        self.regression = regression
        self.target_size = target_size
        self.augment = augment

        all_folders = sorted(
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        )
        if allowed_patients is not None:
            all_folders = [f for f in all_folders if f in allowed_patients]
        if shuffle:
            random.shuffle(all_folders)
        self.patient_folders = all_folders[:max_patients] if max_patients is not None else all_folders

        self.score_file = score_file
        self.outcome_scores: dict[str, float] = {}
        self._load_scores(score_file)

        if score_file is not None:
            self.patient_folders = [
                pid for pid in self.patient_folders if pid in self.outcome_scores
            ]

        self.score_values = pd.Series(list(self.outcome_scores.values()))
        self.bin_boundaries = (
            self._quantile_bin_boundaries(self.score_values)
            if self.quantile_bins
            else self._equal_size_bin_boundaries(self.score_values)
        )
        print(f"Bin boundaries for discretising scores: {self.bin_boundaries}")

    def __len__(self) -> int:
        return len(self.patient_folders)

    def get_class_labels(self) -> list[int]:
        return [self._score_to_class(self.outcome_scores[pid]) for pid in self.patient_folders]

    def _score_to_class(self, score: float) -> int:
        if score < self.bin_boundaries[0]:
            return 0
        for i in range(len(self.bin_boundaries) - 1):
            if self.bin_boundaries[i] <= score < self.bin_boundaries[i + 1]:
                return i
        return self.num_bins - 1

    def _equal_size_bin_boundaries(self, values: pd.Series) -> list[float]:
        if not len(values):
            return [0.0, 1.0]
        mn, mx = values.min(), values.max()
        step = (mx - mn) / self.num_bins
        return [mn + i * step for i in range(self.num_bins + 1)]

    def _quantile_bin_boundaries(self, values: pd.Series) -> list[float]:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            raise ValueError("No valid numeric values found for binning")
        _, edges = pd.qcut(clean, q=self.num_bins, retbins=True, duplicates="drop")
        edges_formatted = [float(f"{e:.2f}") for e in edges]
        assert len(edges_formatted) == self.num_bins + 1
        return edges_formatted

    def _load_scores(self, score_file: str | None) -> None:
        if score_file is None:
            return
        try:
            with open(score_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if not fieldnames or (fieldnames[0] or "").strip().lower() != "brats20id":
                    print(f"Error: '{score_file}' must have 'Brats20ID' as first column.")
                    return
                patient_col = fieldnames[0]
                survival_col = next(
                    (n for n in fieldnames if (n or "").strip().lower() == "survival_days"),
                    None,
                )
                if survival_col is None:
                    print(f"Error: '{score_file}' has no 'Survival_days' column.")
                    return
                for row in reader:
                    pid = (row.get(patient_col) or "").strip()
                    if not pid:
                        continue
                    val = str(row.get(survival_col, "")).strip()
                    if not val:
                        continue
                    try:
                        self.outcome_scores[pid] = float(val)
                    except ValueError:
                        pass
        except FileNotFoundError:
            print(f"Error: score file '{score_file}' not found.")
        except Exception as e:
            print(f"Unexpected error loading scores: {e}")

    def __getitem__(self, idx: int) -> tuple[Tensor, int | Tensor]:
        patient_id = self.patient_folders[idx]
        patient_dir = os.path.join(self.root_dir, patient_id)

        slices = []
        for q in QUANTILES:
            path = os.path.join(patient_dir, f"{patient_id}_t2_{q}.png")
            arr = np.array(Image.open(path), dtype=np.float32) / 255.0
            slices.append(arr)

        tensor = torch.from_numpy(np.stack(slices, axis=0))  # (5, H, W)

        if self.target_size is not None:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        if self.augment:
            import torchvision.transforms.functional as TF
            if random.random() > 0.5:
                tensor = TF.hflip(tensor)
            if random.random() > 0.5:
                tensor = TF.vflip(tensor)
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                tx = int(random.uniform(-0.05, 0.05) * tensor.shape[2])
                ty = int(random.uniform(-0.05, 0.05) * tensor.shape[1])
                scale = random.uniform(0.9, 1.1)
                tensor = TF.affine(
                    tensor, angle=angle, translate=[tx, ty], scale=scale, shear=0.0,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )

        if self.score_file is not None:
            score = self.outcome_scores.get(patient_id)
            if score is None:
                print(f"No outcome score for patient '{patient_id}'")
                return tensor, -1
            if self.regression:
                return tensor, torch.tensor(score, dtype=torch.float32)
            return tensor, self._score_to_class(score)

        return tensor, -1
