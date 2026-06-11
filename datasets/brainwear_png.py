import os
import csv
import re
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

QUANTILES = ["q25", "q38", "q50", "q62", "q75"]


class BrainWearPNGDataset(Dataset):
    """BrainWear dataset backed by 5-channel quantile-slice PNGs.

    Each patient folder contains an ``Axial_T2/`` subdirectory with five PNG
    files named ``Axial_T2_q25.png`` through ``Axial_T2_q75.png``.  These are
    stacked into a ``(5, H, W)`` tensor so that 2D ResNets can consume them.

    Args:
        root_dir: Parent directory containing one sub-folder per patient.
        score_file: CSV of EORTC outcome scores.
        num_bins: Number of discrete classes for binning the outcome score.
        quantile_bins: Use equal-frequency (quantile) bins; otherwise equal-width.
        regression: If True, ``__getitem__`` returns the raw float score instead
            of a class index.
        max_patients: Limit dataset to the first N patients (alphabetical order).
        shuffle: Shuffle patient list before applying ``max_patients``.
        target_size: Not used (slices are loaded at native resolution); kept for
            API symmetry with the 3D dataset.
        augment: Apply random affine augmentation in ``__getitem__``.
        score_name: Which EORTC scale to use as the prediction target (default
            ``"QL2"`` — Global Health Status).
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
        target_size=None,
        augment: bool = False,
        score_name: str = "QL2",
    ):
        self.root_dir = root_dir
        self.num_bins = num_bins
        self.quantile_bins = quantile_bins
        self.regression = regression
        self.augment = augment

        all_folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f)) and self._has_t2_png_dir(f)
        ])

        if shuffle:
            random.shuffle(all_folders)

        self.patient_folders = all_folders[:max_patients] if max_patients is not None else all_folders
        self.score_file = score_file

        self.outcome_scores: dict[str, dict[str, float]] = {}
        self._load_scores(score_file)

        self.score_name = score_name
        if score_file is not None:
            self.patient_folders = [
                pid for pid in self.patient_folders
                if pid in self.outcome_scores.get(self.score_name, {})
            ]

        if score_file is not None and self.patient_folders:
            self.score_values = pd.Series([
                self.outcome_scores[self.score_name][pid] for pid in self.patient_folders
            ])
        else:
            self.score_values = pd.Series([], dtype=float)

        self.bin_boundaries = (
            self._quantile_bin_boundaries(self.score_values)
            if self.quantile_bins
            else self._equal_size_bin_boundaries(self.score_values)
        )
        print(f"Bin boundaries for discretising scores: {self.bin_boundaries}")

    def __len__(self) -> int:
        return len(self.patient_folders)

    def get_class_labels(self) -> list[int]:
        return [
            self._score_to_class(self.outcome_scores[self.score_name][pid])
            for pid in self.patient_folders
        ]

    def _has_t2_png_dir(self, patient_id: str) -> bool:
        standard = os.path.join(self.root_dir, patient_id, "Axial_T2")
        if os.path.isdir(standard):
            return True
        patient_path = os.path.join(self.root_dir, patient_id)
        if not os.path.isdir(patient_path):
            return False
        numbered = [
            d for d in os.listdir(patient_path)
            if re.fullmatch(r"Axial_T2_\d+", d)
            and os.path.isdir(os.path.join(patient_path, d))
        ]
        return len(numbered) > 0

    def _resolve_t2_png_dir(self, patient_id: str) -> str:
        standard = os.path.join(self.root_dir, patient_id, "Axial_T2")
        if os.path.isdir(standard):
            return standard
        patient_path = os.path.join(self.root_dir, patient_id)
        numbered = sorted([
            os.path.join(patient_path, d)
            for d in os.listdir(patient_path)
            if re.fullmatch(r"Axial_T2_\d+", d)
            and os.path.isdir(os.path.join(patient_path, d))
        ])
        if numbered:
            return numbered[0]
        raise FileNotFoundError(f"No Axial_T2 directory found for patient '{patient_id}'")

    def _equal_size_bin_boundaries(self, values: pd.Series) -> list[float]:
        if self.num_bins is None or self.num_bins < 1:
            raise ValueError("num_bins must be >= 1")
        if values.empty:
            return []
        max_score = float(values.max())
        min_score = float(values.min())
        bin_size = (max_score - min_score) / self.num_bins
        return [min_score + i * bin_size for i in range(self.num_bins + 1)]

    def _quantile_bin_boundaries(self, values: pd.Series) -> list[float]:
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1")
        if values.empty:
            return []
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            raise ValueError("No valid numeric values found for binning")
        _, edges = pd.qcut(clean, q=self.num_bins, retbins=True, duplicates="drop")
        edges_formatted = [float(f"{edge:.2f}") for edge in edges]
        assert len(edges_formatted) == self.num_bins + 1, "Should be num_bins + 1 edges"
        return edges_formatted

    def _score_to_class(self, score: float) -> int:
        for i in range(len(self.bin_boundaries) - 1):
            if self.bin_boundaries[i] <= score < self.bin_boundaries[i + 1]:
                return i
        return self.num_bins - 1

    def _load_scores(self, score_file: str | None) -> None:
        if score_file is None:
            return
        try:
            with open(score_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    score_name = (row.get("Question") or "").strip()
                    if not score_name or score_name.lower().startswith("date"):
                        continue
                    if score_name not in self.outcome_scores:
                        self.outcome_scores[score_name] = {}
                    for patient_id, value in row.items():
                        if patient_id == "Question":
                            continue
                        patient_id = patient_id.strip()
                        clean = str(value).strip() if value else ""
                        if not clean:
                            continue
                        try:
                            self.outcome_scores[score_name][patient_id] = float(clean)
                        except ValueError:
                            print(f"Skip: '{clean}' for {patient_id} in {score_name} is not a number.")
        except FileNotFoundError:
            print(f"Error: The file '{score_file}' was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def __getitem__(self, idx: int) -> tuple[Tensor, int | float]:
        patient_id = self.patient_folders[idx]
        t2_dir = self._resolve_t2_png_dir(patient_id)

        series_name = os.path.basename(t2_dir)
        slices = []
        for q in QUANTILES:
            path = os.path.join(t2_dir, f"{series_name}_{q}.png")
            arr = np.array(Image.open(path), dtype=np.float32) / 255.0
            slices.append(torch.from_numpy(arr))

        x = torch.stack(slices, dim=0)  # (5, H, W)

        if self.augment and random.random() > 0.5:
            angle = random.uniform(-15.0, 15.0)
            tx = int(random.uniform(-0.05, 0.05) * x.shape[2])
            ty = int(random.uniform(-0.05, 0.05) * x.shape[1])
            scale = random.uniform(0.9, 1.1)
            x = TF.affine(
                x, angle=angle, translate=[tx, ty], scale=scale, shear=0.0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

        if self.score_file is not None:
            score = self.outcome_scores[self.score_name].get(patient_id)
            if score is None:
                print(f"No outcome score found for patient '{patient_id}'")
                return x, -1
            if self.regression:
                return x, score
            return x, self._score_to_class(score)

        return x, -1
