from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .constants import (
    SPATIAL_MIDPOINT,
    VOLUME_MEDIUM_THRESHOLD,
    VOLUME_MEDIUM_V2_THRESHOLD,
    VOLUME_SMALL_THRESHOLD,
    VOLUME_VERY_SMALL_THRESHOLD,
    VOLUME_2D_VERY_SMALL_THRESHOLD,
    VOLUME_2D_SMALL_THRESHOLD,
    VOLUME_2D_MEDIUM_V2_THRESHOLD,
    VOLUME_2D_MEDIUM_THRESHOLD,
)


class Characterisation(ABC):
    @classmethod
    @abstractmethod
    def less_specific(cls, case_a: NDArray, case_b: NDArray, strict: bool = True) -> NDArray:
        pass

    @classmethod
    @abstractmethod
    def default_case(cls) -> NDArray:
        pass

    @classmethod
    @abstractmethod
    def characterisation_transform(cls, predicted_scene: Tensor) -> NDArray:
        pass

    @classmethod
    @abstractmethod
    def feature_names(cls) -> list[str]:
        """Return a human-readable name for each index of the feature vector."""

    @staticmethod
    def apply_to_all(
        cases: NDArray, apply_func: Callable[[NDArray], NDArray]
    ) -> NDArray:
        if cases.ndim == 1:
            return apply_func(cases)
        return np.array([apply_func(cases[i]) for i in range(cases.shape[0])])


class TumourCharacterisationLarge(Characterisation):
    """72D concept space: region(3) x x_side(2) x y_side(2) x z_side(2) x volume(3).
    Slot format: [class_0, class_1, class_2, class_3, z_centroid, y_centroid, x_centroid, volume]
    """
    # 72D space may be too sparse - may need to reduce by merging volume bins or spatial bins
    DIMS = 72  # 3 * 2 * 2 * 2 * 3

    @classmethod
    def default_case(cls) -> NDArray:
        return np.zeros(cls.DIMS, dtype=int)

    @classmethod
    def less_specific(cls, case_a: NDArray, case_b: NDArray, strict: bool = True) -> NDArray:
        if case_a.ndim == 1:
            case_a = case_a[np.newaxis, :]  # Convert (F,) to (1, F)
        if case_b.ndim == 1:
            case_b = case_b[np.newaxis, :]
        if strict:
            return np.logical_and(
                np.all(case_a <= case_b, axis=-1), np.any(case_a < case_b, axis=-1)
            )
        else:
            return np.all(case_a <= case_b, axis=-1)

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        # predicted_slots: (num_slots, 8)
        # slot layout: [class_0, class_1, class_2, class_3, z, y, x, volume]
        counts = cls.default_case()

        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue  # background slot
            region = class_idx - 1  # maps NCR->0, ED->1, ET->2

            z_centroid = slot[4].item()
            y_centroid = slot[5].item()
            x_centroid = slot[6].item()
            volume = slot[7].item()

            x_bin = 0 if x_centroid < SPATIAL_MIDPOINT else 1   # Left / Right
            y_bin = 0 if y_centroid < SPATIAL_MIDPOINT else 1   # Anterior / Posterior
            z_bin = 0 if z_centroid < SPATIAL_MIDPOINT else 1   # Inferior / Superior

            if volume < VOLUME_SMALL_THRESHOLD:
                vol_bin = 0
            elif volume < VOLUME_MEDIUM_THRESHOLD:
                vol_bin = 1
            else:
                vol_bin = 2

            base_idx = (region * 24) + (x_bin * 12) + (y_bin * 6) + (z_bin * 3)

            # Cumulative volume encoding
            # Every real tumor is at least "Small"
            counts[base_idx + 0] += 1

            # If it exceeds the small threshold, should also trigger "Medium"
            if volume >= VOLUME_SMALL_THRESHOLD:
                counts[base_idx + 1] += 1

            # If it exceeds the medium threshold, should also trigger "Large"
            if volume >= VOLUME_MEDIUM_THRESHOLD:
                counts[base_idx + 2] += 1

            # idx = (region * 24) + (x_bin * 12) + (y_bin * 6) + (z_bin * 3) + vol_bin
            # counts[idx] += 1

        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        x_sides = ["L", "R"]
        y_sides = ["Ant", "Post"]
        z_sides = ["Inf", "Sup"]
        levels = ["Small", "Med", "Large"]
        names = []
        for r in regions:
            for x in x_sides:
                for y in y_sides:
                    for z in z_sides:
                        for l in levels:
                            names.append(f"{r}-{x}-{y}-{z}-{l}")
        return names  # length 72

class TumourCharacterisationSmall(Characterisation):
    """
    9-D concept space: region(3) x cumulative_volume(3).
    Ditching spatial coordinates reduces sparsity so AA-CBR can actually generalise to proxy score.
    Slot format: [class_0, class_1, class_2, class_3, z, y, x, volume]
    """
    DIMS = 9 

    @classmethod
    def default_case(cls) -> NDArray:
        return np.zeros(cls.DIMS, dtype=int)

    @classmethod
    def less_specific(cls, case_a: NDArray, case_b: NDArray, strict: bool = True) -> NDArray:
        if case_a.ndim == 1:
            case_a = case_a[np.newaxis, :]  # Convert (F,) to (1, F)
        if case_b.ndim == 1:
            case_b = case_b[np.newaxis, :]
        if strict:
            return np.logical_and(
                np.all(case_a <= case_b, axis=-1), np.any(case_a < case_b, axis=-1)
            )
        else:
            return np.all(case_a <= case_b, axis=-1)

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()

        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue  # Skip background slot
                
            region = class_idx - 1  # 0=NCR, 1=ED, 2=ET
            volume = slot[7].item()

            # Base index for this specific region (0, 3, or 6)
            base_idx = region * 3

            # Cumulative volume encoding
            # Every real tumor is at least "Small"
            counts[base_idx + 0] += 1
            
            # If it exceeds the small threshold, should also trigger "Medium"
            if volume >= VOLUME_SMALL_THRESHOLD:
                counts[base_idx + 1] += 1
                
            # If it exceeds the medium threshold, should also trigger "Large"
            if volume >= VOLUME_MEDIUM_THRESHOLD:
                counts[base_idx + 2] += 1

        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        levels = ["Small", "Med", "Large"]
        return [f"{r}-{l}" for r in regions for l in levels]


class TumourCharacterisationSmallV2(Characterisation):
    """15D concept space: region(3) x cumulative_volume(5).

    Extends TumourCharacterisationSmall from 3 to 5 volume levels by adding
    p10 (VOLUME_VERY_SMALL) and p50 (VOLUME_MEDIUM_V2) thresholds between the
    existing p25/p75 boundaries. Increases unique feature vectors from ~19 to
    ~50-80 on BraTS, reducing casebase collapse under deduplication.

    Slot format: [class_0, class_1, class_2, class_3, z, y, x, volume]
    """
    DIMS = 15  # 3 regions × 5 cumulative volume levels

    @classmethod
    def default_case(cls) -> NDArray:
        return np.zeros(cls.DIMS, dtype=int)

    @classmethod
    def less_specific(cls, case_a: NDArray, case_b: NDArray, strict: bool = True) -> NDArray:
        if case_a.ndim == 1:
            case_a = case_a[np.newaxis, :]
        if case_b.ndim == 1:
            case_b = case_b[np.newaxis, :]
        if strict:
            return np.logical_and(
                np.all(case_a <= case_b, axis=-1), np.any(case_a < case_b, axis=-1)
            )
        else:
            return np.all(case_a <= case_b, axis=-1)

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()
        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue
            region = class_idx - 1  # 0=NCR, 1=ED, 2=ET
            volume = slot[7].item()
            base_idx = region * 5
            counts[base_idx + 0] += 1                              # always (any presence)
            if volume >= VOLUME_VERY_SMALL_THRESHOLD:
                counts[base_idx + 1] += 1                          # >= p10
            if volume >= VOLUME_SMALL_THRESHOLD:
                counts[base_idx + 2] += 1                          # >= p25
            if volume >= VOLUME_MEDIUM_V2_THRESHOLD:
                counts[base_idx + 3] += 1                          # >= p50
            if volume >= VOLUME_MEDIUM_THRESHOLD:
                counts[base_idx + 4] += 1                          # >= p75
        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        levels = ["Any", "VSmall", "Small", "MedV2", "Med"]
        return [f"{r}-{l}" for r in regions for l in levels]


class TumourCharacterisationLargeV2(Characterisation):
    """120D concept space: region(3) x x_side(2) x y_side(2) x z_side(2) x volume(5).

    Extends TumourCharacterisationLarge from 3 to 5 volume levels using the same
    additional thresholds as SmallV2. Spatial binning (2×2×2) is unchanged.
    Increases unique feature vectors beyond 116/236 on BraTS.

    Slot format: [class_0, class_1, class_2, class_3, z, y, x, volume]
    """
    DIMS = 120  # 3 regions × 2×2×2 spatial × 5 cumulative volume levels

    @classmethod
    def default_case(cls) -> NDArray:
        return np.zeros(cls.DIMS, dtype=int)

    @classmethod
    def less_specific(cls, case_a: NDArray, case_b: NDArray, strict: bool = True) -> NDArray:
        if case_a.ndim == 1:
            case_a = case_a[np.newaxis, :]
        if case_b.ndim == 1:
            case_b = case_b[np.newaxis, :]
        if strict:
            return np.logical_and(
                np.all(case_a <= case_b, axis=-1), np.any(case_a < case_b, axis=-1)
            )
        else:
            return np.all(case_a <= case_b, axis=-1)

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()
        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue
            region = class_idx - 1  # 0=NCR, 1=ED, 2=ET

            z_centroid = slot[4].item()
            y_centroid = slot[5].item()
            x_centroid = slot[6].item()
            volume = slot[7].item()

            x_bin = 0 if x_centroid < SPATIAL_MIDPOINT else 1
            y_bin = 0 if y_centroid < SPATIAL_MIDPOINT else 1
            z_bin = 0 if z_centroid < SPATIAL_MIDPOINT else 1

            base_idx = (region * 40) + (x_bin * 20) + (y_bin * 10) + (z_bin * 5)

            counts[base_idx + 0] += 1                              # always (any presence)
            if volume >= VOLUME_VERY_SMALL_THRESHOLD:
                counts[base_idx + 1] += 1                          # >= p10
            if volume >= VOLUME_SMALL_THRESHOLD:
                counts[base_idx + 2] += 1                          # >= p25
            if volume >= VOLUME_MEDIUM_V2_THRESHOLD:
                counts[base_idx + 3] += 1                          # >= p50
            if volume >= VOLUME_MEDIUM_THRESHOLD:
                counts[base_idx + 4] += 1                          # >= p75
        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        x_sides = ["L", "R"]
        y_sides = ["Ant", "Post"]
        z_sides = ["Inf", "Sup"]
        levels = ["Any", "VSmall", "Small", "MedV2", "Med"]
        names = []
        for r in regions:
            for x in x_sides:
                for y in y_sides:
                    for z in z_sides:
                        for l in levels:
                            names.append(f"{r}-{x}-{y}-{z}-{l}")
        return names  # length 120


class TumourCharacterisationLarge2D(TumourCharacterisationLarge):
    """36D concept space: region(3) x x_side(2) x y_side(2) x volume(3).
    2D slot layout: [class_0, class_1, class_2, class_3, y, x, volume].
    """
    DIMS = 36  # 3 * 2 * 2 * 3

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()

        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue  # background slot
            region = class_idx - 1  # NCR->0, ED->1, ET->2

            y_centroid = slot[4].item()
            x_centroid = slot[5].item()
            volume = slot[6].item()

            x_bin = 0 if x_centroid < SPATIAL_MIDPOINT else 1   # Left / Right
            y_bin = 0 if y_centroid < SPATIAL_MIDPOINT else 1   # Anterior / Posterior

            base_idx = (region * 12) + (x_bin * 6) + (y_bin * 3)

            counts[base_idx + 0] += 1
            if volume >= VOLUME_SMALL_THRESHOLD:
                counts[base_idx + 1] += 1
            if volume >= VOLUME_MEDIUM_THRESHOLD:
                counts[base_idx + 2] += 1

        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        x_sides = ["L", "R"]
        y_sides = ["Ant", "Post"]
        levels = ["Small", "Med", "Large"]
        return [
            f"{r}-{x}-{y}-{l}"
            for r in regions for x in x_sides for y in y_sides for l in levels
        ]  # length 36


class TumourCharacterisationSmall2D(TumourCharacterisationSmall):
    """9D concept space: region(3) x cumulative_volume(3). 2D slot layout."""

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()

        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue
            region = class_idx - 1
            volume = slot[6].item()  # 2D layout: volume at index 6

            base_idx = region * 3
            counts[base_idx + 0] += 1
            if volume >= VOLUME_SMALL_THRESHOLD:
                counts[base_idx + 1] += 1
            if volume >= VOLUME_MEDIUM_THRESHOLD:
                counts[base_idx + 2] += 1

        return counts


class TumourCharacterisationSmall2DV2(TumourCharacterisationSmall2D):
    """15D concept space: region(3) x cumulative_volume(5).

    2D equivalent of TumourCharacterisationSmallV2. Uses 2D-calibrated volume
    thresholds (pixel-fraction percentiles from BraTS PNG dataset).
    Slot format: [class_0, class_1, class_2, class_3, y, x, volume]
    """
    DIMS = 15

    @classmethod
    def default_case(cls) -> NDArray:
        return np.zeros(cls.DIMS, dtype=int)

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()
        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue
            region = class_idx - 1
            volume = slot[6].item()
            base = region * 5
            counts[base + 0] += 1
            if volume >= VOLUME_2D_VERY_SMALL_THRESHOLD:
                counts[base + 1] += 1
            if volume >= VOLUME_2D_SMALL_THRESHOLD:
                counts[base + 2] += 1
            if volume >= VOLUME_2D_MEDIUM_V2_THRESHOLD:
                counts[base + 3] += 1
            if volume >= VOLUME_2D_MEDIUM_THRESHOLD:
                counts[base + 4] += 1
        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        levels  = ["Any", "VSmall", "Small", "MedV2", "Med"]
        return [f"{r}-{l}" for r in regions for l in levels]


class TumourCharacterisationLarge2DV2(TumourCharacterisationLarge2D):
    """60D concept space: region(3) x x_side(2) x y_side(2) x cumulative_volume(5).

    2D equivalent of TumourCharacterisationLargeV2. Drops the z-axis bin
    (not available in 2D slices), giving 3×2×2×5 = 60D instead of 120D.
    Uses 2D-calibrated volume thresholds.
    Slot format: [class_0, class_1, class_2, class_3, y, x, volume]
    Index stride: region×20, x_bin×10, y_bin×5.
    """
    DIMS = 60

    @classmethod
    def default_case(cls) -> NDArray:
        return np.zeros(cls.DIMS, dtype=int)

    @classmethod
    def characterisation_transform(cls, predicted_slots: Tensor) -> NDArray:
        counts = cls.default_case()
        for slot in predicted_slots:
            class_idx = torch.argmax(slot[0:4]).item()
            if class_idx == 0:
                continue
            region     = class_idx - 1
            y_centroid = slot[4].item()
            x_centroid = slot[5].item()
            volume     = slot[6].item()
            x_bin = 0 if x_centroid < SPATIAL_MIDPOINT else 1
            y_bin = 0 if y_centroid < SPATIAL_MIDPOINT else 1
            base  = (region * 20) + (x_bin * 10) + (y_bin * 5)
            counts[base + 0] += 1
            if volume >= VOLUME_2D_VERY_SMALL_THRESHOLD:
                counts[base + 1] += 1
            if volume >= VOLUME_2D_SMALL_THRESHOLD:
                counts[base + 2] += 1
            if volume >= VOLUME_2D_MEDIUM_V2_THRESHOLD:
                counts[base + 3] += 1
            if volume >= VOLUME_2D_MEDIUM_THRESHOLD:
                counts[base + 4] += 1
        return counts

    @classmethod
    def feature_names(cls) -> list[str]:
        regions = ["NCR", "ED", "ET"]
        x_sides = ["L", "R"]
        y_sides  = ["Ant", "Post"]
        levels   = ["Any", "VSmall", "Small", "MedV2", "Med"]
        return [
            f"{r}-{x}-{y}-{l}"
            for r in regions for x in x_sides for y in y_sides for l in levels
        ]