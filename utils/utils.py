import logging
import os
import random
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

from slot_attention.slot_attention import SlotClassifier3D

def seed_all(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int, *, base_seed: int, rank: int = 0) -> None:
    worker_seed = (base_seed + rank * 10_000 + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def compute_cost_matrix(slot_objects: Tensor, labels: Tensor, size: int, coord_dim: int = 3) -> Tensor:
    batch_size, _, _ = labels.shape

    # Reshape the tensors to allow for each potential pairing to be computed
    # between the slot objects and the labels
    # slot_objects: (batch_size, num_slots, num_features) -> (batch_size * num_slots * size, num_features)
    slot_objects = (
        slot_objects.unsqueeze(2)
        .repeat(1, 1, size, 1)
        .reshape(-1, slot_objects.size(2))
    )
    # labels: (batch_size, num_objects, num_features) -> (batch_size * size * num_objects, num_features)
    labels = labels.unsqueeze(1).repeat(1, size, 1, 1).reshape(-1, labels.size(2))

    coord_end = 4 + coord_dim

    pred_class = slot_objects[:, :4]
    pred_coords = slot_objects[:, 4:coord_end]
    pred_vol = slot_objects[:, coord_end:]

    target_class = labels[:, :4]
    target_coords = labels[:, 4:coord_end]
    target_vol = labels[:, coord_end:]

    # Sanitise class tensors before BCE: clamp does not remove NaN/Inf.
    pred_class = torch.nan_to_num(pred_class, nan=0.5, posinf=1.0, neginf=0.0)
    target_class = torch.nan_to_num(target_class, nan=0.0, posinf=1.0, neginf=0.0)
    # Clamp predicted probabilities to avoid log(0) in BCE.
    pred_class = torch.clamp(pred_class, min=1e-7, max=1.0 - 1e-7)
    target_class = torch.clamp(target_class, min=0.0, max=1.0)

    # Sanitise coordinate and volume predictions (AMP overflow can produce Inf/NaN).
    pred_coords = torch.nan_to_num(pred_coords, nan=0.0, posinf=1.0, neginf=0.0)
    pred_vol = torch.nan_to_num(pred_vol, nan=0.0, posinf=1.0, neginf=0.0)

    # Multiply by weights, then take the mean across the class dimension
    # cost_class = (bce_loss * class_weights).mean(dim=-1)
    cost_class = F.binary_cross_entropy(pred_class, target_class, reduction="none").mean(dim=-1)

    # Regression Cost (MSE for coordinates and volume)
    cost_coords = F.mse_loss(pred_coords, target_coords, reduction="none").mean(dim=-1)
    cost_vol = F.mse_loss(pred_vol, target_vol, reduction="none").mean(dim=-1)

    # Combine the costs
    total_cost = cost_class + cost_coords + cost_vol
    total_cost = torch.nan_to_num(total_cost, nan=10.0, posinf=10.0, neginf=0.0)
    # Reshape back to the 2D cost matrix per batch: (batch_size, size, size)
    cost_matrix = total_cost.view(batch_size, size, size)

    return cost_matrix

def compute_cost_matrix_2d(
    slot_objects: Tensor, 
    labels: Tensor, 
    size: int, 
    coord_dim: int = 3,
    pred_masks: Tensor = None,
    target_masks: Tensor = None
) -> Tensor:
    batch_size, _, _ = labels.shape

    # slot_objects: (B, num_slots, num_features) -> (B * num_slots * size, num_features)
    slot_objects_exp = (
        slot_objects.unsqueeze(2)
        .repeat(1, 1, size, 1)
        .reshape(-1, slot_objects.size(2))
    )
    # labels: (B, num_objects, num_features) -> (B * size * num_objects, num_features)
    labels_exp = labels.unsqueeze(1).repeat(1, size, 1, 1).reshape(-1, labels.size(2))

    coord_end = 4 + coord_dim

    # Extract RAW LOGITS from the network prediction
    raw_pred_class = slot_objects_exp[:, :4]
    raw_pred_coords = slot_objects_exp[:, 4:coord_end]
    raw_pred_vol = slot_objects_exp[:, coord_end:]

    target_class = labels_exp[:, :4]
    target_coords = labels_exp[:, 4:coord_end]
    target_vol = labels_exp[:, coord_end:]

    # Apply Sigmoid to convert raw logits to probabilities [0.0, 1.0] BEFORE clamping
    pred_class = torch.sigmoid(raw_pred_class)
    pred_coords = torch.sigmoid(raw_pred_coords)
    pred_vol = torch.sigmoid(raw_pred_vol)

    # Sanitise and clamp safely to prevent log(0) BCE explosions
    pred_class = torch.nan_to_num(pred_class, nan=0.5, posinf=1.0, neginf=0.0)
    pred_class = torch.clamp(pred_class, min=1e-7, max=1.0 - 1e-7)
    
    target_class = torch.nan_to_num(target_class, nan=0.0, posinf=1.0, neginf=0.0)
    target_class = torch.clamp(target_class, min=0.0, max=1.0)

    pred_coords = torch.nan_to_num(pred_coords, nan=0.0, posinf=1.0, neginf=0.0)
    pred_vol = torch.nan_to_num(pred_vol, nan=0.0, posinf=1.0, neginf=0.0)

    # Calculate unreduced 1D losses
    cost_class = F.binary_cross_entropy(pred_class, target_class, reduction="none").mean(dim=-1)
    cost_coords = F.mse_loss(pred_coords, target_coords, reduction="none").mean(dim=-1)
    cost_vol = F.mse_loss(pred_vol, target_vol, reduction="none").mean(dim=-1)

    # Combine costs
    total_cost = cost_class + cost_coords + cost_vol
    total_cost = torch.nan_to_num(total_cost, nan=10.0, posinf=10.0, neginf=0.0)

    # Reshape back to the 2D cost matrix per batch: (batch_size, num_slots, num_objects)
    cost_matrix = total_cost.view(batch_size, size, size)

    # Mathematically prevents the Hungarian algorithm from thrashing slots
    if pred_masks is not None and target_masks is not None:
        # pred_masks shape: (B, num_slots, 1, H, W) -> squeeze out channel -> (B, num_slots, H*W)
        p_mask = pred_masks.squeeze(2).flatten(start_dim=2)
        
        # target_masks shape: (B, num_objects, H, W) -> (B, num_objects, H*W)
        t_mask = target_masks.float().flatten(start_dim=2)
        
        num_slots_p = p_mask.shape[1]
        num_slots_t = t_mask.shape[1]
        p_mask_exp = torch.clamp(p_mask.unsqueeze(2), 1e-7, 1.0 - 1e-7).expand(-1, -1, num_slots_t, -1)
        t_mask_exp = t_mask.unsqueeze(1).expand(-1, num_slots_p, -1, -1)
        
        # Calculate dice score (Intersection over Sum) instead of BCE
        intersection = (p_mask_exp * t_mask_exp).sum(dim=-1)
        denominator = p_mask_exp.sum(dim=-1) + t_mask_exp.sum(dim=-1)
        
        # Laplace Smoothing: 1.0 prevents empty masks from blowing up the cost
        smooth = 1.0
        # Convert Dice Score (1.0 = perfect match) to Dice Cost (0.0 = perfect match)
        cost_mask_dice = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
        
        # Apply heavy weight to force the Hungarian matcher to respect physical shapes
        cost_matrix = cost_matrix + (10.0 * cost_mask_dice)

    return cost_matrix

def compute_cost_matrix_2d_spatial(
    slot_objects: Tensor, 
    labels: Tensor, 
    size: int, 
    coord_dim: int = 2,
    pred_masks: Tensor = None,
    target_masks: Tensor = None
) -> Tensor:
    
    # pred_masks: (B, num_slots, 1, H, W) -> (B, num_slots, H*W)
    p_mask = pred_masks.squeeze(2).flatten(start_dim=2)
    # target_masks: (B, num_objects, H, W) -> (B, num_objects, H*W)
    t_mask = target_masks.float().flatten(start_dim=2)
    
    p_mask_exp = p_mask.unsqueeze(2)
    t_mask_exp = t_mask.unsqueeze(1)
    
    intersection = (p_mask_exp * t_mask_exp).sum(dim=-1)
    denominator = p_mask_exp.sum(dim=-1) + t_mask_exp.sum(dim=-1)
    
    smooth = 1.0
    # Dice cost ranges from 0.0 (perfect) to 1.0 (terrible)
    cost_matrix = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
    
    return cost_matrix

def compute_cost_matrix_2d_centroid(
    slot_objects: Tensor,
    labels: Tensor,
    size: int,
    pred_masks: Tensor,
    centroid_weight: float = 2.0,
) -> Tensor:
    """Matching cost: class BCE + predicted-mask centroid vs GT centroid L2.

    Uses GT centroids from labels (derived summary statistics), not raw GT mask pixels.
    pred_masks: (B, K, 1, H, W) — predicted soft masks from the decoder.
    labels[..., 4:6]: (B, K_gt, 2) — GT (y, x) centroids normalised to [0, 1].
    """
    # Class BCE: (B, K_pred, K_gt)
    cp = torch.clamp(torch.sigmoid(slot_objects[..., :4]), 1e-7, 1.0 - 1e-7)
    gt_cls = torch.clamp(labels[..., :4], 0.0, 1.0)
    bce_cost = -(
        gt_cls.unsqueeze(1) * torch.log(cp.unsqueeze(2)) +
        (1 - gt_cls.unsqueeze(1)) * torch.log(1 - cp.unsqueeze(2))
    ).mean(dim=-1)

    # Predicted centroid from soft-mask center-of-mass
    pm = pred_masks.squeeze(2)  # (B, K, H, W)
    H, W = pm.shape[2:]
    gy = torch.linspace(0, 1, H, device=pm.device).view(1, 1, H, 1)
    gx = torch.linspace(0, 1, W, device=pm.device).view(1, 1, 1, W)
    ms = pm.sum(dim=(2, 3)).clamp(min=1e-6)
    pred_centroids = torch.stack(
        [(pm * gy).sum(dim=(2, 3)) / ms,
         (pm * gx).sum(dim=(2, 3)) / ms],
        dim=-1,
    )  # (B, K_pred, 2)

    gt_centroids = labels[..., 4:6]  # (B, K_gt, 2) — summary stat, no raw pixels

    # Pairwise L2: (B, K_pred, K_gt)
    diff = pred_centroids.unsqueeze(2) - gt_centroids.unsqueeze(1)
    cost_centroid = diff.pow(2).sum(dim=-1).sqrt()

    cost_matrix = bce_cost + centroid_weight * cost_centroid
    return torch.nan_to_num(cost_matrix, nan=10.0, posinf=10.0, neginf=0.0)


def compute_cost_matrix_2d_class_only(
    slot_objects: Tensor,
    labels: Tensor,
    size: int,
) -> Tensor:
    """Matching cost using class BCE only — no GT mask pixels, no spatial statistics."""
    cp = torch.clamp(torch.sigmoid(slot_objects[..., :4]), 1e-7, 1.0 - 1e-7)
    gt_cls = torch.clamp(labels[..., :4], 0.0, 1.0)
    bce_cost = -(
        gt_cls.unsqueeze(1) * torch.log(cp.unsqueeze(2)) +
        (1 - gt_cls.unsqueeze(1)) * torch.log(1 - cp.unsqueeze(2))
    ).mean(dim=-1)
    return torch.nan_to_num(bce_cost, nan=10.0, posinf=10.0, neginf=0.0)


# Taken from https://github.com/addtt/object-centric-library/utils/slot_matching.py
def hungarian_algorithm(cost_matrix: Tensor) -> tuple[Tensor, Tensor]:
    # Finds the optimal matching of predicted objects with the ground truth
    # objects using the cost matrix in order to minimize the final loss. Indices
    # is the list of tuples of indices of the matching of the ground truth with
    # the model's predicted objects.
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.tensor(
        np.array(indices), device=cost_matrix.device, dtype=torch.long
    )

    # Extract the costs of each match from the matching algorithm
    matched_costs = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    ).to(cost_matrix.device)

    return matched_costs, indices


def slot_feature_pred_exceeds_threshold(
    pred_features: Tensor, threshold: float, obj_info: dict
) -> bool:
    # Minimum confidence found across all slots' predicted object properties
    min_confidence = 1
    # The number of labels for each object property
    feature_sizes = list(obj_info.values())

    # Iterate through each slot prediction
    for slot in pred_features:
        current_position = 0

        # The confidence of all of the object property predictions for this slot
        total_confidence = 1

        for size in feature_sizes:
            # Extract the index for the predicted object property
            feature = slot[current_position : current_position + size]
            feature_index = torch.argmax(feature).item()

            # Find the confidence of this prediction
            feature_confidence = feature[feature_index].item()

            # Update the total confidence
            total_confidence *= feature_confidence
            current_position += size

        # Update the minimum confidence
        min_confidence = min(min_confidence, total_confidence)

    return min_confidence > threshold


def select_kmeans_centroid_datapoints(
    model: SlotClassifier3D, dataloader: DataLoader, k: int, seed: int
) -> NDArray:
    model.eval()
    # Enable deterministic slot initialization for reproducible results
    model.set_deterministic_slot_init(seed)

    slot_representations = []
    for x, _ in dataloader:
        # T2 scans are already z-score normalized from preprocessing
        x = x.cuda()

        with torch.no_grad():
            # Forward pass
            _, _, _, slots, _ = model(x)
            slots = slots.cpu().numpy()

        for i in range(x.shape[0]):
            # Concatenate the slots to be passed into the KMeans algorithm
            concat_slots = np.concatenate(slots[i], axis=0)
            slot_representations.append(concat_slots)

    # Flatten the slot representations to be 2D
    flattened_slots = np.array(slot_representations).reshape(
        len(slot_representations), -1
    )

    # Perform KMeans clustering on the slot representations
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(flattened_slots)

    # Get the cluster centers and find the indices of the closest datapoint to each center
    cluster_centers = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(cluster_centers, flattened_slots)

    return closest


def load_model(model_path: str, model: nn.Module) -> nn.Module:
    # Load the model state dict
    ckpt = torch.load(model_path)

    # Load the model state dict into the model
    model.load_state_dict(ckpt["model_state_dict"])

    return model


def get_slot_attention_scheduler(
    optimizer: Optimizer, 
    warmup_steps: int = 10000, 
    decay_rate: float = 0.5, 
    decay_steps: int = 100000
) -> LambdaLR:
    """
    Creates a learning rate scheduler with linear warmup and exponential decay.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Exponential decay
            return decay_rate ** ((current_step - warmup_steps) / decay_steps)

    return LambdaLR(optimizer, lr_lambda)

def get_slot_attention_scheduler_epochs(
    optimizer: Optimizer, 
    warmup_epochs: int = 5, 
    decay_rate: float = 0.05, 
    decay_epochs: int = 145
) -> LambdaLR:
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            return decay_rate ** ((current_epoch - warmup_epochs) / max(1, decay_epochs))
    return LambdaLR(optimizer, lr_lambda)