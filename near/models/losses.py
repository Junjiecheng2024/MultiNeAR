"""
Connectivity-aware losses to reduce fragmentation in segmentation outputs.

Key ideas:
1. Total Variation (TV) Loss: Penalizes discontinuities
2. Morphological Smoothness: Encourages connected regions
3. Compactness Loss: Penalizes scattered predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TotalVariationLoss(nn.Module):
    """
    Total Variation loss to encourage spatial smoothness.
    Reduces fragmentation by penalizing rapid changes in predictions.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, K, D, H, W] probability predictions (after softmax)
        
        Returns:
            TV loss value
        """
        if pred.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,K,D,H,W], got {pred.shape}")
        
        # Compute differences along each spatial dimension
        tv_d = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).mean()
        tv_h = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]).mean()
        tv_w = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]).mean()
        
        return self.weight * (tv_d + tv_h + tv_w) / 3.0

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes excessive boundary pixels.
    Encourages compact, connected regions.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, K, D, H, W] probability predictions
            labels: [B, D, H, W] ground truth labels
        
        Returns:
            Boundary loss value
        """
        # Convert labels to one-hot
        K = pred.shape[1]
        labels_oh = F.one_hot(labels.long(), K).permute(0, 4, 1, 2, 3).float()
        
        # Compute boundary pixels (where neighbors differ)
        boundary_pred = self._compute_boundary(pred)
        boundary_gt = self._compute_boundary(labels_oh)
        
        # Penalize excessive boundaries in prediction
        # Encourage boundaries to match ground truth
        loss = F.mse_loss(boundary_pred, boundary_gt)
        
        return self.weight * loss
    
    def _compute_boundary(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Detect boundary voxels using gradient magnitude.
        
        Args:
            vol: [B, K, D, H, W]
        
        Returns:
            [B, K, D, H, W] boundary map
        """
        grad_d = torch.abs(vol[:, :, 1:, :, :] - vol[:, :, :-1, :, :])
        grad_h = torch.abs(vol[:, :, :, 1:, :] - vol[:, :, :, :-1, :])
        grad_w = torch.abs(vol[:, :, :, :, 1:] - vol[:, :, :, :, :-1])
        
        # Pad to original size
        grad_d = F.pad(grad_d, (0, 0, 0, 0, 0, 1))
        grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))
        grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))
        
        return (grad_d + grad_h + grad_w) / 3.0

class CompactnessLoss(nn.Module):
    """
    Compactness loss using morphological operations.
    Penalizes scattered predictions by comparing to morphologically closed version.
    """
    def __init__(self, kernel_size: int = 3, weight: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = weight
        
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, K, D, H, W] probability predictions
        
        Returns:
            Compactness loss
        """
        # Apply morphological closing (dilation followed by erosion)
        # This fills small holes and connects nearby regions
        closed = self._morphological_close(pred)
        
        # Penalize difference between prediction and its closed version
        # If prediction is already compact/connected, difference will be small
        loss = F.mse_loss(pred, closed)
        
        return self.weight * loss
    
    def _morphological_close(self, x: torch.Tensor) -> torch.Tensor:
        """
        Morphological closing using max pooling (dilation) and min pooling (erosion).
        """
        # Dilation: max pooling
        dilated = F.max_pool3d(
            x, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.kernel_size // 2
        )
        
        # Erosion: -min_pool = max_pool(-x)
        eroded = -F.max_pool3d(
            -dilated,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
        
        return eroded

class ConnectivityAwareLoss(nn.Module):
    """
    Combined connectivity-aware loss with multiple terms.
    
    Args:
        tv_weight: Weight for Total Variation loss
        boundary_weight: Weight for Boundary loss
        compactness_weight: Weight for Compactness loss
    """
    def __init__(
        self, 
        tv_weight: float = 0.1,
        boundary_weight: float = 0.05,
        compactness_weight: float = 0.05
    ):
        super().__init__()
        self.tv_loss = TotalVariationLoss(weight=tv_weight)
        self.boundary_loss = BoundaryLoss(weight=boundary_weight)
        self.compactness_loss = CompactnessLoss(weight=compactness_weight)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred: [B, K, D, H, W] probability predictions (after softmax)
            labels: [B, D, H, W] ground truth labels
        
        Returns:
            total_loss: Combined connectivity loss
            loss_dict: Dictionary with individual loss components
        """
        tv = self.tv_loss(pred)
        boundary = self.boundary_loss(pred, labels)
        compactness = self.compactness_loss(pred)
        
        total = tv + boundary + compactness
        
        loss_dict = {
            'connectivity/tv': tv.item(),
            'connectivity/boundary': boundary.item(),
            'connectivity/compactness': compactness.item(),
            'connectivity/total': total.item()
        }
        
        return total, loss_dict

# Simplified API for easy integration
def connectivity_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    tv_weight: float = 0.1,
    boundary_weight: float = 0.05,
    compactness_weight: float = 0.05
) -> tuple[torch.Tensor, dict]:
    """
    Compute connectivity-aware loss.
    
    Args:
        pred: [B, K, D, H, W] softmax probabilities
        labels: [B, D, H, W] ground truth labels
        tv_weight: Weight for smoothness (reduce discontinuities)
        boundary_weight: Weight for boundary consistency
        compactness_weight: Weight for region compactness
    
    Returns:
        loss: Total connectivity loss
        loss_dict: Individual loss components for logging
    
    Example usage in training:
        >>> logits = model(x)
        >>> probs = F.softmax(logits, dim=1)
        >>> conn_loss, conn_dict = connectivity_loss(probs, labels)
        >>> total_loss = dice_loss + ce_loss + conn_loss
    """
    loss_fn = ConnectivityAwareLoss(tv_weight, boundary_weight, compactness_weight)
    return loss_fn(pred, labels)
