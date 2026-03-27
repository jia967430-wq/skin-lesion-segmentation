"""
Enhanced Loss Functions for Skin Lesion Segmentation
Supports deep supervision and combination of Dice + BCE + Lovasz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class DeepSupervisionLoss(nn.Module):
    """
    Loss with deep supervision support
    Combines losses from multiple decoder outputs
    """
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3, ds_weights=[1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.ds_weights = ds_weights
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        
        return 1 - dice
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: List of outputs [main_out, ds_out1, ds_out2, ds_out3] if deep supervision
                     or single tensor if no deep supervision
            target: Ground truth mask
        """
        # Ensure target has correct shape [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        if isinstance(outputs, list):
            # Deep supervision
            total_loss = 0
            for i, (out, weight) in enumerate(zip(outputs, self.ds_weights)):
                dice = self.dice_loss(out, target)
                bce = self.bce_loss(out, target)
                total_loss += weight * (self.dice_weight * dice + self.bce_weight * bce)
            return total_loss
        else:
            # Single output
            dice = self.dice_loss(outputs, target)
            bce = self.bce_loss(outputs, target)
            return self.dice_weight * dice + self.bce_weight * bce


class LovaszHingeLoss(nn.Module):
    """Lovasz Hinge Loss for segmentation (differentiable IoU surrogate)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        signs = 2. * target - 1.
        errors = 1. - pred * signs
        
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = target[perm]
        
        grad = self._lovasz_grad(errors_sorted)
        
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def _lovasz_grad(self, gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


class CombinedSegLoss(nn.Module):
    """
    Combined Segmentation Loss: Dice + BCE + Lovasz
    """
    
    def __init__(self, dice_weight=0.6, bce_weight=0.2, lovasz_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszHingeLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        lovasz = self.lovasz_loss(torch.sigmoid(pred), target)
        
        return (self.dice_weight * dice + 
                self.bce_weight * bce + 
                self.lovasz_weight * lovasz)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice with control over FP/FN"""
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        true_pos = (pred * target).sum()
        false_neg = (target * (1 - pred)).sum()
        false_pos = ((1 - target) * pred).sum()
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        return 1 - tversky