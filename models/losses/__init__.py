import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred.squeeze(1))
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class HybridDiceBCELoss(nn.Module):
    """Combined Dice and BCE Loss"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target.float())
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred.squeeze(1))
        
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge segmentation"""
    
    def __init__(self, theta=0.5):
        super().__init__()
        self.theta = theta
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        target_dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        target_eroded = F.max_pool2d(1 - target, kernel_size=3, stride=1, padding=1)
        
        boundary_target = target_dilated - target_eroded
        
        pred_dilated = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
        pred_eroded = F.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1)
        boundary_pred = pred_dilated - pred_eroded
        
        boundary_loss = F.mse_loss(boundary_pred, boundary_target)
        
        return boundary_loss


class CombinedLoss(nn.Module):
    """Combined loss with multiple components"""
    
    def __init__(self, dice_weight=0.3, bce_weight=0.3, focal_weight=0.2, boundary_weight=0.2):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.boundary = BoundaryLoss()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target) +
                self.bce_weight * self.bce(pred, target) +
                self.focal_weight * self.focal(pred, target) +
                self.boundary_weight * self.boundary(pred, target))


def get_loss(config):
    """Get loss function from config"""
    loss_name = config.get('name', 'HybridDiceBCELoss')
    
    if loss_name == 'DiceLoss':
        return DiceLoss()
    elif loss_name == 'HybridDiceBCELoss':
        return HybridDiceBCELoss(
            dice_weight=config.get('dice_weight', 0.5),
            bce_weight=config.get('bce_weight', 0.5)
        )
    elif loss_name == 'FocalLoss':
        return FocalLoss(
            alpha=config.get('alpha', 0.25),
            gamma=config.get('gamma', 2.0)
        )
    elif loss_name == 'CombinedLoss':
        return CombinedLoss(
            dice_weight=config.get('dice_weight', 0.3),
            bce_weight=config.get('bce_weight', 0.3),
            focal_weight=config.get('focal_weight', 0.2),
            boundary_weight=config.get('boundary_weight', 0.2)
        )
    else:
        return HybridDiceBCELoss()
