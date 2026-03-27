import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.losses import DiceLoss, HybridDiceBCELoss, FocalLoss, BoundaryLoss, CombinedLoss
from models.losses_enhanced import DeepSupervisionLoss, CombinedSegLoss


class TestDiceLoss:
    def test_dice_loss_forward(self, sample_batch, device):
        _, masks = sample_batch
        masks = masks.to(device)
        
        loss_fn = DiceLoss()
        pred = torch.randn(masks.shape).to(device)
        
        loss = loss_fn(pred, masks)
        
        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss.item() <= 1
    
    def test_dice_loss_perfect_prediction(self, device):
        loss_fn = DiceLoss()
        pred = torch.ones(2, 1, 64, 64).to(device)
        target = torch.ones(2, 1, 64, 64).to(device)
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1
    
    def test_dice_loss_empty_mask(self, device):
        loss_fn = DiceLoss()
        pred = torch.zeros(2, 1, 64, 64).to(device)
        target = torch.zeros(2, 1, 64, 64).to(device)
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1


class TestHybridDiceBCELoss:
    def test_hybrid_loss_forward(self, sample_batch, device):
        _, masks = sample_batch
        masks = masks.to(device)
        
        loss_fn = HybridDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        pred = torch.randn(masks.shape).to(device)
        
        loss = loss_fn(pred, masks)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_hybrid_loss_weights(self, device):
        loss_fn = HybridDiceBCELoss(dice_weight=1.0, bce_weight=0.0)
        pred = torch.randn(2, 1, 64, 64).to(device)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0


class TestFocalLoss:
    def test_focal_loss_forward(self, sample_batch, device):
        _, masks = sample_batch
        masks = masks.to(device)
        
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(masks.shape).to(device)
        
        loss = loss_fn(pred, masks)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_focal_loss_class_imbalance(self, device):
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
        target = torch.zeros(2, 1, 64, 64).to(device)
        target[:, :, :32, :] = 1
        pred = torch.randn(2, 1, 64, 64).to(device)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0


class TestBoundaryLoss:
    def test_boundary_loss_forward(self, sample_batch, device):
        _, masks = sample_batch
        masks = masks.to(device)
        
        loss_fn = BoundaryLoss()
        pred = torch.randn(masks.shape).to(device)
        
        loss = loss_fn(pred, masks)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestCombinedLoss:
    def test_combined_loss_forward(self, sample_batch, device):
        _, masks = sample_batch
        masks = masks.to(device)
        
        loss_fn = CombinedLoss()
        pred = torch.randn(masks.shape).to(device)
        
        loss = loss_fn(pred, masks)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestDeepSupervisionLoss:
    def test_deep_supervision_loss(self, device):
        loss_fn = DeepSupervisionLoss(dice_weight=0.5, bce_weight=0.5)
        
        outputs = [torch.randn(2, 1, 224, 224).to(device),
                   torch.randn(2, 1, 112, 112).to(device),
                   torch.randn(2, 1, 56, 56).to(device)]
        target = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
        
        loss = loss_fn(outputs, target)
        assert loss.item() > 0


class TestCombinedSegLoss:
    def test_combined_seg_loss(self, device):
        loss_fn = CombinedSegLoss(dice_weight=0.4, bce_weight=0.4, lovasz_weight=0.2)
        
        pred = torch.randn(2, 1, 224, 224).to(device)
        target = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0 or loss.item() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])