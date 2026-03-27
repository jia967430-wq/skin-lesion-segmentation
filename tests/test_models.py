import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNet, AttentionUNet, EnhancedAttentionUNet, CBAM, GatedMLP


class TestUNet:
    def test_unet_forward(self, sample_batch, device):
        images, _ = sample_batch
        images = images.to(device)
        
        model = UNet(in_channels=3, out_channels=1).to(device)
        output = model(images)
        
        assert output.shape == images.shape
        assert not torch.isnan(output).any()
    
    def test_unet_different_sizes(self, device):
        model = UNet(in_channels=3, out_channels=1).to(device)
        
        for size in [128, 224, 256]:
            x = torch.randn(1, 3, size, size).to(device)
            output = model(x)
            assert output.shape == (1, 1, size, size)
    
    def test_unet_base_filters(self, device):
        model = UNet(in_channels=3, out_channels=1, base_filters=32).to(device)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0


class TestAttentionUNet:
    def test_attention_unet_forward(self, sample_batch, device):
        images, _ = sample_batch
        images = images.to(device)
        
        model = AttentionUNet(in_channels=3, out_channels=1).to(device)
        output = model(images)
        
        assert output.shape == images.shape
    
    def test_attention_unet_lite(self, device):
        model = AttentionUNetLite(in_channels=3, out_channels=1, base_filters=16).to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        output = model(x)
        assert output.shape == (1, 1, 224, 224)


class TestEnhancedAttentionUNet:
    def test_enhanced_unet_forward(self, sample_batch, device):
        images, _ = sample_batch
        images = images.to(device)
        
        model = EnhancedAttentionUNet(in_channels=3, out_channels=1).to(device)
        output = model(images)
        
        assert output.shape == images.shape or output[0].shape == images.shape
    
    def test_enhanced_unet_deep_supervision(self, device):
        model = EnhancedAttentionUNet(in_channels=3, out_channels=1, deep_supervision=True).to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        output = model(x)
        
        assert isinstance(output, list)
        assert len(output) > 1


class TestCBAM:
    def test_cbam_forward(self, device):
        cbam = CBAM(channels=64, reduction=16).to(device)
        x = torch.randn(2, 64, 32, 32).to(device)
        output = cbam(x)
        
        assert output.shape == x.shape
    
    def test_gated_mlp(self, device):
        gated = GatedMLP(64, 32).to(device)
        x = torch.randn(2, 64).to(device)
        output = gated(x)
        
        assert output.shape == (2, 32)


class TestModelInference:
    def test_model_output_range(self, device):
        model = UNet(in_channels=3, out_channels=1).to(device)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224).to(device)
            output = model(x)
            
            assert output.min() < 10 and output.max() > -10
    
    def test_model_gradients(self, device):
        model = UNet(in_channels=3, out_channels=1).to(device)
        x = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
        
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])