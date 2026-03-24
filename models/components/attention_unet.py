"""
UNet with CBAM Attention for Medical Image Segmentation

This model combines:
- U-Net encoder-decoder architecture
- Gated MLP blocks for feature refinement
- CBAM attention mechanisms for enhanced feature selection

Reference:
- CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
- U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv
from .mamba import GatedMLP


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class EncoderBlock(nn.Module):
    """Encoder block with DoubleConv and optional Gated MLP + CBAM"""
    
    def __init__(self, in_channels, out_channels, use_gated_mlp=True, use_attention=True):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        
        self.use_gated_mlp = use_gated_mlp
        self.use_attention = use_attention
        
        if use_gated_mlp:
            self.gated_mlp = GatedMLP(out_channels, d_state=16, expand=2)
        
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x):
        x = self.double_conv(x)
        
        if self.use_gated_mlp:
            x = self.gated_mlp(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_gated_mlp=True, use_attention=True):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
        
        self.use_gated_mlp = use_gated_mlp
        self.use_attention = use_attention
        
        if use_gated_mlp:
            self.gated_mlp = GatedMLP(out_channels, d_state=16, expand=2)
        
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        
        if self.use_gated_mlp:
            x = self.gated_mlp(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x


class AttentionUNet(nn.Module):
    """
    U-Net with Gated MLP and CBAM attention
    
    Architecture:
    - Encoder: 4 levels with DoubleConv + Gated MLP + CBAM
    - Bottleneck: DoubleConv + Gated MLP + CBAM
    - Decoder: 4 levels with skip connections
    
    Args:
        in_channels: Input image channels (3 for RGB)
        out_channels: Output segmentation channels (1 for binary)
        base_filters: Number of filters in first layer (64)
        use_gated_mlp: Whether to use gated MLP blocks
        use_attention: Whether to use CBAM attention
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, 
                 use_gated_mlp=True, use_attention=True, bilinear=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters, use_gated_mlp, use_attention)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, use_gated_mlp, use_attention)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, use_gated_mlp, use_attention)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8, use_gated_mlp, use_attention)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = EncoderBlock(base_filters * 8, base_filters * 16 // factor, use_gated_mlp, use_attention)
        
        # Decoder
        self.dec4 = DecoderBlock(base_filters * 16 // factor, base_filters * 8, 
                                  base_filters * 8 // factor, use_gated_mlp, use_attention)
        self.dec3 = DecoderBlock(base_filters * 8 // factor, base_filters * 4,
                                  base_filters * 4 // factor, use_gated_mlp, use_attention)
        self.dec2 = DecoderBlock(base_filters * 4 // factor, base_filters * 2,
                                  base_filters * 2 // factor, use_gated_mlp, use_attention)
        self.dec1 = DecoderBlock(base_filters * 2 // factor, base_filters,
                                  base_filters, use_gated_mlp, use_attention)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Output
        self.outc = nn.Conv2d(base_filters, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))
        
        # Decoder
        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        
        return self.outc(x)
    
    def get_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionUNetLite(nn.Module):
    """
    Lightweight Attention-UNet variant with reduced parameters
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        
        self.enc1 = EncoderBlock(in_channels, base_filters, use_gated_mlp=True, use_attention=True)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, use_gated_mlp=True, use_attention=True)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, use_gated_mlp=True, use_attention=True)
        
        self.bottleneck = EncoderBlock(base_filters * 4, base_filters * 8, use_gated_mlp=True, use_attention=True)
        
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4, use_gated_mlp=True, use_attention=True)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2, use_gated_mlp=True, use_attention=True)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters, base_filters, use_gated_mlp=True, use_attention=True)
        
        self.pool = nn.MaxPool2d(2)
        self.outc = nn.Conv2d(base_filters, out_channels, 1)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x = self.bottleneck(self.pool(x3))
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.outc(x)
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_name, config):
    """Factory function to create models"""
    models = {
        'unet': lambda: nn.Sequential(
        ),
        'attention_unet': lambda: AttentionUNet(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1),
            base_filters=config.get('base_filters', 64),
            use_gated_mlp=True,
            use_attention=True
        ),
        'attention_unet_lite': lambda: AttentionUNetLite(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1),
            base_filters=config.get('base_filters', 32)
        ),
    }
    
    if model_name == 'unet':
        from .unet import UNet
        return UNet(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1),
            base_filters=config.get('base_filters', 64)
        )
    elif model_name == 'attention_unet':
        return AttentionUNet(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1),
            base_filters=config.get('base_filters', 64),
            use_gated_mlp=True,
            use_attention=True
        )
    elif model_name == 'attention_unet_lite':
        return AttentionUNetLite(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 1),
            base_filters=config.get('base_filters', 32)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
