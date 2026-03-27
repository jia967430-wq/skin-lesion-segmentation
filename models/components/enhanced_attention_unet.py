"""
Enhanced UNet with Advanced Attention for Skin Lesion Segmentation

Improvements over base AttentionUNet:
1. Residual connections in encoder/decoder blocks
2. Deep supervision
3. Spatial-channel squeeze-excitation attention
4. Improved feature fusion

Target: Dice > 0.91
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEBlock(nn.Module):
    """Spatial and Channel SE Block"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel SE
        self.cSE = SEBlock(channels, reduction)
        # Spatial SE
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        return cSE * sSE


class ResidualConvBlock(nn.Module):
    """Residual Convolution Block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection with 1x1 conv if dimensions differ
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class AttentionGate(nn.Module):
    """Attention Gate for skip connections"""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncoderBlock(nn.Module):
    """Enhanced Encoder Block with Residual Connections and Attention"""
    
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SCSEBlock(out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x, x  # Return both for skip connection


class DecoderBlock(nn.Module):
    """Enhanced Decoder Block with Attention Gate"""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ResidualConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(in_channels // 2, skip_channels, skip_channels // 2)
            self.attention = SCSEBlock(out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        if self.use_attention:
            skip = self.attention_gate(x, skip)
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x


class DeepSupervision(nn.Module):
    """Deep Supervision Module"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.conv(x)


class EnhancedAttentionUNet(nn.Module):
    """
    Enhanced Attention UNet for Skin Lesion Segmentation
    
    Key improvements:
    1. Residual connections
    2. SCSE attention (better than CBAM for segmentation)
    3. Attention gates for skip connections
    4. Deep supervision
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, deep_supervision=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = EncoderBlock(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters, base_filters)
        
        # Output
        self.outc = nn.Conv2d(base_filters, out_channels, 1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.ds4 = DeepSupervision(base_filters * 8, out_channels)
            self.ds3 = DeepSupervision(base_filters * 4, out_channels)
            self.ds2 = DeepSupervision(base_filters * 2, out_channels)
    
    def forward(self, x):
        input_size = x.size()
        
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(self.enc1.pool(x1))
        x3, skip3 = self.enc3(self.enc2.pool(x2))
        x4, skip4 = self.enc4(self.enc3.pool(x3))
        
        # Bottleneck
        x5, _ = self.bottleneck(self.enc4.pool(x4))
        
        # Decoder
        d4 = self.dec4(x5, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        # Main output
        main_out = self.outc(d1)
        
        if self.deep_supervision and self.training:
            # Deep supervision outputs
            ds4 = F.interpolate(self.ds4(d4), size=input_size[2:], mode='bilinear', align_corners=True)
            ds3 = F.interpolate(self.ds3(d3), size=input_size[2:], mode='bilinear', align_corners=True)
            ds2 = F.interpolate(self.ds2(d2), size=input_size[2:], mode='bilinear', align_corners=True)
            return [main_out, ds4, ds3, ds2]
        
        return main_out
    
    def get_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_enhanced_model(in_channels=3, out_channels=1, base_filters=64, deep_supervision=True):
    """Factory function to create enhanced attention UNet"""
    return EnhancedAttentionUNet(in_channels, out_channels, base_filters, deep_supervision)