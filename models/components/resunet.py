"""
Baseline Model: ResUNet

A simple ResNet-style baseline for comparison.
Uses residual connections in encoder blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDoubleConv(nn.Module):
    """Residual double convolution block"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Residual connection
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        identity = self.res_conv(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        return F.relu(x + identity)


class ResUNet(nn.Module):
    """Residual U-Net baseline"""
    
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = ResDoubleConv(in_channels, base_filters)
        self.enc2 = ResDoubleConv(base_filters, base_filters*2)
        self.enc3 = ResDoubleConv(base_filters*2, base_filters*4)
        self.enc4 = ResDoubleConv(base_filters*4, base_filters*8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*16, 3, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*16, base_filters*16, 3, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, 2, stride=2)
        self.dec3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        
        # Output convs
        self.out_enc4 = ResDoubleConv(base_filters*8, base_filters*4)
        self.out_enc3 = ResDoubleConv(base_filters*4, base_filters*2)
        self.out_enc2 = ResDoubleConv(base_filters*2, base_filters)
        self.out_enc1 = ResDoubleConv(base_filters, base_filters)
        
        self.out_conv = nn.Conv2d(base_filters, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = self.out_enc4(d4 + e4)
        
        d3 = self.dec3(d4)
        d3 = self.out_enc3(d3 + e3)
        
        d2 = self.dec2(d3)
        d2 = self.out_enc2(d2 + e2)
        
        d1 = self.dec1(d2)
        d1 = self.out_enc1(d1 + e1)
        
        return self.out_conv(d1)


if __name__ == '__main__':
    model = ResUNet(3, 1, 32)
    params = sum(p.numel() for p in model.parameters())
    print(f"ResUNet: {params:,} params")
    
    # Test
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output: {out.shape}")
