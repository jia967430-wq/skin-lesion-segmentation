"""
Gate MLP and Attention Blocks for Medical Image Segmentation

Reference:
- CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
- Gated networks (Highway networks, Squeeze-and-Excitation networks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GatedMLP(nn.Module):
    """Gated MLP Block for feature gating."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        bias: bool = False
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        
        self.conv1x1 = nn.Conv2d(d_model, self.d_inner, kernel_size=1, bias=bias)
        self.conv1x1_out = nn.Conv2d(self.d_inner, d_model, kernel_size=1, bias=bias)
        self.gate_weight = nn.Parameter(torch.zeros(1))
        self.silu = nn.SiLU(inplace=True)
        
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.conv1x1.weight)
        nn.init.xavier_uniform_(self.conv1x1_out.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.conv1x1(x)
        x = self.silu(x)
        x = self.conv1x1_out(x)
        
        gate = torch.sigmoid(self.gate_weight)
        out = gate * x + (1 - gate) * identity
        
        return out


class SpatialGatedConv(nn.Module):
    """Spatial-aware module with depthwise convolution."""
    
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        expand: int = 2
    ) -> None:
        super().__init__()
        
        self.gated_conv = GatedMLP(channels, d_state, expand)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3,
                padding=1, groups=channels, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gated_conv(x)
        x = self.depthwise_conv(x)
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ) -> None:
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        avg_out = self.channel_fc(self.avg_pool(x).view(b, c))
        max_out = self.channel_fc(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial, _ = x.max(dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_concat))
        x = x * spatial_att
        
        return x


class AttentionBlock(nn.Module):
    """Combined attention block for encoder/decoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_ssm: bool = True,
        use_attention: bool = True,
        d_state: int = 16
    ) -> None:
        super().__init__()
        
        self.use_ssm = use_ssm
        self.use_attention = use_attention
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if use_ssm:
            self.gated_mlp = GatedMLP(out_channels, d_state=d_state)
        
        if use_attention:
            self.attention = CBAM(out_channels)
        
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        
        x = self.conv(x)
        
        if self.use_ssm:
            x = self.gated_mlp(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return F.relu(x + identity)


def create_attention_block(
    channels: int,
    use_ssm: bool = True,
    use_attention: bool = True,
    d_state: int = 16
) -> nn.Module:
    """Factory function to create attention block."""
    return AttentionBlock(
        channels, channels,
        use_ssm=use_ssm,
        use_attention=use_attention,
        d_state=d_state
    )
