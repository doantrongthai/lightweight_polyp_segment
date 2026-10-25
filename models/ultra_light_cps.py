import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


Swish = nn.SiLU


def _get_valid_num_groups(channels):
    for d in (8, 4, 2, 1):
        if channels % d == 0:
            return d
    return 1


# ---------------------------------------------------------------------------
# Encoder (ULC_Encoder): lightweight EfficientNet-style backbone built from
# depthwise-separable "MBConv" blocks with an ECA channel-attention gate.
# ---------------------------------------------------------------------------
class ULCDropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            keep_prob = 1 - self.drop_prob
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < keep_prob
            x = x / keep_prob * mask
        return x


class ECABlock(nn.Module):
    """Efficient Channel Attention: cheap 1D-conv channel gate."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class EfficientDepthwiseConv(nn.Module):
    """Expand (1x1) -> depthwise (kxk) -> project (1x1), a la MBConv."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1,
                      groups=math.gcd(in_channels, hidden_channels), bias=False),
            nn.BatchNorm2d(hidden_channels),
            Swish(),
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride, padding=1,
                      groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            Swish(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1,
                     groups=math.gcd(hidden_channels, out_channels), bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.depthwise(x)
        return self.pointwise(x)


class ULCConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, drop_path_prob=0.05):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        self.conv = nn.Sequential(
            EfficientDepthwiseConv(in_channels, out_channels, kernel_size, stride, expand_ratio),
            ECABlock(out_channels),
        )
        self.drop_path = ULCDropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = self.drop_path(out) + x
        return out


class ULCEncoder(nn.Module):
    """features[0]      : stem 3->32, stride 2
    features[1..7]    : 7 MBConv stages (stride pattern 1,2,2,2,1,2,1)
    classifier         : global-pool head (unused by the segmentation model)
    """

    def __init__(self, num_classes=1000, width_mult=1.0, depth_mult=0.3, dropout_rate=0.1):
        super().__init__()

        base_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_layers = [1, 1, 1, 1, 1, 1, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expand_ratios = [1, 3, 3, 3, 3, 3, 3]

        channels = [math.ceil(c * width_mult) for c in base_channels]
        layers = [math.ceil(l * depth_mult) for l in base_layers]

        self.features = nn.ModuleList()
        self.features.append(nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish(),
        ))

        in_channels = channels[0]
        for i in range(7):
            stage = []
            for j in range(layers[i]):
                stride = strides[i] if j == 0 else 1
                stage.append(ULCConvBlock(
                    in_channels=in_channels,
                    out_channels=channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    expand_ratio=expand_ratios[i],
                    drop_path_prob=0.05,
                ))
                in_channels = channels[i + 1]
            self.features.append(nn.Sequential(*stage))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[7], num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Bridge: compressed multi-scale feature module (MSFE) + CBAM attention.
# ---------------------------------------------------------------------------
class LightSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class MSFEBlock(nn.Module):
    """Compressed multi-scale feature extraction: three dilated separable
    branches plus an identity branch, fused back to `out_channels`."""

    def __init__(self, in_channels, out_channels, atrous_rates=(3, 5, 7)):
        super().__init__()
        mid_channels = out_channels // 4

        self.eca = ECABlock(in_channels)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.branch1 = LightSeparableConv2d(mid_channels, mid_channels,
                                            dilation=atrous_rates[0], padding=atrous_rates[0])
        self.branch2 = LightSeparableConv2d(mid_channels, mid_channels,
                                            dilation=atrous_rates[1], padding=atrous_rates[1])
        self.branch3 = LightSeparableConv2d(mid_channels, mid_channels,
                                            dilation=atrous_rates[2], padding=atrous_rates[2])

        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.eca(x)
        x = self.shared_conv(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = x  # identity branch

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.final_conv(out)
        out = out + shortcut
        return self.relu(out)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv2d(torch.cat([avg_out, max_out], dim=1)))
        return x * out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class MSFE_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.rfb = MSFEBlock(in_channels, out_channels, atrous_rates)
        self.cbam = CBAM(channels=out_channels)

    def forward(self, x):
        x = self.rfb(x)
        return self.cbam(x)


# ---------------------------------------------------------------------------
# Decoder: multi-scale refinement + a LSAD-style global semantic assembly
# that fuses high-level (bridge) and low-level (encoder) features.
# ---------------------------------------------------------------------------
class DropBlock2d(nn.Module):
    def __init__(self, drop_prob, block_size):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        return x * mask * (mask.numel() / mask.sum())


class Decoder(nn.Module):
    """Fuses bridge output `x` (high level) with `low_level_feat` (early
    encoder stage) via: multi-scale refinement -> upsample to the low-level
    resolution -> global semantic assembly (channel "codebook" projected
    with spatial coefficients) -> depthwise-separable fusion conv."""

    def __init__(self, in_channels, low_level_channels, out_channels=128,
                 n=4, dropout_rate=0.1, reduction_factor=16):
        super().__init__()
        self.in_channels = in_channels
        self.n = n

        if in_channels % reduction_factor != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by reduction_factor ({reduction_factor})"
            )

        reduced_channels = max(4, in_channels // reduction_factor)

        # Multi-scale feature extraction
        self.multi_scale_compress = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(reduced_channels, reduced_channels // 2, 3, padding=d, dilation=d,
                         groups=_get_valid_num_groups(reduced_channels // 2)),
                nn.BatchNorm2d(reduced_channels // 2),
                nn.SiLU(inplace=True),
            ) for d in (1, 3)
        ])
        self.multi_scale_fusion = nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
        )

        self.upsample = nn.Conv2d(reduced_channels, reduced_channels, 3, padding=1, bias=False)

        # Global semantic assembly (LSAD-inspired)
        self.channel_coeff = nn.Conv2d(reduced_channels, n * n, 1, bias=True)
        self.semantic_points = nn.Conv2d(reduced_channels, n, 1, bias=True)
        self.spatial_coeff = nn.Sequential(
            nn.Conv2d(reduced_channels + low_level_channels, n, 1, bias=False),
            nn.BatchNorm2d(n),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(n, n, 3, padding=1, groups=n, bias=False),  # depthwise
            nn.Conv2d(n, out_channels, 1, bias=False),            # pointwise
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DropBlock2d(dropout_rate, block_size=5),
        )

    def forward(self, x, low_level_feat):
        if x.size(1) != self.in_channels:
            raise RuntimeError(f"Expected {self.in_channels} input channels, got {x.size(1)}")

        # Multi-scale feature extraction
        y = self.multi_scale_compress(x)
        multi_scale_outs = [conv(y) for conv in self.multi_scale]
        x = self.multi_scale_fusion(torch.cat(multi_scale_outs, dim=1))

        x = self.conv1(x)

        x = F.interpolate(x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=True)
        x = self.upsample(x)

        # Global semantic assembly
        C = self.channel_coeff(x)
        C = C.view(C.size(0), self.n, self.n, C.size(2), C.size(3))
        P = self.semantic_points(x)
        L = torch.einsum("bnchw,bchw->bnc", C, P)
        L = F.normalize(L, dim=-1)

        fused = torch.cat([x, low_level_feat], dim=1)
        S = self.spatial_coeff(fused)
        x = torch.einsum("bnc,bchw->bnhw", L, S)

        return self.conv2(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
@register_model("ultralightcps")
class UltraLightCPS(nn.Module):
    """Ultra-lightweight colon-polyp segmentation model.

    encoder : depthwise-separable MBConv backbone (ULCEncoder)
    bridge  : compressed multi-scale feature module + CBAM (MSFE_CBAM)
    decoder : global semantic assembly decoder fusing low/high level features
    """

    ENCODER_CONFIG = {
        "ulc_encoder_func": {
            "low_level_layer": "features.2",
            "high_level_layer": "features.7",
            "low_level_channels": 24,
            "high_level_channels": 320,
        },
    }

    def __init__(self, encoder_name="ulc_encoder_func", bridge="MSFE_CBAM", classes=1):
        super().__init__()

        if encoder_name not in self.ENCODER_CONFIG:
            raise ValueError(
                f"Unsupported encoder: {encoder_name}. Supported: {list(self.ENCODER_CONFIG.keys())}"
            )
        config = self.ENCODER_CONFIG[encoder_name]

        self.encoder = ULCEncoder()
        self.encoder.classifier = nn.Identity()

        dilations = (3, 5, 7)
        self.aspp = MSFE_CBAM(config["high_level_channels"], 256, dilations)
        self.decoder = Decoder(256, config["low_level_channels"], out_channels=128)

        self.final_conv = nn.Conv2d(128, classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        size = x.shape[2:]

        x = self.encoder.features[0](x)
        x = self.encoder.features[1](x)
        x_low = self.encoder.features[2](x)
        x = self.encoder.features[3](x_low)
        x = self.encoder.features[4](x)
        x = self.encoder.features[5](x)
        x = self.encoder.features[6](x)
        x_high = self.encoder.features[7](x)

        x = self.aspp(x_high)
        x = self.decoder(x, x_low)

        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x = self.final_conv(x)
        return self.activation(x)


__all__ = ["UltraLightCPS"]