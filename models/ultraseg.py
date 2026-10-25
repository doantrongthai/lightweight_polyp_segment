import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def _resize(x, size, align_corners=True):
    if x.shape[2:] == tuple(size):
        return x
    return F.interpolate(
        x, size=size, mode="bilinear", align_corners=align_corners
    )


def _resize_like(x, ref, align_corners=True):
    return _resize(x, ref.shape[2:], align_corners=align_corners)


class EnhancedDilatedBlock(nn.Module):
    """
    Enhanced Dilated Block (EDB) used at encoder stage 3.

    This follows the released UltraSeg implementation:
    three grouped 3x3 dilated convolutions with dilation 1/2/3,
    concatenation, 1x1 fusion and a lightweight residual branch.
    """

    def __init__(self, channels, learnable_residual=False):
        super().__init__()
        if channels % 3 != 0:
            raise ValueError("EDB requires channels divisible by 3.")

        group_channels = channels // 3
        self.dilated_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    group_channels,
                    3,
                    padding=1,
                    dilation=1,
                    groups=group_channels,
                ),
                nn.Conv2d(
                    channels,
                    group_channels,
                    3,
                    padding=2,
                    dilation=2,
                    groups=group_channels,
                ),
                nn.Conv2d(
                    channels,
                    group_channels,
                    3,
                    padding=3,
                    dilation=3,
                    groups=group_channels,
                ),
            ]
        )
        self.fusion = nn.Conv2d(channels, channels, 1)

        if learnable_residual:
            self.res_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer(
                "res_weight", torch.tensor(0.1), persistent=False
            )

    def forward(self, x):
        features = [conv(x) for conv in self.dilated_convs]
        fused = torch.cat(features, dim=1)
        return x + self.fusion(fused) * self.res_weight


class LayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm supporting NCHW tensors."""

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        data_format="channels_last",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(
                f"Unsupported data_format: {data_format}"
            )

        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )


class Down(nn.Module):
    """BN followed by a learned 2x2 stride-2 downsampling conv."""

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.conv(self.bn(x))


class ImagePredictionGenerator(nn.Module):
    """
    Region prediction head used at the deepest decoder stage.

    It predicts a region logit map and reweights the feature map
    using sigmoid(region_prediction).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        region_logit = self.conv(x)
        x = x + x * torch.sigmoid(region_logit)
        return x, region_logit


class PredictionGenerator(nn.Module):
    """
    Joint region/boundary prediction head used by shallower
    decoder stages.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.boundary = nn.Conv2d(in_channels, 1, 1)
        self.region = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        boundary_prob = torch.sigmoid(self.boundary(x))
        region_logit = self.region(x)

        x = (
            x
            + x * boundary_prob
            + x * torch.sigmoid(region_logit)
        )
        return x, region_logit, boundary_prob


class GroupShuffleBlock(nn.Module):
    """
    Group-Shuffle Fusion / Attention block retained from LB-UNet.

    The channels are split into four groups. Each group is
    modulated by a learned spatial prior, shuffled, normalized,
    then projected by a depthwise + pointwise path.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in % 4 != 0:
            raise ValueError(
                "GroupShuffleBlock requires dim_in divisible by 4."
            )

        c_dim = dim_in // 4

        self.share_space1 = nn.Parameter(
            torch.ones(1, c_dim, 8, 8)
        )
        self.share_space2 = nn.Parameter(
            torch.ones(1, c_dim, 8, 8)
        )
        self.share_space3 = nn.Parameter(
            torch.ones(1, c_dim, 8, 8)
        )
        self.share_space4 = nn.Parameter(
            torch.ones(1, c_dim, 8, 8)
        )

        self.conv1 = self._group_branch(c_dim)
        self.conv2 = self._group_branch(c_dim)
        self.conv3 = self._group_branch(c_dim)
        self.conv4 = self._group_branch(c_dim)

        self.norm1 = LayerNorm(
            dim_in, eps=1e-6, data_format="channels_first"
        )
        self.norm2 = LayerNorm(
            dim_in, eps=1e-6, data_format="channels_first"
        )

        self.ldw = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=3,
                padding=1,
                groups=dim_in,
            ),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    @staticmethod
    def _group_branch(channels):
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    @staticmethod
    def _apply_spatial_prior(x, prior, branch):
        prior = F.interpolate(
            prior,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        return x * branch(prior)

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x1 = self._apply_spatial_prior(
            x1, self.share_space1, self.conv1
        )
        x2 = self._apply_spatial_prior(
            x2, self.share_space2, self.conv2
        )
        x3 = self._apply_spatial_prior(
            x3, self.share_space3, self.conv3
        )
        x4 = self._apply_spatial_prior(
            x4, self.share_space4, self.conv4
        )

        # Official group shuffle order.
        x = torch.cat([x2, x4, x1, x3], dim=1)
        x = self.norm2(x)
        return self.ldw(x)


class PredictGatedFusion(nn.Module):
    """
    Predict-Gated Fusion (PGF).

    Paper equation:
        x_out = x1' + x2
              + alpha * sigmoid(P_region) * x2
              + beta  * P_boundary * x2

    alpha/beta are learnable scalars.
    """

    def __init__(self, channels, use_boundary=True):
        super().__init__()
        self.use_boundary = use_boundary

        self.region_weight = nn.Parameter(torch.tensor(0.3))

        if use_boundary:
            self.boundary_weight = nn.Parameter(
                torch.tensor(0.1)
            )

        if channels > 16:
            hidden = max(channels // 8, 4)
            self.feature_adapter = nn.Sequential(
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
            )
        else:
            self.feature_adapter = nn.Identity()

    def forward(
        self,
        decoder_feature,
        encoder_feature,
        region_logit,
        boundary_prob=None,
    ):
        decoder_feature = self.feature_adapter(decoder_feature)

        if region_logit.shape[1] == 1:
            region_logit = region_logit.expand(
                -1,
                encoder_feature.shape[1],
                -1,
                -1,
            )

        base = decoder_feature + encoder_feature
        region = (
            torch.sigmoid(region_logit)
            * encoder_feature
            * self.region_weight
        )

        boundary = 0.0
        if self.use_boundary and boundary_prob is not None:
            if boundary_prob.shape[1] == 1:
                boundary_prob = boundary_prob.expand(
                    -1,
                    encoder_feature.shape[1],
                    -1,
                    -1,
                )
            boundary = (
                boundary_prob
                * encoder_feature
                * self.boundary_weight
            )

        return base + region + boundary


class AttentionGuidedFusion(nn.Module):
    """
    Attention Guided Fusion (AGF), used only by UltraSeg-130K.

    Stage-3 and stage-4 features are projected to the same
    channel count. A tiny spatial-attention network predicts
    one softmax weight map per input feature.
    """

    def __init__(self, channels_list):
        super().__init__()
        self.output_ch = channels_list[-1]

        transforms = []
        for ch in channels_list:
            if ch != self.output_ch:
                transforms.append(
                    nn.Conv2d(ch, self.output_ch, 1)
                )
            else:
                transforms.append(nn.Identity())

        self.transforms = nn.ModuleList(transforms)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                len(channels_list) * self.output_ch,
                self.output_ch // 4,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(self.output_ch // 4),
            nn.GELU(),
            nn.Conv2d(
                self.output_ch // 4,
                len(channels_list),
                1,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, features):
        target_size = features[0].shape[2:]
        transformed = []

        for feat, transform in zip(
            features, self.transforms
        ):
            if feat.shape[2:] != target_size:
                feat = _resize(feat, target_size)
            transformed.append(transform(feat))

        concat = torch.cat(transformed, dim=1)
        weights = self.spatial_attention(concat)

        return sum(
            weight * feat
            for weight, feat in zip(
                weights.split(1, dim=1),
                transformed,
            )
        )


class SimpleSpatialAttention(nn.Module):
    """
    Simple Spatial Attention (SSA), used only by UltraSeg-130K.
    """

    def forward(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        attention = torch.sigmoid(attention)
        return x * attention


class _UltraSegBase(nn.Module):
    """
    Shared implementation of UltraSeg-108K and UltraSeg-130K.

    aux=False:
        return final segmentation logits only, matching the
        simple interface used by the other models in this project.

    aux=True:
        return:
            final_logits,
            region_predictions,
            boundary_predictions

        This exposes the deep-supervision outputs required to
        reproduce the loss described in the paper.
    """

    def __init__(
        self,
        classes=1,
        in_channels=3,
        aux=False,
        use_cllf=False,
    ):
        super().__init__()

        self.aux = aux
        self.use_cllf = use_cllf
        self.channels = [8, 16, 48, 64, 96]
        c1, c2, c3, c4, c5 = self.channels

        self.encoder1 = nn.Conv2d(
            in_channels, c1, 3, padding=1
        )
        self.encoder2 = nn.Conv2d(c1, c2, 3, padding=1)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            EnhancedDilatedBlock(
                c3, learnable_residual=use_cllf
            ),
            EnhancedDilatedBlock(
                c3, learnable_residual=use_cllf
            ),
        )

        self.encoder4 = GroupShuffleBlock(c3, c4)

        if use_cllf:
            self.multiscale_fusion = AttentionGuidedFusion(
                [c3, c4]
            )
            self.simple_attention = SimpleSpatialAttention()
            self.attention_alpha = nn.Parameter(
                torch.tensor(0.3)
            )

        self.bottleneck = nn.Conv2d(
            c4, c5, 3, padding=1
        )

        self.Down1 = Down(c1)
        self.Down2 = Down(c2)
        self.Down3 = Down(c3)

        self.merge1 = PredictGatedFusion(
            c1, use_boundary=True
        )
        self.merge2 = PredictGatedFusion(
            c2, use_boundary=True
        )
        self.merge3 = PredictGatedFusion(
            c3, use_boundary=True
        )
        self.merge4 = PredictGatedFusion(
            c4, use_boundary=False
        )

        self.decoder1 = GroupShuffleBlock(c5, c4)
        self.decoder2 = GroupShuffleBlock(c4, c3)
        self.decoder3 = GroupShuffleBlock(c3, c2)
        self.decoder4 = nn.Conv2d(
            c2, c1, 3, padding=1
        )

        self.pred1 = ImagePredictionGenerator(c4)
        self.gate1 = PredictionGenerator(c3)
        self.gate2 = PredictionGenerator(c2)
        self.gate3 = PredictionGenerator(c1)

        self.ebn1 = nn.GroupNorm(4, c1)
        self.ebn2 = nn.GroupNorm(4, c2)
        self.ebn3 = nn.GroupNorm(4, c3)
        self.ebn4 = nn.GroupNorm(4, c4)

        self.dbn1 = nn.GroupNorm(4, c4)
        self.dbn2 = nn.GroupNorm(4, c3)
        self.dbn3 = nn.GroupNorm(4, c2)
        self.dbn4 = nn.GroupNorm(4, c1)

        self.final = nn.Conv2d(c1, classes, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight, std=0.02
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            n = (
                module.kernel_size[0]
                * module.out_channels
            )
            module.weight.data.normal_(
                0, math.sqrt(2.0 / n)
            )

        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(
                0, math.sqrt(2.0 / fan_out)
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, x):
        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))
        t1 = out

        out = self.encoder2(out)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out

        out = self.encoder3(out)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out

        out = self.encoder4(out)
        out = F.gelu(
            F.max_pool2d(self.ebn4(out), 2)
        )
        t4 = out

        if self.use_cllf:
            t3_down = _resize_like(t3, t4)
            fused = self.multiscale_fusion(
                [t3_down, t4]
            )
            bottleneck = self.bottleneck(fused)

            attended = self.simple_attention(
                bottleneck
            )
            bottleneck = (
                (1.0 - self.attention_alpha)
                * bottleneck
                + self.attention_alpha * attended
            )
        else:
            bottleneck = self.bottleneck(t4)

        return bottleneck, (t1, t2, t3, t4)

    def forward(self, x):
        input_size = x.shape[2:]

        out, (t1, t2, t3, t4) = self._encode(x)

        # Decoder stage 4: H/16, 96 -> 64
        out = self.decoder1(out)
        out = F.gelu(self.dbn1(out))

        out, region4 = self.pred1(out)
        out = self.merge4(out, t4, region4)
        region4_full = _resize(
            region4, input_size
        )

        # Decoder stage 3: H/16 -> H/8, 64 -> 48
        out = self.decoder2(out)
        out = F.gelu(
            F.interpolate(
                self.dbn2(out),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        )
        out, region3, boundary3 = self.gate1(out)
        out = self.merge3(
            out, t3, region3, boundary3
        )
        region3_full = _resize(
            region3, input_size
        )
        boundary3_full = _resize(
            boundary3, input_size
        )

        # Decoder stage 2: H/8 -> H/4, 48 -> 16
        out = self.decoder3(out)
        out = F.gelu(
            F.interpolate(
                self.dbn3(out),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        )
        out, region2, boundary2 = self.gate2(out)
        out = self.merge2(
            out, t2, region2, boundary2
        )
        region2_full = _resize(
            region2, input_size
        )
        boundary2_full = _resize(
            boundary2, input_size
        )

        # Decoder stage 1: H/4 -> H/2, 16 -> 8
        out = self.decoder4(out)
        out = F.gelu(
            F.interpolate(
                self.dbn4(out),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        )
        out, region1, boundary1 = self.gate3(out)
        out = self.merge1(
            out, t1, region1, boundary1
        )
        region1_full = _resize(
            region1, input_size
        )
        boundary1_full = _resize(
            boundary1, input_size
        )

        final_logits = self.final(out)
        final_logits = _resize(
            final_logits, input_size
        )

        if not self.aux:
            return final_logits

        # The official code sigmoid-activates region
        # deep-supervision predictions before returning them.
        region_predictions = (
            torch.sigmoid(region4_full),
            torch.sigmoid(region3_full),
            torch.sigmoid(region2_full),
            torch.sigmoid(region1_full),
        )

        boundary_predictions = (
            boundary3_full,
            boundary2_full,
            boundary1_full,
        )

        return (
            final_logits,
            region_predictions,
            boundary_predictions,
        )


@register_model("ultraseg")
@register_model("ultraseg_108k")
class UltraSeg108K(_UltraSegBase):
    """
    UltraSeg-108K:
    channel schedule [8, 16, 48, 64, 96] + EDB + PGF,
    without Cross-Layer Lightweight Fusion.

    Also registered as "ultraseg" (default/base alias).
    """

    def __init__(
        self,
        classes=1,
        in_channels=3,
        aux=False,
    ):
        super().__init__(
            classes=classes,
            in_channels=in_channels,
            aux=aux,
            use_cllf=False,
        )


@register_model("ultraseg_130k")
class UltraSeg130K(_UltraSegBase):
    """
    UltraSeg-130K:
    UltraSeg-108K + CLLF = AGF + SSA.
    """

    def __init__(
        self,
        classes=1,
        in_channels=3,
        aux=False,
    ):
        super().__init__(
            classes=classes,
            in_channels=in_channels,
            aux=aux,
            use_cllf=True,
        )


__all__ = [
    "UltraSeg108K",
    "UltraSeg130K",
    "EnhancedDilatedBlock",
    "GroupShuffleBlock",
    "PredictGatedFusion",
    "AttentionGuidedFusion",
    "SimpleSpatialAttention",
]
