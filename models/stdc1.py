import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class ConvBNReLU(nn.Module):
    """Conv -> BN -> ReLU co ban, dung xuyen suot STDC."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class STDCCatModule(nn.Module):
    """STDC module (kieu 'cat'): moi nhanh giam 1/2 so kenh so voi nhanh
    truoc, sau do concat toan bo lai (Short-Term Dense Concatenate).
    Nhanh dau tien dung conv 1x1, cac nhanh sau dung conv 3x3.
    Neu stride=2 thi nhanh thu 2 duoc downsample (depthwise 3x3 stride 2),
    con nhanh dau (out1) duoc downsample bang AvgPool de lam skip khi concat.
    """

    def __init__(self, in_channels, out_channels, num_branches=4, stride=1):
        super().__init__()
        assert num_branches > 1, "num_branches phai > 1"
        self.stride = stride

        # Neu stride=2, viec downsample do self.down dam nhiem (ap dung truoc
        # branch thu 2), nen ban than cac conv trong branches luon dung stride=1.
        branch_stride = 1 if stride == 2 else stride

        self.branches = nn.ModuleList()
        for idx in range(num_branches):
            if idx == 0:
                self.branches.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1))
            elif idx == 1 and num_branches == 2:
                self.branches.append(ConvBNReLU(out_channels // 2, out_channels // 2, stride=branch_stride))
            elif idx == 1:
                self.branches.append(ConvBNReLU(out_channels // 2, out_channels // 4, stride=branch_stride))
            elif idx < num_branches - 1:
                c_in = out_channels // int(math.pow(2, idx))
                c_out = out_channels // int(math.pow(2, idx + 1))
                self.branches.append(ConvBNReLU(c_in, c_out))
            else:
                c = out_channels // int(math.pow(2, idx))
                self.branches.append(ConvBNReLU(c, c))

        if stride == 2:
            mid = out_channels // 2
            self.down = nn.Sequential(
                nn.Conv2d(mid, mid, 3, stride=2, padding=1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
            )
            self.skip_down = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        outs = []
        out1 = self.branches[0](x)

        out = out1
        for idx, branch in enumerate(self.branches[1:]):
            if idx == 0 and self.stride == 2:
                out = branch(self.down(out))
            else:
                out = branch(out)
            outs.append(out)

        if self.stride == 2:
            out1 = self.skip_down(out1)
        outs.insert(0, out1)

        return torch.cat(outs, dim=1)


class STDC1Backbone(nn.Module):
    """STDC1 (STDCNet813 trong paper goc) - config: base=64, layers=[2,2,2],
    num_branches=4, module kieu 'cat'. Backbone tra ve 5 feature map o
    cac ty le 1/2, 1/4, 1/8, 1/16, 1/32 so voi anh dau vao (dung cho STDC-Seg).
    """

    def __init__(self, base=64, layers=(2, 2, 2), num_branches=4, in_channels=3):
        super().__init__()

        stem = [
            ConvBNReLU(in_channels, base // 2, kernel_size=3, stride=2),
            ConvBNReLU(base // 2, base, kernel_size=3, stride=2),
        ]

        stages = []
        for i, num_blocks in enumerate(layers):
            for j in range(num_blocks):
                if i == 0 and j == 0:
                    stages.append(STDCCatModule(base, base * 4, num_branches, stride=2))
                elif j == 0:
                    c_in = base * int(math.pow(2, i + 1))
                    c_out = base * int(math.pow(2, i + 2))
                    stages.append(STDCCatModule(c_in, c_out, num_branches, stride=2))
                else:
                    c = base * int(math.pow(2, i + 2))
                    stages.append(STDCCatModule(c, c, num_branches, stride=1))

        features = nn.Sequential(*stem, *stages)

        # stem[0] -> 1/2, stem[1] -> 1/4, stage nhom 0 -> 1/8, nhom 1 -> 1/16, nhom 2 -> 1/32
        self.stage_x2 = features[0:1]
        self.stage_x4 = features[1:2]
        self.stage_x8 = features[2:2 + layers[0]]
        self.stage_x16 = features[2 + layers[0]:2 + layers[0] + layers[1]]
        self.stage_x32 = features[2 + layers[0] + layers[1]:]

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
        feat2 = self.stage_x2(x)
        feat4 = self.stage_x4(feat2)
        feat8 = self.stage_x8(feat4)
        feat16 = self.stage_x16(feat8)
        feat32 = self.stage_x32(feat16)
        return feat2, feat4, feat8, feat16, feat32


@register_model("stdc1_cls")
class STDC1(nn.Module):
    """STDC1 cho phan loai anh (dung de pretrain tren ImageNet / doi chieu
    so luong tham so voi paper). KHONG dung truc tiep de segmentation - xem
    STDC1Seg (registry "stdc1") ben duoi, day chi la backbone + cls head."""

    def __init__(self, num_classes=1000, base=64, layers=(2, 2, 2),
                 num_branches=4, in_channels=3, dropout=0.20, use_conv_last=False):
        super().__init__()
        self.use_conv_last = use_conv_last
        self.backbone = STDC1Backbone(base=base, layers=layers, num_branches=num_branches, in_channels=in_channels)

        last_channels = max(1024, base * 16)
        self.conv_last = ConvBNReLU(base * 16, last_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_channels, last_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(last_channels, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, _, _, feat32 = self.backbone(x)
        out = self.conv_last(feat32)
        out = self.gap(out).flatten(1)
        out = self.relu(self.fc(out))
        out = self.dropout(out)
        out = self.classifier(out)
        return out


class AttentionRefinementModule(nn.Module):
    """ARM (BiSeNet/STDC-Seg): tinh 1 vector attention theo kenh tu global
    context (GAP) roi nhan lai vao feature map, dung o cac tang sau
    (feat16, feat32) de loc nhieu context thua."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        attn = self.attn(self.gap(feat))
        return feat * attn


class FeatureFusionModule(nn.Module):
    """FFM (BiSeNet/STDC-Seg): concat detail feature (tang nong, nhieu
    canh/chi tiet) voi context feature (tang sau, nhieu ngu nghia), sau do
    tinh lai attention theo kenh de can chinh dong gop cua 2 nhanh."""

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced = max(out_channels // reduction, 8)
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, detail, context):
        feat = self.conv(torch.cat([detail, context], dim=1))
        attn = self.attn(self.gap(feat))
        return feat + feat * attn


class SegHead(nn.Module):
    """Head 1x1 conv -> logits 1 kenh -> upsample bilinear ve dung kich
    thuoc anh dau vao (dung chung cho nhanh main va cac nhanh aux)."""

    def __init__(self, in_channels, mid_channels, num_classes=1):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3)
        self.classifier = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, x, out_size):
        x = self.conv(x)
        x = self.classifier(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=True)


@register_model("stdc1")
class STDC1Seg(nn.Module):
    """STDC1 cho segmentation (polyp), decoder nhe kieu BiSeNet/STDC-Seg:
    ARM tren feat32 va feat16 de loc context, FFM ket hop voi feat8 (detail)
    de giu bien/chi tiet nho, cuoi cung upsample bilinear ve dung 352x352.
    Co 2 nhanh aux (tu feat32, feat16) de deep supervision khi training,
    dung dung interface trainer dang goi: main_out, aux_outputs = model(x).
    """

    def __init__(self, num_classes=1, base=64, layers=(2, 2, 2), num_branches=4,
                 in_channels=3, decoder_channels=128, img_size=352):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.backbone = STDC1Backbone(base=base, layers=layers, num_branches=num_branches, in_channels=in_channels)

        c8, c16, c32 = base * 4, base * 8, base * 16  # 256, 512, 1024 voi base=64

        self.arm32 = AttentionRefinementModule(c32, decoder_channels)
        self.arm16 = AttentionRefinementModule(c16, decoder_channels)
        self.ffm = FeatureFusionModule(c8 + decoder_channels, decoder_channels)

        self.seg_head = SegHead(decoder_channels, decoder_channels, num_classes)
        self.aux_head32 = SegHead(decoder_channels, decoder_channels // 2, num_classes)
        self.aux_head16 = SegHead(decoder_channels, decoder_channels // 2, num_classes)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in [self.arm32, self.arm16, self.ffm, self.seg_head, self.aux_head32, self.aux_head16]:
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
                    if mm.bias is not None:
                        nn.init.constant_(mm.bias, 0)
                elif isinstance(mm, nn.BatchNorm2d):
                    nn.init.constant_(mm.weight, 1)
                    nn.init.constant_(mm.bias, 0)

    def forward(self, x):
        out_size = x.shape[2:]  # luon bam theo kich thuoc anh dau vao thuc te (mac dinh 352x352)
        _, _, feat8, feat16, feat32 = self.backbone(x)

        ctx32 = self.arm32(feat32)
        ctx32_up = F.interpolate(ctx32, size=feat16.shape[2:], mode="bilinear", align_corners=True)

        ctx16 = self.arm16(feat16) + ctx32_up
        ctx16_up = F.interpolate(ctx16, size=feat8.shape[2:], mode="bilinear", align_corners=True)

        fused = self.ffm(feat8, ctx16_up)
        main_out = self.seg_head(fused, out_size)

        if self.training:
            aux_out32 = self.aux_head32(ctx32, out_size)
            aux_out16 = self.aux_head16(ctx16, out_size)
            return main_out, aux_out32, aux_out16

        return main_out


__all__ = ["STDC1", "STDC1Backbone", "STDC1Seg"]
