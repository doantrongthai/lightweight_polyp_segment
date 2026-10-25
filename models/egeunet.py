import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

from . import register_model


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - mean) / torch.sqrt(var + self.eps) + self.bias[:, None, None]


class GroupAggregationBridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=(1, 2, 5, 7)):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.groups = nn.ModuleList([
            nn.Sequential(
                LayerNorm(group_size + 1, data_format="channels_first"),
                nn.Conv2d(
                    group_size + 1,
                    group_size + 1,
                    kernel_size=3,
                    padding=(k_size + (k_size - 1) * (d - 1)) // 2,
                    dilation=d,
                    groups=group_size + 1,
                ),
            )
            for d in d_list
        ])
        self.tail_conv = nn.Sequential(
            LayerNorm(dim_xl * 2 + 4, data_format="channels_first"),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1),
        )

    def forward(self, xh, xl, mask):
        xh = F.interpolate(self.pre_project(xh), size=xl.shape[2:], mode="bilinear", align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        outs = [g(torch.cat((xh[i], xl[i], mask), dim=1)) for i, g in enumerate(self.groups)]
        return self.tail_conv(torch.cat(outs, dim=1))


class GroupedMultiAxisAttention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        c_dim = dim_in // 4
        self.params_xy = nn.Parameter(torch.ones(1, c_dim, x, y), requires_grad=True)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim, c_dim, 3, padding=1, groups=c_dim), nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.params_zx = nn.Parameter(torch.ones(1, 1, c_dim, x), requires_grad=True)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim, c_dim, 3, padding=1, groups=c_dim), nn.GELU(), nn.Conv1d(c_dim, c_dim, 1))
        self.params_zy = nn.Parameter(torch.ones(1, 1, c_dim, y), requires_grad=True)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim, c_dim, 3, padding=1, groups=c_dim), nn.GELU(), nn.Conv1d(c_dim, c_dim, 1))
        self.dw = nn.Sequential(nn.Conv2d(c_dim, c_dim, 1), nn.GELU(), nn.Conv2d(c_dim, c_dim, 3, padding=1, groups=c_dim))
        self.norm1 = LayerNorm(dim_in, data_format="channels_first")
        self.norm2 = LayerNorm(dim_in, data_format="channels_first")
        self.ldw = nn.Sequential(nn.Conv2d(dim_in, dim_in, 3, padding=1, groups=dim_in), nn.GELU(), nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        params_xy = F.interpolate(self.params_xy, size=x1.shape[2:4], mode="bilinear", align_corners=True)
        x1 = x1 * self.conv_xy(params_xy)
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = F.interpolate(self.params_zx, size=x2.shape[2:4], mode="bilinear", align_corners=True).squeeze(0)
        x2 = (x2 * self.conv_zx(params_zx).unsqueeze(0)).permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = F.interpolate(self.params_zy, size=x3.shape[2:4], mode="bilinear", align_corners=True).squeeze(0)
        x3 = (x3 * self.conv_zy(params_zy).unsqueeze(0)).permute(0, 2, 1, 3)
        x4 = self.dw(x4)
        x = self.norm2(torch.cat([x1, x2, x3, x4], dim=1))
        return self.ldw(x)


@register_model("egeunet")
class EGEUNet(nn.Module):
    def __init__(self, classes=1, input_channels=3, c_list=(8, 16, 24, 32, 48, 64)):
        super().__init__()
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, padding=1)
        self.encoder4 = GroupedMultiAxisAttention(c_list[2], c_list[3])
        self.encoder5 = GroupedMultiAxisAttention(c_list[3], c_list[4])
        self.encoder6 = GroupedMultiAxisAttention(c_list[4], c_list[5])
        self.gab1 = GroupAggregationBridge(c_list[1], c_list[0])
        self.gab2 = GroupAggregationBridge(c_list[2], c_list[1])
        self.gab3 = GroupAggregationBridge(c_list[3], c_list[2])
        self.gab4 = GroupAggregationBridge(c_list[4], c_list[3])
        self.gab5 = GroupAggregationBridge(c_list[5], c_list[4])
        self.gt_conv1 = nn.Conv2d(c_list[4], 1, 1)
        self.gt_conv2 = nn.Conv2d(c_list[3], 1, 1)
        self.gt_conv3 = nn.Conv2d(c_list[2], 1, 1)
        self.gt_conv4 = nn.Conv2d(c_list[1], 1, 1)
        self.gt_conv5 = nn.Conv2d(c_list[0], 1, 1)
        self.decoder1 = GroupedMultiAxisAttention(c_list[5], c_list[4])
        self.decoder2 = GroupedMultiAxisAttention(c_list[4], c_list[3])
        self.decoder3 = GroupedMultiAxisAttention(c_list[3], c_list[2])
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, padding=1)
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.final = nn.Conv2d(c_list[0], classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out
        out = F.gelu(self.encoder6(out))
        t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        t5 = self.gab5(t6, t5, self.gt_conv1(out5))
        out5 = out5 + t5
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), size=t4.shape[2:], mode="bilinear", align_corners=True))
        t4 = self.gab4(t5, t4, self.gt_conv2(out4))
        out4 = out4 + t4
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), size=t3.shape[2:], mode="bilinear", align_corners=True))
        t3 = self.gab3(t4, t3, self.gt_conv3(out3))
        out3 = out3 + t3
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), size=t2.shape[2:], mode="bilinear", align_corners=True))
        t2 = self.gab2(t3, t2, self.gt_conv4(out2))
        out2 = out2 + t2
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), size=t1.shape[2:], mode="bilinear", align_corners=True))
        t1 = self.gab1(t2, t1, self.gt_conv5(out1))
        out1 = out1 + t1
        return F.interpolate(self.final(out1), size=size, mode="bilinear", align_corners=True)
