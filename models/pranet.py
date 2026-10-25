import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def _resize(x, size, align_corners=True):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=26, scale=4, stype="normal"):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = 1 if scale == 1 else scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        out = None
        sp = None
        for i in range(self.nums):
            sp = spx[i] if i == 0 or self.stype == "stage" else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if out is None else torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Res2Net(nn.Module):
    def __init__(self, block, layers, base_width=26, scale=4, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.base_width = base_width
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                base_width=self.base_width,
                scale=self.scale,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, scale=self.scale))
        return nn.Sequential(*layers)


RES2NET_PRETRAINED_URLS = {
    (26, 4): "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
}


def _load_res2net_pretrained(model, base_width, scale):
    url = RES2NET_PRETRAINED_URLS.get((base_width, scale))
    if url is None:
        warnings.warn(
            f"No public ImageNet-pretrained checkpoint registered for Res2Net "
            f"(base_width={base_width}, scale={scale}); training backbone from scratch."
        )
        return
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
    except Exception as e:
        warnings.warn(f"Could not download pretrained Res2Net weights ({e}); training backbone from scratch.")
        return
    own_state = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    skipped = len(state_dict) - len(matched)
    if skipped:
        warnings.warn(f"Loaded pretrained Res2Net backbone, skipped {skipped} non-matching keys.")


def res2net50_v1b_26w_4s(pretrained=True):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], base_width=26, scale=4)
    if pretrained:
        _load_res2net_pretrained(model, base_width=26, scale=4)
    return model


class RFBModified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x_cat = self.conv_cat(torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1))
        return self.relu(x_cat + self.conv_res(x))


class Aggregation(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
            * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


@register_model("pranet")
class PraNet(nn.Module):
    def __init__(self, classes=1, channel=32, pretrained=True):
        super().__init__()
        if classes != 1:
            raise ValueError("PraNet in this pipeline is configured for binary segmentation.")
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained)
        self.rfb2_1 = RFBModified(512, channel)
        self.rfb3_1 = RFBModified(1024, channel)
        self.rfb4_1 = RFBModified(2048, channel)
        self.agg1 = Aggregation(channel)
        self.ra4_conv1 = BasicConv2d(2048, 256, 1)
        self.ra4_conv2 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, 1)
        self.ra3_conv1 = BasicConv2d(1024, 64, 1)
        self.ra3_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, 3, padding=1)
        self.ra2_conv1 = BasicConv2d(512, 64, 1)
        self.ra2_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, 3, padding=1)

    def forward(self, x):
        size = x.shape[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)

        crop_4 = _resize(ra5_feat, x4.shape[2:], align_corners=False)
        ra = 1 - torch.sigmoid(crop_4)
        ra = ra.expand(-1, 2048, -1, -1).mul(x4)
        ra = self.ra4_conv5(F.relu(self.ra4_conv4(F.relu(self.ra4_conv3(F.relu(self.ra4_conv2(self.ra4_conv1(ra))))))))
        x_4 = ra + crop_4

        crop_3 = _resize(x_4, x3.shape[2:], align_corners=False)
        ra = 1 - torch.sigmoid(crop_3)
        ra = ra.expand(-1, 1024, -1, -1).mul(x3)
        ra = self.ra3_conv4(F.relu(self.ra3_conv3(F.relu(self.ra3_conv2(self.ra3_conv1(ra))))))
        x_3 = ra + crop_3

        crop_2 = _resize(x_3, x2.shape[2:], align_corners=False)
        ra = 1 - torch.sigmoid(crop_2)
        ra = ra.expand(-1, 512, -1, -1).mul(x2)
        ra = self.ra2_conv4(F.relu(self.ra2_conv3(F.relu(self.ra2_conv2(self.ra2_conv1(ra))))))
        x_2 = ra + crop_2

        out = _resize(x_2, size, align_corners=False)
        map4 = _resize(x_4, size, align_corners=False)
        map3 = _resize(x_3, size, align_corners=False)
        map5 = _resize(ra5_feat, size, align_corners=False)
        return out, map3, map4, map5


__all__ = ["PraNet"]
