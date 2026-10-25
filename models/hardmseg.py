import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .pranet import Aggregation, BasicConv2d, RFBModified


def _resize(x, size, align_corners=False):
    # align_corners=False by default here to match HarDMSEG.py's original
    # `F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')` call, which
    # never set align_corners (PyTorch default => False).
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(True),
        )


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                link.append(layer - dv)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keep_base=False):
        super().__init__()
        self.keepBase = keep_base
        self.links = []
        self.out_channels = 0
        layers = []
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            layers.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        layers = [x]
        for layer_idx, layer in enumerate(self.layers):
            tin = [layers[i] for i in self.links[layer_idx]]
            out = layer(torch.cat(tin, 1) if len(tin) > 1 else tin[0])
            layers.append(out)
        outputs = []
        for i, item in enumerate(layers):
            if (i == 0 and self.keepBase) or (i == len(layers) - 1) or (i % 2 == 1):
                outputs.append(item)
        return torch.cat(outputs, 1)


# ImageNet-pretrained checkpoints from PingoLH/Pytorch-HarDNet, the same source the
# original HarDNet-MSEG README points to ("First download pretrain_weight: hardnet68.pth
# ... in https://github.com/PingoLH/Pytorch-HarDNet"). Only the non-depthwise 68/85
# variants have public releases; hardnet39 only has a depthwise ("ds") release upstream,
# which isn't compatible with this non-depthwise backbone, so it's left unpretrained.
HARDNET_PRETRAINED_URLS = {
    68: "https://ping-chao.com/hardnet/hardnet68-5d684880.pth",
    85: "https://ping-chao.com/hardnet/hardnet85-a28faa00.pth",
}


class HarDNetBackbone(nn.Module):
    def __init__(self, arch=68, pretrained=True):
        super().__init__()
        first_ch = [32, 64]
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        down_samp = [1, 0, 1, 1, 0]
        if arch == 85:
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            down_samp = [1, 0, 1, 0, 1, 0]
        elif arch == 39:
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            down_samp = [1, 1, 1, 0]

        self.arch = arch
        self.base = nn.ModuleList()
        self.base.append(ConvLayer(3, first_ch[0], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], 1.7 if arch != 39 else 1.6, n_layers[i])
            ch = blk.get_out_ch()
            self.base.append(blk)
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if down_samp[i] == 1:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        url = HARDNET_PRETRAINED_URLS.get(self.arch)
        if url is None:
            warnings.warn(
                f"No public ImageNet-pretrained checkpoint for HarDNet-{self.arch} "
                f"(non-depthwise); training backbone from scratch."
            )
            return
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
        except Exception as e:  # noqa: BLE001 - don't let a download hiccup crash model construction
            warnings.warn(f"Could not download pretrained HarDNet-{self.arch} weights ({e}); training backbone from scratch.")
            return
        own_state = self.state_dict()
        # The upstream checkpoint includes an ImageNet classifier head
        # (`base.16`: AdaptiveAvgPool + Linear) that this backbone-only module
        # doesn't define, so we only load matching keys and ignore the rest.
        matched = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        self.load_state_dict(matched, strict=False)
        skipped = len(state_dict) - len(matched)
        if skipped:
            warnings.warn(f"Loaded pretrained HarDNet-{self.arch} backbone, skipped {skipped} non-matching keys (classifier head, etc.).")

    def forward(self, x):
        out_branch = []
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in (4, 9, 12, 15):
                out_branch.append(x)
        return out_branch


@register_model("hardnet")
@register_model("hardmseg")
class HarDMSEG(nn.Module):
    def __init__(self, classes=1, channel=32, pretrained=True):
        super().__init__()
        if classes != 1:
            raise ValueError("HarDMSEG in this pipeline is configured for binary segmentation.")
        self.rfb2_1 = RFBModified(320, channel)
        self.rfb3_1 = RFBModified(640, channel)
        self.rfb4_1 = RFBModified(1024, channel)
        self.agg1 = Aggregation(channel)
        self.hardnet = HarDNetBackbone(arch=68, pretrained=pretrained)

    def forward(self, x):
        size = x.shape[2:]
        hardnet_out = self.hardnet(x)
        x2 = hardnet_out[1]
        x3 = hardnet_out[2]
        x4 = hardnet_out[3]
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        out = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        # align_corners=False matches the original HarDMSEG.py's
        # `F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')`, which
        # never set align_corners.
        return _resize(out, size, align_corners=False)
