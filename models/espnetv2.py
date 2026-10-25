import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from . import register_model


def _resize(x, size, align_corners=True):
    if x.shape[2:] == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def _resize_like(x, ref, align_corners=True):
    return _resize(x, ref.shape[2:], align_corners=align_corners)


class V2PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = nn.ModuleList([V2C(features, features, 3, 1, groups=features) for _ in sizes])
        self.project = V2CBR(features * (len(sizes) + 1), out_features, 1, 1)

    def forward(self, feats):
        h, w = feats.shape[2:]
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            out.append(_resize(stage(feats), (h, w)))
        return self.project(torch.cat(out, dim=1))


class V2CBR(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1, groups=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class V2BR(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        return self.act(self.bn(x))


class V2CB(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1, groups=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class V2C(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1, groups=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False, groups=groups)

    def forward(self, x):
        return self.conv(x)


class V2CDilated(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1, d=1, groups=1):
        super().__init__()
        padding = ((k_size - 1) // 2) * d
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, dilation=d, bias=False, groups=groups)

    def forward(self, x):
        return self.conv(x)


class EESP(nn.Module):
    def __init__(self, n_in, n_out, stride=1, k=4, r_lim=7, down_method="esp"):
        super().__init__()
        self.stride = stride
        n = int(n_out / k)
        n1 = n_out - (k - 1) * n
        assert down_method in ["avg", "esp"]
        assert n == n1
        self.proj_1x1 = V2CBR(n_in, n, 1, stride=1, groups=k)
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        k_sizes = []
        for i in range(k):
            ksize = 3 + 2 * i
            k_sizes.append(ksize if ksize <= r_lim else 3)
        k_sizes.sort()
        self.spp_dw = nn.ModuleList([
            V2CDilated(n, n, 3, stride=stride, groups=n, d=map_receptive_ksize[ksize])
            for ksize in k_sizes
        ])
        self.conv_1x1_exp = V2CB(n_out, n_out, 1, 1, groups=k)
        self.br_after_cat = V2BR(n_out)
        self.module_act = nn.PReLU(n_out)
        self.downAvg = down_method == "avg"

    def forward(self, x):
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for idx in range(1, len(self.spp_dw)):
            output.append(self.spp_dw[idx](output1) + output[idx - 1])
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == x.size():
            expanded = expanded + x
        return self.module_act(expanded)


class EESPDownSampler(nn.Module):
    def __init__(self, n_in, n_out, k=4, r_lim=9, reinf=True):
        super().__init__()
        nout_new = n_out - n_in
        self.eesp = EESP(n_in, nout_new, stride=2, k=k, r_lim=r_lim, down_method="avg")
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.reinf = reinf
        if reinf:
            self.inp_reinf = nn.Sequential(V2CBR(3, 3, 3, 1), V2CB(3, n_out, 1, 1))
        self.act = nn.PReLU(n_out)

    def forward(self, x, input2=None):
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        avg_out = _resize_like(avg_out, eesp_out)
        out = torch.cat([avg_out, eesp_out], 1)
        if self.reinf and input2 is not None:
            reinf = input2
            while reinf.shape[2:] != out.shape[2:]:
                next_reinf = F.avg_pool2d(reinf, kernel_size=3, padding=1, stride=2)
                if next_reinf.shape[2] < out.shape[2] or next_reinf.shape[3] < out.shape[3]:
                    reinf = _resize(next_reinf, out.shape[2:])
                    break
                reinf = next_reinf
            out = out + self.inp_reinf(_resize(reinf, out.shape[2:]))
        return self.act(out)


class EESPNet(nn.Module):
    def __init__(self, classes=1000, s=1):
        super().__init__()
        reps = [0, 3, 7, 3]
        channels = 3
        r_lim = [13, 11, 9, 7, 5]
        k = [4] * len(r_lim)
        base = 32
        config = [base] * 5
        base_s = 0
        for i in range(5):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / k[0]) * k[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s in [1.5, 2]:
            config.append(1280)
        else:
            raise ValueError("Unsupported EESPNet scale")
        self.input_reinforcement = True
        self.level1 = V2CBR(channels, config[0], 3, 2)
        self.level2_0 = EESPDownSampler(config[0], config[1], k=k[0], r_lim=r_lim[0], reinf=True)
        self.level3_0 = EESPDownSampler(config[1], config[2], k=k[1], r_lim=r_lim[1], reinf=True)
        self.level3 = nn.ModuleList([EESP(config[2], config[2], stride=1, k=k[2], r_lim=r_lim[2]) for _ in range(reps[1])])
        self.level4_0 = EESPDownSampler(config[2], config[3], k=k[2], r_lim=r_lim[2], reinf=True)
        self.level4 = nn.ModuleList([EESP(config[3], config[3], stride=1, k=k[3], r_lim=r_lim[3]) for _ in range(reps[2])])
        self.level5_0 = EESPDownSampler(config[3], config[4], k=k[3], r_lim=r_lim[3], reinf=False)
        self.level5 = nn.ModuleList([EESP(config[4], config[4], stride=1, k=k[4], r_lim=r_lim[4]) for _ in range(reps[3])])
        self.level5.append(V2CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(V2CBR(config[4], config[5], 1, 1, groups=k[4]))
        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, p=0.2, seg=True):
        out_l1 = self.level1(x)
        reinf = x if self.input_reinforcement else None
        out_l2 = self.level2_0(out_l1, reinf)
        out_l3_0 = self.level3_0(out_l2, reinf)
        out_l3 = out_l3_0
        for layer in self.level3:
            out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, reinf)
        out_l4 = out_l4_0
        for layer in self.level4:
            out_l4 = layer(out_l4)
        if seg:
            return out_l1, out_l2, out_l3, out_l4
        out_l5 = self.level5_0(out_l4)
        for layer in self.level5:
            out_l5 = layer(out_l5)
        out_g = F.dropout(F.adaptive_avg_pool2d(out_l5, 1), p=p, training=self.training)
        return self.classifier(out_g.view(out_g.size(0), -1))


@register_model("espnetv2")
class EESPNet_Seg(nn.Module):
    def __init__(self, classes=1, s=2, pretrained=None):
        super().__init__()
        self.net = EESPNet(classes=1000, s=s)
        if pretrained:
            self.net.load_state_dict(torch.load(pretrained, map_location="cpu"))
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        p = 0.1 if s <= 0.5 else 0.2
        l4_channels = self.net.level4[-1].module_act.num_parameters
        l3_channels = self.net.level3[-1].module_act.num_parameters
        self.proj_L4_C = V2CBR(l4_channels, l3_channels, 1, 1)
        psp_size = 2 * l3_channels
        self.pspMod = nn.Sequential(EESP(psp_size, psp_size // 2, stride=1, k=4, r_lim=7), V2PSPModule(psp_size // 2, psp_size // 2))
        self.project_l3 = nn.Sequential(nn.Dropout2d(p=p), V2C(psp_size // 2, classes, 1, 1))
        self.act_l3 = V2BR(classes)
        self.project_l2 = V2CBR(self.net.level2_0.act.num_parameters + classes, classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(p=p), V2C(self.net.level1.act.num_parameters + classes, classes, 1, 1))

    def forward(self, x):
        size = x.shape[2:]
        out_l1, out_l2, out_l3, out_l4 = self.net(x, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = _resize_like(out_l4_proj, out_l3)
        merged_l3 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_l3_before = self.project_l3(merged_l3)
        proj_l3 = self.act_l3(proj_l3_before)
        out_up_l3 = _resize_like(proj_l3, out_l2)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = _resize_like(merge_l2, out_l1)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        return _resize(merge_l1, size)


__all__ = ["EESPNet_Seg"]
