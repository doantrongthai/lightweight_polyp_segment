import torch
import torch.nn as nn
from . import register_model


def _max_unpool(unpool_layer, x, indices, output_size):
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    out = unpool_layer(x, indices, output_size=output_size)
    torch.use_deterministic_algorithms(prev)
    return out


class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.ext_branch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat([main, ext], dim=1)
        out = self.batch_norm(out)
        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.01):
        super().__init__()

        internal_channels = out_channels // 4

        self.main_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

        self.out_channels = out_channels

    def forward(self, x):
        main, max_indices = self.main_maxpool(x)

        n, ch, h, w = main.size()
        pad_ch = self.out_channels - ch
        if pad_ch > 0:
            padding = torch.zeros(n, pad_ch, h, w, device=x.device, dtype=x.dtype)
            main = torch.cat([main, padding], dim=1)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        out = main + ext
        return self.out_activation(out), max_indices


class RegularBottleneck(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1,
                 dilation=1, asymmetric=False, dropout_prob=0.1):
        super().__init__()

        internal_channels = channels // 4

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels,
                          kernel_size=(kernel_size, 1), padding=(padding, 0), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(),
                nn.Conv2d(internal_channels, internal_channels,
                          kernel_size=(1, kernel_size), padding=(0, padding), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels,
                          kernel_size=kernel_size, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()

        internal_channels = in_channels // 4

        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.main_unpool = nn.MaxUnpool2d(kernel_size=2)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )
        self.ext_tconv = nn.ConvTranspose2d(
            internal_channels, internal_channels,
            kernel_size=2, stride=2, bias=False
        )
        self.ext_tconv_bn = nn.BatchNorm2d(internal_channels)
        self.ext_tconv_act = nn.PReLU()

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv(x)
        main = _max_unpool(self.main_unpool, main, max_indices, output_size)

        ext = self.ext_conv1(x)
        ext = self.ext_tconv(ext, output_size=output_size)
        ext = self.ext_tconv_bn(ext)
        ext = self.ext_tconv_act(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        return self.out_activation(main + ext)


class RegularBottleneckDecoder(nn.Module):
    def __init__(self, channels, dropout_prob=0.1):
        super().__init__()

        internal_channels = channels // 4

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)


@register_model("enet")
class ENetModel(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        self.num_classes = num_classes

        self.initial_block = InitialBlock(in_channels, out_channels=16)

        self.downsample1_0 = DownsamplingBottleneck(16, 64, dropout_prob=0.01)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01)

        self.downsample2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.regular4_1 = RegularBottleneckDecoder(64, dropout_prob=0.1)
        self.regular4_2 = RegularBottleneckDecoder(64, dropout_prob=0.1)

        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.regular5_1 = RegularBottleneckDecoder(16, dropout_prob=0.1)

        self.transposed_conv = nn.ConvTranspose2d(
            16, num_classes,
            kernel_size=3, stride=2, padding=1,
            output_padding=1, bias=False
        )

    def forward(self, x):
        x = self.initial_block(x)

        stage1_size = x.size()
        x, max_idx1 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        stage2_size = x.size()
        x, max_idx2 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        x = self.upsample4_0(x, max_idx2, output_size=stage2_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        x = self.upsample5_0(x, max_idx1, output_size=stage1_size)
        x = self.regular5_1(x)

        x = self.transposed_conv(x)

        return x


def build_model(num_classes=1):
    return ENetModel(num_classes=num_classes, in_channels=3)
