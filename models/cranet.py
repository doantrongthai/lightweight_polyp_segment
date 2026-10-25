import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from . import register_model
from .pranet import Aggregation, BasicConv2d, RFBModified


def _resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


@register_model("cranet")
class CRANet(nn.Module):
    def __init__(self, classes=1, channel=32):
        super().__init__()
        if classes != 1:
            raise ValueError("CRANet in this pipeline is configured for binary segmentation.")
        self.resnet = resnet50(weights=None)
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

        crop_4 = _resize(ra5_feat, x4.shape[2:])
        ra = 1 - torch.sigmoid(crop_4)
        ra = ra.expand(-1, 2048, -1, -1).mul(x4)
        ra = self.ra4_conv5(F.relu(self.ra4_conv4(F.relu(self.ra4_conv3(F.relu(self.ra4_conv2(self.ra4_conv1(ra))))))))
        x = ra + crop_4

        crop_3 = _resize(x, x3.shape[2:])
        ra = 1 - torch.sigmoid(crop_3)
        ra = ra.expand(-1, 1024, -1, -1).mul(x3)
        ra = self.ra3_conv4(F.relu(self.ra3_conv3(F.relu(self.ra3_conv2(self.ra3_conv1(ra))))))
        x = ra + crop_3

        crop_2 = _resize(x, x2.shape[2:])
        ra = 1 - torch.sigmoid(crop_2)
        ra = ra.expand(-1, 512, -1, -1).mul(x2)
        ra = self.ra2_conv4(F.relu(self.ra2_conv3(F.relu(self.ra2_conv2(self.ra2_conv1(ra))))))
        return _resize(ra + crop_2, size)
