from __future__ import absolute_import

from .resnet_ibn import resnet18_ibn_a, resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a, \
                        resnet18_ibn_b, resnet34_ibn_b, resnet50_ibn_b, resnet101_ibn_b, resnet152_ibn_b
from .densenet_ibn import densenet121_ibn_a, densenet169_ibn_a, densenet201_ibn_a, densenet161_ibn_a
from .resnext_ibn import resnext50_ibn_a, resnext101_ibn_a, resnext152_ibn_a
from .se_resnet_ibn import se_resnet50_ibn_a, se_resnet101_ibn_a, se_resnet152_ibn_a

import torch.nn as nn

class IBNCounter_ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(IBNCounter_ResNet, self).__init__()
        
        resnet_sw = resnet50_ibn_b(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet_sw.children())[:7])
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=16)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x