from .backbones import *
import torch.nn as nn

sw_cfg = dict(type='SW',
            sw_type=2,
            num_pergroup=16,
            T=5,
            tie_weight=False,
            momentum=0.9,
            affine=True)

class SWCounter_VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(SWCounter_VGG, self).__init__()

        self.vgg = vgg19(sw_cfg)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x):
        x = self.vgg(x)
        x = self.upsample(x)
        return x

class SWCounter_ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SWCounter_ResNet, self).__init__()
        
        resnet_sw = resnet50(pretrained=True, sw_cfg=sw_cfg)
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