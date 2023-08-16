import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from collections import OrderedDict
from torchvision import models

from ..utils import build_norm_layer

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        mod = models.vgg19(pretrained=True)
        # self._initialize_weights()
        fsd = OrderedDict()
        # 10 convlution *(weight, bias) = 20 parameters
        idx_dict = {0:0, 2:3, 5:7, 7:10, 10:14, 12:17, 14:20, 16:23, 19:27, 21:30, 23:33, 25:36, 28:40, 30:43, 32:46, 34:49}

        # self.features.load_state_dict(mod.features.state_dict(), strict=False)
        for i in range(len(mod.features.state_dict().items())):
            temp_key = list(mod.features.state_dict().items())[i][0]
            idx = int(temp_key.split('.')[0])
            new_key = str(idx_dict[idx]) + temp_key.split('.')[1]
            fsd[new_key] = list(mod.features.state_dict().items())[i][1]
            
        # for i in range(len(self.features.state_dict().items())):
        #     temp_key = list(self.features.state_dict().items())[i][0]
            
        #     fsd[temp_key] = list(mod.state_dict().items())[i][1]
        self.features.load_state_dict(fsd, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, sw_cfg=None):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if sw_cfg:
                layers += [conv2d, build_norm_layer(sw_cfg, v)[1], nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(sw_cfg):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], sw_cfg=sw_cfg))
    return model