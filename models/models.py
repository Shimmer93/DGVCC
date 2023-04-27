import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y

def upsample(x, scale_factor=2, mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg_feats = list(vgg.features.children())
        self.enc = nn.Sequential(*vgg_feats[:26])

        self.dec = nn.Sequential(
            ConvBlock(512, 512, bn=True),
            ConvBlock(512, 256, bn=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(256, 256, bn=True),
            ConvBlock(256, 256, bn=True),
            ConvBlock(256, 256, bn=True),
            ConvBlock(256, 128, bn=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 128, bn=True),
            ConvBlock(128, 64, bn=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(64, 64, bn=True),
            ConvBlock(64, 3, kernel_size=1, padding=0, relu=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
    
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
    
class DensityRegressor2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        self.stage1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.stage2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.stage3 = nn.Sequential(*list(vgg.features.children())[33:43])

        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True)
        )

        self.den_head = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 2, kernel_size=1, padding=0, relu=False),
        )

        self.cls_head = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 3, kernel_size=1, padding=0, relu=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)

        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)

        y_cat = torch.cat([y1, y2, y3], dim=1)

        c = self.cls_head(x3)
        resized_c = upsample(c, scale_factor=8, mode='nearest')
        d = self.den_head(y_cat)
        dc = d[:, 0:1] * resized_c[:, 1:3].sum(dim=1, keepdim=True) + d[:, 1:2] * resized_c[:, 2:3]
        dc = upsample(dc, scale_factor=4)

        return dc, d, c
    
class DensityRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        self.stage1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.stage2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.stage3 = nn.Sequential(*list(vgg.features.children())[33:43])

        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True)
        )

        self.den_head = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 1, kernel_size=1, padding=0)
        )

        self.cls_head = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 3, kernel_size=1, padding=0, relu=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)

        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)

        y_cat = torch.cat([y1, y2, y3], dim=1)

        c = self.cls_head(x3)
        resized_c = upsample(c, scale_factor=8, mode='nearest')
        d = self.den_head(y_cat)
        dc = d * resized_c[:, 1:3].sum(dim=1, keepdim=True)
        dc = upsample(dc, scale_factor=4)

        return dc, d, c
    
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

def get_models():
    gen = Generator()
    reg = DensityRegressor()
    return gen, reg