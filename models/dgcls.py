import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, insn = False, relu=True):
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
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg_feats = list(vgg.features.children())
        self.enc = nn.Sequential(*vgg_feats[:26])
        # self.enc.requires_grad_(False)

        # self.z_enc = nn.Sequential(
        #     ConvBlock(128, 256),
        #     ConvBlock(256, 512)
        # )

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

        self._init_params()

    def _init_params(self):
        for m in self.dec.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.enc(x)
    
    def decode(self, x, z=None):
        if z is not None:
            z = self.z_enc(z)
            x = x + z
        x = self.dec(x)
        return x

    def forward(self, x, z=None):
        x = self.enc(x)
        if z is not None:
            z = self.z_enc(z)
            x = x + z
        x = self.dec(x)
        return x

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
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 1, kernel_size=1, padding=0)
        )

        self.cls_head = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
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
        resized_c = upsample(c, scale_factor=4, mode='nearest')
        d = self.den_head(y_cat) * resized_c
        d = upsample(d, scale_factor=4)

        return d, c
    
class DGCLS(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.gen = Generator()
        self.reg = DensityRegressor(pretrained=pretrained)

    def _add_noise(self, x, n):
        return n

    def forward_gen(self, x, z=None):
        self.gen.requires_grad_(True)
        self.reg.requires_grad_(False)

        f = self.gen.encode(x)
        n = self.gen.decode(f)
        # n0 = self.gen.decode(f)
        x_n = self._add_noise(x, n)

        d_n, c_n = self.reg(x_n)

        return d_n, c_n, x_n
    
    def forward_reg(self, x, z=None):
        self.gen.requires_grad_(False)
        self.reg.requires_grad_(True)
        # self.reg.cls_head.requires_grad_(True)

        n = self.gen(x)
        x_n = self._add_noise(x, n)
        x_cat = torch.cat([x, x_n], dim=0)

        d_cat, c_cat = self.reg(x_cat)

        d, d_n = torch.chunk(d_cat, 2, dim=0)
        c, c_n = torch.chunk(c_cat, 2, dim=0)

        return d, d_n, c, c_n
    
    def forward(self, x):
        return self.reg(x)
    
    def forward_vis(self, x, z=None):
        n = self.gen(x)
        x_n = self._add_noise(x, n)
        x_cat = torch.cat([x, x_n], dim=0)

        d_cat, c_cat = self.reg(x_cat)

        d, d_n = torch.chunk(d_cat, 2, dim=0)
        c, c_n = torch.chunk(c_cat, 2, dim=0)

        return d, d_n, c, c_n, x_n
    
    def forward_test(self, x, z=None):
        n = self.gen(x)
        x_n = self._add_noise(x, n)
        return x_n
    

if __name__ == '__main__':
    m = Generator()
    x = torch.randn(1, 3, 16, 16)
    y = m(x)
    print(y)