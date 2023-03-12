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

def upsample(x, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

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
        #return (1+gamma)*(x)+beta

class VGGEncoder(nn.Module):
    def __init__(self, pretrained=True, trunc_at=-1):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:trunc_at])

    def forward(self, x):
        return self.encoder(x)
    
class Generator(nn.Module):
    def __init__(self, style_dim=64, pretrained=True):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)
        vgg_feats = list(vgg.features.children())
        self.stage1 = nn.Sequential(*vgg_feats[:4])
        self.stage2 = nn.Sequential(*vgg_feats[4:9])
        self.stage3 = nn.Sequential(*vgg_feats[9:18])
        self.stage4 = nn.Sequential(*vgg_feats[18:27])

        self.adain1 = AdaIN2d(style_dim, 64)
        self.adain2 = AdaIN2d(style_dim, 128)
        self.adain3 = AdaIN2d(style_dim, 256)
        self.adain4 = AdaIN2d(style_dim, 512)

        self.dec4 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 256)
        )

        self.dec3 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 128)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64)
        )

        self.head = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 3, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x, z=None):
        x1 = self.stage1(x) # 64
        x2 = self.stage2(x1) # 128
        x3 = self.stage3(x2) # 256
        x4 = self.stage4(x3) # 512

        if z is not None:
            x1 = self.adain1(x1, z)
            x2 = self.adain2(x2, z)
            x3 = self.adain3(x3, z)
            x4 = self.adain4(x4, z)

        x = self.dec4(x4) # 256
        x = upsample(x) # 256
        x = torch.cat([x, x3], dim=1) # 512

        x = self.dec3(x) # 128
        x = upsample(x) # 128
        x = torch.cat([x, x2], dim=1) # 256

        x = self.dec2(x) # 64
        x = upsample(x) # 64
        x = torch.cat([x, x1], dim=1) # 128

        x = self.dec1(x) # 64

        x = self.head(x)

        return x
    
class ReverseVGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(64, 64),
            ConvBlock(64, 3, bn=False, relu=False)
        )

    def forward(self, x):
        return self.decoder(x)
    
class AdaINAutoEncoder(nn.Module):
    def __init__(self, style_dim=64):
        super().__init__()
        self.encoder = VGGEncoder(pretrained=True, trunc_at=21)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.adain = AdaIN2d(style_dim, 512)
        self.decoder = ReverseVGGDecoder()

    def forward(self, x, z=None):
        x = self.encoder(x)
        if z is not None:
            x = self.adain(x, z)
        t = x
        x = self.decoder(x)
        return x, t
    

class MultiScaleVGGEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        self.stage1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.stage2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.stage3 = nn.Sequential(*list(vgg.features.children())[33:43])

        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024),
            ConvBlock(1024, 512)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 256)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 128)
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

        y = torch.cat([y1, y2, y3], dim=1)

        return y, x3

class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 1, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        y = self.head(x)
        return y

class DensityRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = MultiScaleVGGEncoder(pretrained=pretrained)
        self.channel_reduce = nn.Sequential(
            ConvBlock(512+256+128, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128)
        )

        self.head = nn.Sequential(
            ConvBlock(128, 1, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        f, f_var = self.enc(x)
        f_inv = self.channel_reduce(f)
        density = self.head(f_inv)
        return f_inv, f_var, density

class DGVCCNet(nn.Module):
    def __init__(self, style_dim=64, pretrained=True):
        super().__init__()
        # self.gen = AdaINAutoEncoder(style_dim=style_dim)
        # self.gen_cyc = AdaINAutoEncoder(style_dim=style_dim)
        self.gen = Generator(style_dim=style_dim, pretrained=pretrained)
        self.gen_cyc = Generator(style_dim=style_dim, pretrained=pretrained)
        self.reg = DensityRegressor(pretrained=pretrained)

    def forward_dg(self, x, z1, z2, mode):
        if mode == 'gen':
            x_gen = self.gen(x, z1)
            x_gen2 = self.gen(x, z2)
            x_cyc = self.gen_cyc(x_gen)

            _, f_var, _ = self.reg(x)
            _, f_gen_var, d_gen = self.reg(x_gen)

            loss_cyc = F.mse_loss(x, x_cyc)
            loss_div = -torch.clamp(F.mse_loss(x_gen, x_gen2), max=1)

            f_var_ = f_var.view(f_var.size(0), f_var.size(1), -1)
            f_gen_var_ = f_gen_var.view(f_gen_var.size(0), f_gen_var.size(1), -1)

            f_var_normed = f_var_ / (torch.norm(f_var_, dim=1, keepdim=True) + 1e-8)
            f_gen_var_normed = f_gen_var_ / (torch.norm(f_gen_var_, dim=1, keepdim=True) + 1e-8)

            sim_var = torch.bmm(f_var_normed.transpose(1, 2), f_gen_var_normed)

            loss_ortho = torch.sum(torch.pow(torch.diagonal(sim_var, dim1=-2, dim2=-1), 2)) / sim_var.size(0)

            print('loss_cyc: {:.4f}, loss_div: {:.4f}, loss_ortho: {:.4f}'.format(loss_cyc.item(), loss_div.item(), loss_ortho.item()))

            return d_gen, loss_cyc, loss_div, loss_ortho
        
        elif mode == 'reg':
            x_gen = self.gen(x, z1)

            f_inv, f_var, d = self.reg(x)
            f_gen_inv, f_gen_var, d_gen = self.reg(x_gen)

            loss_sim = F.mse_loss(f_inv, f_gen_inv)

            # f_inv_ = f_inv.view(f_inv.size(0), f_inv.size(1), -1)
            # f_gen_inv_ = f_gen_inv.view(f_gen_inv.size(0), f_gen_inv.size(1), -1)

            # sim_inv = torch.bmm(f_inv_.transpose(1, 2), f_gen_inv_)

            # sim_gt = torch.linspace(0, f_inv.size(2)*f_inv.size(3)-1, f_inv.size(2)*f_inv.size(3)).unsqueeze(0).repeat(f_inv.size(0), 1).to(f_inv.device)
            # loss_sim = F.cross_entropy(sim_inv, sim_gt.long())

            print('loss_sim: {:.4f}'.format(loss_sim.item()))

            return d, d_gen, loss_sim
        
        elif mode == 'test':
            x_gen = self.gen(x, z1)
            x_gen2 = self.gen(x, z2)
            x_cyc = self.gen_cyc(x_gen)

            _, _, d = self.reg(x)
            _, _, d_gen = self.reg(x_gen)

            return d, d_gen, x_gen, x_gen2, x_cyc
    
    def forward(self, x):
        return self.reg(x)[-1]