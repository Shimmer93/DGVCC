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

class Generator(nn.Module):
    def __init__(self, style_dim=64, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        vgg_feats = list(vgg.features.children())
        self.stage1 = nn.Sequential(*vgg_feats[:6])
        self.stage2 = nn.Sequential(*vgg_feats[6:13])
        self.stage3 = nn.Sequential(*vgg_feats[13:23])

        self.adain1 = AdaIN2d(style_dim, 64)
        self.adain2 = AdaIN2d(style_dim, 128)
        self.adain3 = AdaIN2d(style_dim, 256)

        self.dec3 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 128)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64)
        )

        self.head = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 3, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x, z=None):
        x1 = self.stage1(x) # 64
        x2 = self.stage2(x1) # 128
        x3 = self.stage3(x2) # 256

        if z is not None:
            x1 = self.adain1(x1, z)
            x2 = self.adain2(x2, z)
            x3 = self.adain3(x3, z)

        x = self.dec3(x3) # 128
        x = upsample(x) # 128
        x = torch.cat([x, x2], dim=1) # 128+128

        x = self.dec2(x) # 64
        x = upsample(x) # 64
        x = torch.cat([x, x1], dim=1) # 64+64

        x = self.dec1(x) # 64

        x = self.head(x) # 3

        return x
    
class DensityRegressor(nn.Module):
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

        self.var_to_inv = nn.Sequential(
            ConvBlock(512+256+128, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128)
        )

        self.final_layer = ConvBlock(128, 1, kernel_size=1, padding=0, bn=False, relu=False)

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

        y_var = torch.cat([y1, y2, y3], dim=1)

        y_inv = self.var_to_inv(y_var)

        y = self.final_layer(y_inv)

        return y_var, y_inv, y

class DGVCCNet2(nn.Module):
    def __init__(self, style_dim=64, pretrained=True):
        super().__init__()
        self.gen = Generator(style_dim=style_dim, pretrained=pretrained)
        self.gen_cyc = Generator(style_dim=style_dim, pretrained=pretrained)
        self.reg = DensityRegressor(pretrained=pretrained)

    def forward_rec(self, x):
        x_rec = self.gen(x)
        loss_rec = F.mse_loss(x_rec, x)
        return x_rec, loss_rec

    def forward_den(self, x):
        _, _, den = self.reg(x)
        return den
    
    def forward_dg(self, x, z1, z2, mode):
        if mode == 'gen':
            x_gen = self.gen(x, z1)
            x_gen2 = self.gen(x, z2)
            x_cyc = self.gen_cyc(x_gen)

            x_cat = torch.cat([x, x_gen])
            _, f_var_cat, d_cat = self.reg(x_cat)
            f_var = f_var_cat[:x.shape[0]]
            f_gen_var = f_var_cat[x.shape[0]:]
            d_gen = d_cat[x.shape[0]:]
            # _, f_var, _ = self.reg(x)
            # _, f_gen_var, d_gen = self.reg(x_gen)

            loss_cyc = F.mse_loss(x, x_cyc)
            loss_div = -torch.clamp(F.mse_loss(x_gen, x_gen2), max=1)

            loss_dissim = -torch.clamp(F.mse_loss(f_var, f_gen_var), max=10)

            print('loss_cyc: {:.4f}, loss_div: {:.4f}, loss_dissim: {:.4f}'.format(loss_cyc.item(), loss_div.item(), loss_dissim.item()))

            return d_gen, loss_cyc, loss_div, loss_dissim
        
        elif mode == 'reg':
            x_gen = self.gen(x, z1)

            x_cat = torch.cat([x, x_gen], dim=0)
            _, f_var_cat, d_cat = self.reg(x_cat)
            f_var = f_var_cat[:x.shape[0]]
            f_gen_var = f_var_cat[x.shape[0]:]

            # f_inv, f_var, d = self.reg(x)
            # f_gen_inv, f_gen_var, d_gen = self.reg(x_gen)

            loss_sim = F.mse_loss(f_var, f_gen_var)

            print('loss_sim: {:.4f}'.format(loss_sim.item()))

            # return d, d_gen, loss_sim
            return d_cat, loss_sim
        
        elif mode == 'reg_final':
            self.gen.eval()
            with torch.no_grad():
                x_gen = self.gen(x, z1).detach()

            x_cat = torch.cat([x, x_gen], dim=0)
            _, _, d_cat = self.reg(x_cat)

            return d_cat
        
        elif mode == 'test':
            x_gen = self.gen(x, z1)
            x_gen2 = self.gen(x, z2)
            x_cyc = self.gen_cyc(x_gen)

            _, _, d = self.reg(x)
            _, _, d_gen = self.reg(x_gen)

            return d, d_gen, x_gen, x_gen2, x_cyc

    def load_sd(self, sd_path, mode, device):
        sd = torch.load(sd_path, map_location=device)
        if mode == 'gen':
            self.gen.load_state_dict(sd)
        elif mode == 'gen_cyc':
            self.gen_cyc.load_state_dict(sd)
        elif mode == 'reg':
            self.reg.load_state_dict(sd)
        elif mode == 'all':
            # from collections import OrderedDict
            # new_sd = OrderedDict()
            # for k, v in sd.items():
            #     if k.startswith('gen.'):
            #         new_sd[k[4:]] = v
            # self.gen.load_state_dict(new_sd)
            self.load_state_dict(sd)
        else:
            raise ValueError('Invalid mode')
        
    def save_sd(self, sd_path, mode):
        if mode == 'gen':
            torch.save(self.gen.state_dict(), sd_path)
        elif mode == 'gen_cyc':
            torch.save(self.gen_cyc.state_dict(), sd_path)
        elif mode == 'reg':
            torch.save(self.reg.state_dict(), sd_path)
        elif mode == 'all':
            torch.save(self.state_dict(), sd_path)
        else:
            raise ValueError('Invalid mode')