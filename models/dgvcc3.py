import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from enum import Enum

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
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)

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
    def __init__(self, style_dim=64, pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        vgg_feats = list(vgg.features.children())
        self.enc = nn.Sequential(*vgg_feats[:23])
        # self.enc.requires_grad_(False)
        self.adain = AdaIN2d(style_dim, 256)
        self.dec = nn.Sequential(
            ConvBlock(256, 256, bn=True),
            ConvBlock(256, 256, bn=True),
            ConvBlock(256, 256, bn=True),
            ConvBlock(256, 128, bn=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 128, bn=True),
            ConvBlock(128, 64, bn=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(64, 64, bn=True),
            ConvBlock(64, 3, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x, z=None):
        # x0 = self.enc(x)
        # if z is not None:
        #     x_ = self.adain(x0, z)
        # else:
        #     x_ = x0
        # x_ = self.dec(x_)
        # x1 = torch.tanh(x_)
        # x_new = torch.clamp(0.2 * x1 + x, min=-1, max=1)
        # return x_new, x1
        x0 = self.enc(x)
        if z is not None:
            x_ = self.adain(x0, z)
        else:
            x_ = x0
        x_ = self.dec(x_)
        x_new = torch.tanh(x_)
        return x_new
    
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

        self.to_inv = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=False),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2)
        )

        self.final_layer = ConvBlock(256, 1, kernel_size=1, padding=0, bn=False)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        y_shallow = [x1, x2, x3]

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

        y_inv = self.to_inv(y_cat)

        y = self.final_layer(y_inv)
        y = upsample(y, scale_factor=4)

        return y_cat, y_inv, y


class ModelComponent(Enum):
    GENERATOR = 1
    GENERATOR_CYCLIC = 2
    REGRESSOR = 3
    ALL = 4

class DGVCCModel(nn.Module):
    def __init__(self, style_dim=64, pretrained=True):
        super().__init__()
        self.gen = Generator(style_dim, pretrained)
        self.gen_cyc = Generator(style_dim, pretrained)
        self.reg = DensityRegressor(pretrained)

    def _dissim_loss(self, x, y):
        return torch.mean(x * y)
    
    def _div_loss(self, x, y, thres=None):
        if thres is None:
            return -F.mse_loss(x, y)
        else:
            return -torch.max(F.mse_loss(x, y), thres)
    
    def _ortho_loss(self, x, y):
        x_ = x.view(x.shape[0], x.shape[1], -1)
        y_ = y.view(y.shape[0], y.shape[1], -1)
        logits = torch.bmm(x_.transpose(1, 2), y_)
        loss = torch.sum(torch.pow(torch.diagonal(logits, dim1=-2, dim2=-1), 2))
        return loss
    
    def _rec_loss(self, x, y):
        x_ = x.view(x.shape[0], x.shape[1], -1)
        y_ = y.view(y.shape[0], y.shape[1], -1)
        logits = torch.bmm(x_.transpose(1, 2), y_)
        gt = torch.linspace(0, y.shape[2] * y.shape[3] - 1, y.shape[2] * y.shape[3]).unsqueeze(0).repeat(y.shape[0], 1).to(x.device)
        loss = F.cross_entropy(logits, gt.long())
        return loss
    
    def _multi_dissim_loss(self, x, y):
        return self._dissim_loss(x[0], y[0]) + self._dissim_loss(x[1], y[1]) + self._dissim_loss(x[2], y[2])

    def forward_generator(self, x):
        x_rec = self.gen(x)
        loss_rec = F.mse_loss(x_rec, x)
        return x_rec, loss_rec
    
    def forward_regressor(self, x):
        return self.reg(x)[-1]
    
    def forward_joint(self, x, z1, z2):
        x_gen = self.gen(x, z1)
        x_gen2 = self.gen(x, z2)
        res = x_gen - x
        res2 = x_gen2 - x
        b = x.shape[0]

        x_cyc = self.gen_cyc(x_gen)
        x_cyc2 = self.gen_cyc(x_gen2)

        # _, res_gen = self.gen(x_gen)
        # _, res_gen2 = self.gen(x_gen2)

        x_cat = torch.cat([x, x_gen, x_gen2])
        f_shallow_cat, f_inv_cat, d_cat = self.reg(x_cat)
        # f_shallow_ = [f_shallow_cat[0][:b], f_shallow_cat[1][:b], f_shallow_cat[2][:b]]
        # f_shallow_gen_ = [f_shallow_cat[0][b:2*b], f_shallow_cat[1][b:2*b], f_shallow_cat[2][b:2*b]]
        # f_shallow_gen2_ = [f_shallow_cat[0][2*b:], f_shallow_cat[1][2*b:], f_shallow_cat[2][2*b:]]
        f_shallow_ = f_shallow_cat[:b]
        f_shallow_gen_ = f_shallow_cat[b:2*b]
        f_shallow_gen2_ = f_shallow_cat[2*b:]

        d = d_cat[:b]
        d_gen = d_cat[b:2*b]
        d_gen2 = d_cat[2*b:]

        loss_cyc = F.mse_loss(x_cyc, x) + F.mse_loss(x_cyc2, x)

        loss_div = -torch.clamp(F.mse_loss(x_gen, x_gen2), max=0.1)
        loss_var = -torch.clamp(torch.mean(torch.var(res, dim=(2, 3)) + torch.var(res2, dim=(2, 3))), max=0.1)
        loss_mean = torch.mean(torch.mean(res**2, dim=(2, 3)) + torch.mean(res2**2, dim=(2, 3)))

        loss_dissim = self._dissim_loss(f_shallow_, f_shallow_gen_) + \
            self._dissim_loss(f_shallow_, f_shallow_gen2_) + \
            self._dissim_loss(f_shallow_gen_, f_shallow_gen2_)
        
        loss_sim = F.mse_loss(d, d_gen) + F.mse_loss(d, d_gen2) + F.mse_loss(d_gen, d_gen2)

        return d_cat, loss_div, loss_dissim, loss_sim, loss_var, loss_cyc, loss_mean
    
    def forward_augmented(self, x, z1, z2, z3):
        self.gen.eval()
        with torch.no_grad():
            x_gen1 = self.gen(x, z1)
            x_gen2 = self.gen(x, z2)
            x_gen3 = self.gen(x, z3)
        x_cat = torch.cat([x, x_gen1, x_gen2, x_gen3])
        d_cat = self.reg(x_cat)[-1]

        return x_cat, d_cat
    
    def forward_test(self, x, z1, z2):
        x_gen = self.gen(x, z1)
        x_gen2 = self.gen(x, z2)

        d = self.reg(x)[-1]
        d_gen = self.reg(x_gen)[-1]

        return d, d_gen, x_gen, x_gen2
    
    def get_params(self, component):
        if component == ModelComponent.GENERATOR:
            return self.gen.parameters()
        elif component == ModelComponent.GENERATOR_CYCLIC:
            return self.gen_cyc.parameters()
        elif component == ModelComponent.REGRESSOR:
            return self.reg.parameters()
        elif component == ModelComponent.ALL:
            return self.parameters()
        else:
            raise ValueError('Invalid component')

    def load_sd(self, sd_path, component, device):
        sd = torch.load(sd_path, map_location=device)
        if component == ModelComponent.GENERATOR:
            self.gen.load_state_dict(sd)
        elif component == ModelComponent.GENERATOR_CYCLIC:
            self.gen_cyc.load_state_dict(sd)
        elif component == ModelComponent.REGRESSOR:
            self.reg.load_state_dict(sd)
        elif component == ModelComponent.ALL:
            self.load_state_dict(sd)
        else:
            raise ValueError('Invalid component')
        
    def save_sd(self, sd_path, component):
        if component == ModelComponent.GENERATOR:
            torch.save(self.gen.state_dict(), sd_path)
        elif component == ModelComponent.GENERATOR_CYCLIC:
            torch.save(self.gen_cyc.state_dict(), sd_path)
        elif component == ModelComponent.REGRESSOR:
            torch.save(self.reg.state_dict(), sd_path)
        elif component == ModelComponent.ALL:
            torch.save(self.state_dict(), sd_path)
        else:
            raise ValueError('Invalid component')