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
        x = self.enc(x)
        if z is not None:
            x = self.adain(x, z)
        x = self.dec(x)
        x_new = torch.tanh(x)
        return x_new

class Generator_old(nn.Module):
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

        x = torch.clamp(x, -1, 1)

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
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=False, relu=False),
            nn.Dropout2d(p=0.5)
        )

        self.part_num = 1024
        variance = 1.0
        self.mem_inv = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))

        self.final_layer = ConvBlock(256, 1, kernel_size=1, padding=0, bn=False, relu=False)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        y_var = [x1, x2, x3]

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

        y_inv = self.var_to_inv(y_cat)

        memory = self.mem_inv.repeat(x.shape[0], 1, 1)
        memory_key = memory.transpose(1, 2)
        y_inv_ = y_inv.view(y_inv.shape[0], y_inv.shape[1], -1)
        logits = torch.bmm(memory_key, y_inv_)
        y_new = torch.bmm(memory_key.transpose(1, 2), F.softmax(logits, dim=1))
        y_new_ = y_new.view(y_new.shape[0], y_new.shape[1], y_inv.shape[2], y_inv.shape[3])

        y = self.final_layer(y_new_)
        y = upsample(y, scale_factor=4)
        y = torch.relu(y)

        return y_inv, y_var, y_new_, y

class DensityRegressor1(nn.Module):
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

        y_var = [x1, x2, x3]

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

        y_inv = self.var_to_inv(y_cat)

        y = self.final_layer(y_inv)
        y = upsample(y, scale_factor=4)
        y = torch.relu(y)

        return y_inv, y_var, y

class DensityRegressor0(nn.Module):
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

        y_var = x3

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

        y_inv = self.var_to_inv(y_cat)

        y = self.final_layer(y_inv)
        y = torch.relu(y)
        y = upsample(y, scale_factor=4)

        return y_inv, y_cat, y

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

    def forward_generator(self, x):
        x_rec = self.gen(x)
        loss_rec = F.mse_loss(x_rec, x)
        return x_rec, loss_rec
    
    def forward_regressor(self, x):
        return self.reg(x)[-1]
    
    def forward_joint(self, x, z1, z2):
        x_gen = self.gen(x, z1)
        x_gen2 = self.gen(x, z2)
        x_cyc = self.gen_cyc(x_gen)

        x_cat = torch.cat([x, x_gen])
        # f_var_cat, f_inv_cat, d_cat = self.reg(x_cat)
        f_inv_cat, f_var_cat, d_cat = self.reg(x_cat)
        f_inv = f_inv_cat[:x.shape[0]]
        f_gen_inv = f_inv_cat[x.shape[0]:]
        f_var1 = f_var_cat[0][:x.shape[0]]
        f_gen_var1 = f_var_cat[0][x.shape[0]:]
        f_var2 = f_var_cat[1][:x.shape[0]]
        f_gen_var2 = f_var_cat[1][x.shape[0]:]
        f_var3 = f_var_cat[2][:x.shape[0]]
        f_gen_var3 = f_var_cat[2][x.shape[0]:]

        loss_cyc = F.mse_loss(x, x_cyc)
        loss_div = -torch.clamp(F.mse_loss(x_gen, x_gen2), max=0.5)

        loss_sim = F.mse_loss(f_inv, f_gen_inv)
        loss_dissim = self._dissim_loss(f_var1, f_gen_var1) + self._dissim_loss(f_var2, f_gen_var2) + self._dissim_loss(f_var3, f_gen_var3)

        print(f'loss_cyc: {loss_cyc.item():.4f}, loss_div: {loss_div.item():.4f}, loss_sim: {loss_sim.item():.4f}, loss_dissim: {loss_dissim.item():.4f}')

        return d_cat, f_inv_cat, loss_cyc, loss_div, loss_sim, loss_dissim
    
    def forward_augmented(self, x, z1, z2, z3):
        self.gen.eval()
        with torch.no_grad():
            x_gen1 = self.gen(x, z1)
            x_gen2 = self.gen(x, z2)
            x_gen3 = self.gen(x, z3)
        x_cat = torch.cat([x, x_gen1, x_gen2, x_gen3])
        _, _, d_cat = self.reg(x_cat)

        return x_cat, d_cat
    
    def forward_test(self, x, z1, z2):
        x_gen = self.gen(x, z1)
        x_gen2 = self.gen(x, z2)
        x_cyc = self.gen_cyc(x_gen)

        _, _, d = self.reg(x)
        _, _, d_gen = self.reg(x_gen)

        return d, d_gen, x_gen, x_gen2, x_cyc
    
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