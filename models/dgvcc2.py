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
        return (1 + gamma) * self.norm(x) + beta, gamma, beta

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
            x, gamma, beta = self.adain(x, z)
            x = self.dec(x)
            x_new = torch.tanh(x)
            return x_new, gamma, beta
        else:
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

        self.to_inv = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=False),
            nn.Dropout2d(p=0.5)
        )

        self.to_var = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0, bn=False),
            nn.Dropout2d(p=0.5)
        )

        self.part_num = 1024
        variance = 1.0
        self.mem_inv = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))
        self.mem_var = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))

        self.final_layer = ConvBlock(256, 1, kernel_size=1, padding=0, bn=False, relu=False)

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
        # y_shallow = [y1, y2, y3]

        y_cat = torch.cat([y1, y2, y3], dim=1)

        y_inv = self.to_inv(y_cat)
        y_var = self.to_var(y_cat)

        b, c, h, w = y_inv.shape
        m_inv = self.mem_inv.repeat(b, 1, 1)
        m_key_inv = m_inv.transpose(1, 2)
        y_inv_ = y_inv.view(b, c, -1)
        logits_inv = torch.bmm(m_key_inv, y_inv_)
        y_inv_new = torch.bmm(m_key_inv.transpose(1, 2), F.softmax(logits_inv, dim=1))
        y_inv_new_ = y_inv_new.view(b, c, h, w)

        m_var = self.mem_var.repeat(b, 1, 1)
        m_key_var = m_var.transpose(1, 2)
        y_var_ = y_var.view(b, c, -1)
        logits_var = torch.bmm(m_key_var, y_var_)
        y_var_new = torch.bmm(m_key_var.transpose(1, 2), F.softmax(logits_var, dim=1))
        y_var_new_ = y_var_new.view(b, c, h, w)

        y = self.final_layer(y_inv_new_)
        y = upsample(y, scale_factor=4)
        y = torch.relu(y)

        return y_shallow, y_inv, y_var, y_inv_new_, y_var_new_, y


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
        x_gen, gamma1, beta1 = self.gen(x, z1)
        x_gen2, gamma2, beta2 = self.gen(x, z2)
        b = x.shape[0]

        x_cyc = self.gen_cyc(x_gen)

        x_cat = torch.cat([x, x_gen, x_gen2])
        f_shallow_cat, f_inv_cat, f_var_cat, f_inv_new_cat, f_var_new_cat, d_cat = self.reg(x_cat)
        f_shallow = [f_shallow_cat[0][:b], f_shallow_cat[1][:b], f_shallow_cat[2][:b]]
        f_shallow_gen = [f_shallow_cat[0][b:2*b], f_shallow_cat[1][b:2*b], f_shallow_cat[2][b:2*b]]
        f_shallow_gen2 = [f_shallow_cat[0][2*b:], f_shallow_cat[1][2*b:], f_shallow_cat[2][2*b:]]

        d = d_cat[:b]
        d_gen = d_cat[b:2*b]
        d_gen2 = d_cat[2*b:]

        loss_cyc = F.mse_loss(x, x_cyc)
        loss_div = -torch.clamp(F.mse_loss(x_gen, x_gen2), max=0.1)

        loss_dissim = self._multi_dissim_loss(f_shallow, f_shallow_gen) + \
            self._multi_dissim_loss(f_shallow, f_shallow_gen2) + \
            self._multi_dissim_loss(f_shallow_gen, f_shallow_gen2)
        
        loss_sim = F.mse_loss(d, d_gen) + F.mse_loss(d, d_gen2) + F.mse_loss(d_gen, d_gen2)

        loss_rec = self._rec_loss(f_inv_cat, f_inv_new_cat) # + self._rec_loss(f_var_cat, f_var_new_cat)
        loss_ortho = self._ortho_loss(f_inv_cat, f_var_cat)

        return d_cat, loss_cyc, loss_div, loss_dissim, loss_sim, loss_rec, loss_ortho

    def forward_joint0(self, x, z1, z2):
        x_gen, gamma1, beta1 = self.gen(x, z1)
        x_gen2, gamma2, beta2 = self.gen(x, z2)
        b = x.shape[0]
        # z_cat = torch.cat([z1, z2])
        # x_gen_cat = self.gen(x, z_cat)
        # x_gen = x_gen_cat[:b]
        # x_gen2 = x_gen_cat[b:]

        x_cyc = self.gen_cyc(x_gen)

        x_cat = torch.cat([x, x_gen])
        # f_var_cat, f_inv_cat, d_cat = self.reg(x_cat)
        f_shallow_cat, f_inv_cat, f_var_cat, f_inv_new_cat, f_var_new_cat, d_cat = self.reg(x_cat)

        f_shallow = [f_shallow_cat[0][:b], f_shallow_cat[1][:b], f_shallow_cat[2][:b]]
        f_shallow_gen = [f_shallow_cat[0][b:], f_shallow_cat[1][b:], f_shallow_cat[2][b:]]
        f_inv = f_inv_cat[:b]
        f_inv_gen = f_inv_cat[b:]
        f_var = f_var_cat[:b]
        f_var_gen = f_var_cat[b:]
        f_inv_new = f_inv_new_cat[:b]
        f_inv_gen_new = f_inv_new_cat[b:]
        f_var_new = f_var_new_cat[:b]
        f_var_gen_new = f_var_new_cat[b:]

        d = d_cat[:b]
        d_gen = d_cat[b:]
        
        loss_cyc = F.mse_loss(x, x_cyc)
        loss_div = -torch.clamp(F.mse_loss(x_gen, x_gen2), max=0.5)
        # loss_div = torch.clamp(self._div_loss(gamma1, gamma2) + self._div_loss(beta1, beta2), min=-1)

        loss_dissim = self._dissim_loss(f_shallow[0], f_shallow_gen[0]) + \
            self._dissim_loss(f_shallow[1], f_shallow_gen[1]) + self._dissim_loss(f_shallow[2], f_shallow_gen[2])
        # loss_dissim = self._dissim_loss(f_var, f_var_gen) # + self._dissim_loss(f_var_new, f_var_gen_new)
        
        # loss_sim = F.mse_loss(f_inv, f_inv_gen) + F.mse_loss(f_inv_new, f_inv_gen_new)
        loss_sim = F.mse_loss(d, d_gen)
        loss_rec = self._rec_loss(f_inv_cat, f_inv_new_cat) # + self._rec_loss(f_var_cat, f_var_new_cat)
        loss_ortho = self._ortho_loss(f_inv_cat, f_var_cat)

        return d_cat, loss_cyc, loss_div, loss_dissim, loss_sim, loss_rec, loss_ortho
    
    def forward_augmented(self, x, z1, z2, z3):
        self.gen.eval()
        with torch.no_grad():
            x_gen1 = self.gen(x, z1)[0]
            x_gen2 = self.gen(x, z2)[0]
            x_gen3 = self.gen(x, z3)[0]
        x_cat = torch.cat([x, x_gen1, x_gen2, x_gen3])
        d_cat = self.reg(x_cat)[-1]

        return x_cat, d_cat
    
    def forward_test(self, x, z1, z2):
        x_gen = self.gen(x, z1)[0]
        x_gen2 = self.gen(x, z2)[0]
        x_cyc = self.gen_cyc(x_gen)

        d = self.reg(x)[-1]
        d_gen = self.reg(x_gen)[-1]

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