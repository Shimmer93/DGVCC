import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from enum import Enum
from losses.triplet import triplet_loss
from queue import Queue

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

def stylize_by_novel_style(x, lamda=1):
    mu = torch.mean(x, dim=(2, 3))
    sigma = torch.std(x, dim=(2, 3))
    novel_mu = torch.randn_like(mu) * sigma * lamda + mu
    novel_sigma = torch.randn_like(sigma) * sigma * lamda + sigma
    return (x - mu) / sigma * novel_sigma + novel_mu

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg_feats = list(vgg.features.children())
        self.enc = nn.Sequential(*vgg_feats[:26])
        self.enc.requires_grad_(False)

        # self.adain = AdaIN2d(64, 512)

        self.dec = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(64, 64),
            ConvBlock(64, 3, kernel_size=1, padding=0, relu=False),
            nn.Tanh()
        )

    def forward(self, x, z=None):
        x = self.enc(x)
        feats = x
        if z is not None:
            x = self.adain(x, z)
        x = self.dec(x)
        return x, feats
    
    def transfer(self, x, z):
        return self.adain(x, z)
    
    def decode(self, x):
        x = self.dec(x)
        return x
    
    def _get_sty(self, x):
        mu = torch.mean(x, dim=(2,3))
        std = torch.std(x, dim=(2,3))
        return torch.cat([mu, std], dim=1)

    def encode(self, x):
        f1 = self.enc[:4](x)
        f2 = self.enc[4:9](f1)
        f3 = self.enc[9:18](f2)
        f4 = self.enc[18:](f3)

        cot = f4
        sty = torch.cat([self._get_sty(f1), self._get_sty(f2), self._get_sty(f3)], dim=1)

        return cot, sty
    
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
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )

        self.to_var = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )

        self.final_layer = nn.Sequential(
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 1, kernel_size=1, padding=0)
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

        y_inv = self.to_inv(y_cat)
        y_var = self.to_var(y_cat)

        y = self.final_layer(y_inv)
        y = upsample(y, scale_factor=4)

        return y, y_inv, y_var
    
class MapAggregator(nn.Module):
    def __init__(self, num_maps):
        super().__init__()
        self.num_maps = num_maps
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(num_maps, num_maps, kernel_size=1, padding=0),
            ConvBlock(num_maps, num_maps, kernel_size=1, padding=0, relu=False),
        )

        self.local_attn = nn.Sequential(
            ConvBlock(num_maps, num_maps),
            ConvBlock(num_maps, num_maps),
            ConvBlock(num_maps, num_maps, kernel_size=1, padding=0, relu=False),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        global_attn = self.global_attn(x)
        local_attn = self.local_attn(x)

        attn = global_attn + local_attn
        attn = self.softmax(attn)

        return (x*attn).sum(dim=1, keepdim=True)

class DGNet3(nn.Module):
    def __init__(self, pretrained=True, num_samples=3, max_num_styles=512):
        super().__init__()
        self.gen = Generator()
        self.gen_cyc = Generator()
        self.reg = DensityRegressor(pretrained=pretrained)
        self.final = nn.Sequential(
            ConvBlock(1+num_samples, 1+num_samples),
            ConvBlock(1+num_samples, 1+num_samples),
            ConvBlock(1+num_samples, 1, kernel_size=1, padding=0)
        )

        self.register_buffer('style_queue', torch.zeros(max_num_styles, 896))
        self.max_num_styles = max_num_styles
        self.num_styles = 0

        self.num_samples = num_samples

    def _add_to_queue(self, batch):
        b = batch.size(0)
        if self.num_styles + b <= self.max_num_styles:
            self.style_queue[self.num_styles:self.num_styles+b] = batch.detach()
            self.num_styles += b
        else:
            r = self.num_styles + b - self.max_num_styles
            self.style_queue[:self.max_num_styles-b] = self.style_queue[r:r+self.max_num_styles-b].clone()
            self.style_queue[self.max_num_styles-b:] = batch.detach()
            self.num_styles = self.max_num_styles

    def _get_queue(self):
        return self.style_queue[:self.num_styles]

    def _dissim_loss(self, x, y, thrs, amap=None):
        res = ((x-y)**2) * amap if amap is not None else (x-y)**2
        if len(x.size()) == 4:
            return -(res).mean([2,3]).clamp(max=thrs).mean()
        elif len(x.size()) == 3:
            return -(res).mean([2]).clamp(max=thrs).mean()
        elif len(x.size()) == 2:
            return -(res).clamp(max=thrs).mean()
        else:
            raise ValueError('Invalid input size')
        
    def _rbf_loss(self, x, y, gamma):
        # loss based on RBF kernel
        # x: N x C
        # y: M x C
        # gamma: float
        # return: N
        return torch.exp(-gamma * (x.unsqueeze(1)-y.unsqueeze(0)).pow(2).sum(2)).mean(1)

    def forward_gen(self, x, z=None):
        self.gen.requires_grad_(True)
        return self.gen(x, z)
    
    def forward_reg(self, x):
        self.reg.requires_grad_(True)
        return self.reg(x)[0]
    
    def forward_joint_gen(self, x, z, emap):
        self.gen.requires_grad_(True)
        self.gen.enc.requires_grad_(False)
        self.reg.requires_grad_(False)

        amap = 1 + emap.detach() / emap.detach().max()

        f, s = self.gen.encode(x)
        f_trans = self.gen.transfer(f, z)

        x_rec = self.gen.decode(f)
        x_gen = self.gen.decode(f_trans)

        f_gen, s_gen = self.gen.encode(x_gen)

        x_cat = torch.cat([x, x_gen], dim=0)
        y_cat, _, _ = self.reg(x_cat)

        y, y_gen = torch.chunk(y_cat, chunks=2, dim=0)

        if self.num_styles == 0:
            self._add_to_queue(s[0:1])
        else:
            cur_styles = self._get_queue()
            l_stys = self._rbf_loss(s, cur_styles, 0.1)
            s_best = s[l_stys.argmin()]
            # if s_best.size(0) > 0:
            self._add_to_queue(s_best.unsqueeze(0))
                # print(f'Added {s_best.size(0)} styles to queue')

        cur_styles = self._get_queue()
        l_gen_stys = self._rbf_loss(s_gen, cur_styles, 0.1)
        s_gen_best = s_gen[l_gen_stys.argmin()]
        # if s_gen_best.size(0) > 0:
        self._add_to_queue(s_gen_best.unsqueeze(0))
            # print(f'Added {s_gen_best.size(0)} styles to queue')
        
        l_sty = l_gen_stys.mean()
        l_rec = F.mse_loss(x_rec, x)
        l_ctt = F.mse_loss(f, f_gen)
        l_den = F.mse_loss(y, y_gen)

        print(f'l_sty: {l_sty:.3f}, l_rec: {l_rec:.3f}, l_ctt: {l_ctt:.3f}, l_den: {l_den:.3f}')

        return 1000 * l_sty + 10 * l_rec + 10 * l_ctt + 10 * l_den
    
    def forward_joint_reg(self, x, z):
        self.reg.requires_grad_(True)
        self.gen.requires_grad_(False)

        x_gen, _ = self.gen(x, z)

        x_cat = torch.cat([x, x_gen], dim=0)
        y_cat, _, _ = self.reg(x_cat)

        y, y_gen = torch.chunk(y_cat, chunks=2, dim=0)
        # f_inv, f_inv_gen = torch.chunk(f_inv_cat, chunks=2, dim=0)
        # f_var, f_var_gen = torch.chunk(f_var_cat, chunks=2, dim=0)

        # l_sim = F.mse_loss(f_inv, f_inv_gen)
        # l_var = self._dissim_loss(f_var, f_var_gen, 1)

        # print(f'l_var: {l_var:.3f}')
        l_den = F.mse_loss(y, y_gen)

        return y_cat, l_den
    
    def forward_joint(self, x, z1, z2):
        self.reg.requires_grad_(True)
        self.gen.requires_grad_(True)

        x_gen1, f_cot = self.gen(x, z1)
        x_gen2, _ = self.gen(x, z2)
        x_rec, _ = self.gen(x)
        f_cot_gen1 = self.gen.enc(x_gen1)
        f_cot_gen2 = self.gen.enc(x_gen2)

        x_cat = torch.cat([x, x_gen1, x_gen2], dim=0)
        y_cat, f_inv_cat, f_var_cat = self.reg(x_cat)

        f_inv, f_inv_gen1, f_inv_gen2 = torch.chunk(f_inv_cat, chunks=3, dim=0)
        f_var, f_var_gen1, f_var_gen2 = torch.chunk(f_var_cat, chunks=3, dim=0)

        l_div = -(x_gen1-x_gen2).abs().mean([1,2,3]).clamp(max=0.2).mean()
        l_rec = F.mse_loss(x_rec, x)
        l_cot = F.mse_loss(f_cot, f_cot_gen1) + F.mse_loss(f_cot, f_cot_gen2)
        l_var = -torch.clamp(F.mse_loss(f_var, f_var_gen1), max=1) - torch.clamp(F.mse_loss(f_var, f_var_gen2), max=1) - torch.clamp(F.mse_loss(f_var_gen1, f_var_gen2), max=1)
        l_sim = F.mse_loss(f_inv, f_inv_gen1) + F.mse_loss(f_inv, f_inv_gen2)

        print(f'l_div: {l_div:.3f}, l_rec: {l_rec:.3f}, l_cot: {l_cot:.3f}, l_var: {l_var:.3f}, l_sim: {l_sim:.3f}')

        return y_cat, 10 * l_div + 100 * l_rec + 10 * l_cot + 10 * l_var + 100 * l_sim
    
    def forward_final(self, x, zs):
        self.reg.requires_grad_(False)
        self.gen.requires_grad_(False)

        z_li = torch.chunk(zs, chunks=self.num_samples, dim=1)
        x_li = [self.gen(x, z)[0] for z in z_li] + [x]

        x_cat = torch.cat(x_li, dim=0)
        y_cat = self.reg(x_cat)[0]

        y_li = torch.chunk(y_cat, chunks=self.num_samples+1, dim=0)
        y_all = torch.cat(y_li, dim=1)
        y_final = self.final(y_all)

        return y_final
    
    def forward_afterjoint(self, x, zs):
        self.reg.requires_grad_(True)
        self.gen.requires_grad_(False)

        z_li = torch.chunk(zs, chunks=self.num_samples, dim=1)
        x_li = [self.gen(x, z)[0] for z in z_li] + [x]

        x_cat = torch.cat(x_li, dim=0)
        y_cat = self.reg(x_cat)[0]

        return y_cat
    
    def forward_test(self, x):
        zs = torch.randn(x.shape[0], self.num_samples * 64, device=x.device)
        z_li = torch.chunk(zs, chunks=self.num_samples, dim=1)
        x_li = [self.gen(x, z)[0] for z in z_li] + [x]

        y_li = [self.reg(xi)[0] for xi in x_li]

        y_all = torch.cat(y_li, dim=1)
        y_final = self.final(y_all)

        return y_final, x_li, y_li


        


