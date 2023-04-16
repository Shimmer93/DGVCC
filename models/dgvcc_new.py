import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from enum import Enum
from losses.triplet import triplet_loss

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

def adaptive_instance_normalization(x, y, eps=1e-5):
    """
    Adaptive Instance Normalization. Perform neural style transfer given content image x
    and style image y.
    Args:
        x (torch.FloatTensor): Content image tensor
        y (torch.FloatTensor): Style image tensor
        eps (float, default=1e-5): Small value to avoid zero division
    Return:
        output (torch.FloatTensor): AdaIN style transferred output
    """ 
    mu_x = torch.mean(x, dim=[2, 3])
    mu_y = torch.mean(y, dim=[2, 3])
    mu_x = mu_x.unsqueeze(-1).unsqueeze(-1)
    mu_y = mu_y.unsqueeze(-1).unsqueeze(-1) 
    sigma_x = torch.std(x, dim=[2, 3])
    sigma_y = torch.std(y, dim=[2, 3])
    sigma_x = sigma_x.unsqueeze(-1).unsqueeze(-1) + eps
    sigma_y = sigma_y.unsqueeze(-1).unsqueeze(-1) + eps
        
    info_x = torch.cat([mu_x, sigma_x], dim=1).squeeze(-1).squeeze(-1)
    info_y = torch.cat([mu_y, sigma_y], dim=1).squeeze(-1).squeeze(-1)

    return (x - mu_x) / sigma_x * sigma_y  + mu_y, info_x, info_y

class NovelStyleModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc_x = nn.Linear(dim, dim)
        self.fc_y = nn.Linear(dim, dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc_final = nn.Linear(dim, dim)

    def forward(self, x, y):
        x = self.fc_x(x)
        x = self.relu1(x)
        y = self.fc_y(y)
        y = self.relu2(y)
        out = self.fc_final(x + y)

        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg_feats = list(vgg.features.children())
        # self.enc = nn.Sequential(*vgg_feats[:41])
        self.enc = nn.Sequential(*vgg_feats[:26])
        self.enc.requires_grad_(False)
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

    def _style_loss(self, x, y):
        return F.mse_loss(torch.mean(x, dim=[2, 3]), torch.mean(y, dim=[2, 3])) + \
            F.mse_loss(torch.std(x, dim=[2, 3]), torch.std(y, dim=[2, 3]))

    def forward_rec(self, x):
        feats = self.enc(x)
        out = self.dec(feats)
        return out

    def forward(self, cot, sty):
        cot_feats = self.enc(cot)
        sty_feats = self.enc(sty)
        trans_feats = adaptive_instance_normalization(cot_feats, sty_feats)[0]
        out = self.dec(trans_feats)

        sty_f1 = self.enc[:4](sty)
        out_f1 = self.enc[:4](out)

        sty_f2 = self.enc[4:9](sty_f1)
        out_f2 = self.enc[4:9](out_f1)

        sty_f3 = self.enc[9:18](sty_f2)
        out_f3 = self.enc[9:18](out_f2)

        out_feats = self.enc[18:](out_f3)

        loss_cot = F.mse_loss(cot_feats, out_feats)
        loss_sty = self._style_loss(sty_f1, out_f1) + \
            self._style_loss(sty_f2, out_f2) + \
            self._style_loss(sty_f3, out_f3)
        
        return out, loss_cot, loss_sty

class Generator2(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg_feats = list(vgg.features.children())
        # self.enc = nn.Sequential(*vgg_feats[:41])
        self.enc = nn.Sequential(*vgg_feats[:26])
        self.enc.requires_grad_(False)

        self.style_module = NovelStyleModule(1024)

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

    def _style_loss(self, x, y):
        return F.mse_loss(torch.mean(x, dim=[2, 3]), torch.mean(y, dim=[2, 3])) + \
            F.mse_loss(torch.std(x, dim=[2, 3]), torch.std(y, dim=[2, 3]))
    
    def _style_triplet_loss(self, a, p, n):
        return triplet_loss(a.mean(dim=[2, 3]), p.mean(dim=[2, 3]), n.mean(dim=[2, 3]), margin=0.5) + \
            triplet_loss(a.std(dim=[2, 3]), p.std(dim=[2, 3]), n.std(dim=[2, 3]), margin=0.5)

    def forward_rec(self, x):
        feats = self.enc(x)
        out = self.dec(feats)
        return out
    
    def _transfer_with_infos(self, x, x_infos, y_infos):
        x_mus, x_sigmas = torch.chunk(x_infos, 2, dim=1)
        y_mus, y_sigmas = torch.chunk(y_infos, 2, dim=1)
        x_mus = x_mus.unsqueeze(2).unsqueeze(3)
        x_sigmas = x_sigmas.unsqueeze(2).unsqueeze(3)
        y_mus = y_mus.unsqueeze(2).unsqueeze(3)
        y_sigmas = y_sigmas.unsqueeze(2).unsqueeze(3)
        out = y_sigmas * (x - x_mus) / x_sigmas + y_mus

        return out

    def forward(self, cot, sty):
        cot_feats = self.enc(cot)
        sty_feats = self.enc(sty)
        trans_feats, cot_infos, sty_infos = adaptive_instance_normalization(cot_feats, sty_feats)
        out = self.dec(trans_feats)

        novel_infos = self.style_module(cot_infos, sty_infos)
        
        novel_trans_feats = self._transfer_with_infos(cot_feats, cot_infos, novel_infos)
        novel_out = self.dec(novel_trans_feats)

        sty_f1 = self.enc[:4](sty)
        out_f1 = self.enc[:4](out)
        novel_f1 = self.enc[:4](novel_out)

        sty_f2 = self.enc[4:9](sty_f1)
        out_f2 = self.enc[4:9](out_f1)
        novel_f2 = self.enc[4:9](novel_f1)

        sty_f3 = self.enc[9:18](sty_f2)
        out_f3 = self.enc[9:18](out_f2)
        novel_f3 = self.enc[9:18](novel_f2)

        out_feats = self.enc[18:](out_f3)
        novel_feats = self.enc[18:](novel_f3)

        loss_cot = F.mse_loss(cot_feats, out_feats) + F.mse_loss(cot_feats, novel_feats)
        # loss_sty = self._style_triplet_loss(sty_f1, out_f1, novel_f1) + \
        #     self._style_triplet_loss(sty_f2, out_f2, novel_f2) + \
        #     self._style_triplet_loss(sty_f3, out_f3, novel_f3)
        loss_sty = self._style_loss(sty_f1, out_f1) + self._style_loss(sty_f2, out_f2) + self._style_loss(sty_f3, out_f3) - \
            torch.clamp(self._style_loss(sty_f1, novel_f1) + self._style_loss(sty_f2, novel_f2) + self._style_loss(sty_f3, novel_f3), max=10)
        
        return out, novel_out, loss_cot, loss_sty

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
            ConvBlock(256, 256),
        )

        self.final_layer = nn.Sequential(
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        y_feats = [x1, x2, x3]

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

        return y, y_feats
    
class DGNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.gen = Generator2()
        self.gen_cyc = Generator2()
        self.reg = DensityRegressor(pretrained=pretrained)

    def forward_gen(self, cot, sty):
        trans_img, noval_img, loss_cot, loss_sty = self.gen(cot, sty)
        # print(f'loss_cot: {loss_cot:.4f}, loss_sty: {loss_sty:.4f}')
        return trans_img, noval_img, 10 * loss_cot + loss_sty
    
    def forward_reg(self, x):
        return self.reg(x)[0]
    
    def forward(self, x_cot, x_sty):
        b = x_cot.shape[0]

        x_gen, x_new, loss_cot, loss_sty = self.gen(x_cot, x_sty)
        x_rec_gen = self.gen_cyc.forward_rec(x_gen)
        x_rec_new = self.gen_cyc.forward_rec(x_new)
        loss_rec = F.mse_loss(x_rec_gen, x_cot) + F.mse_loss(x_rec_new, x_cot)


        x_cat = torch.cat([x_cot, x_gen, x_new], dim=0)
        d_cat, d_feats_cat = self.reg(x_cat)

        # print(f'loss_cot: {loss_cot:.4f}, loss_sty: {loss_sty:.4f}, loss_rec: {loss_rec:.4f}')

        return d_cat, 10 * loss_cot + 10 * loss_sty + 100 * loss_rec

        