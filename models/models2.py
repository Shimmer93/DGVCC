import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt

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
        y = self.enc(x)
        y = self.dec(y)
        return y
    
class Generator0(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg_feats = list(vgg.features.children())
        self.enc1 = nn.Sequential(*vgg_feats[:9])
        self.enc2 = nn.Sequential(*vgg_feats[9:18])
        self.enc3 = nn.Sequential(*vgg_feats[18:26])

        self.dec3 = nn.Sequential(
            ConvBlock(512, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(256, 128, bn=True),
            ConvBlock(128, 64, bn=True)
        )

        self.head = nn.Sequential(
            ConvBlock(64, 64, bn=True),
            ConvBlock(64, 3, kernel_size=1, padding=0, relu=False),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec3(x3)
        x = upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        x = upsample(x)
        x = self.head(x)

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
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 1, kernel_size=1, padding=0)
        )

        self.cls_head = nn.Sequential(
            ConvBlock(512, 256),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x, c_gt=None):
        x1 = self.stage1(x)
        x1 = F.instance_norm(x1)
        x2 = self.stage2(x1)
        x2 = F.instance_norm(x2)
        x3 = self.stage3(x2)
        x3 = F.instance_norm(x3)

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
        if c_gt is not None:
            new_c = c_gt
        else:
            new_c = c.clone().detach()
            new_c[c<0.5] = 0
            new_c[c>=0.5] = 1
        resized_c = upsample(new_c, scale_factor=4, mode='nearest')
        d = self.den_head(y_cat)
        dc = d * resized_c
        dc = upsample(dc, scale_factor=4)

        return dc, d, c, x3
    
class DensityRegressorM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.thrs = 0.5
        self.part_num = 1024
        self.final_dim = 256
        variance = 1.0

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

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, self.final_dim, kernel_size=1, padding=0, bn=True)
        )

        # self.cls_dec = nn.Sequential(
        #     ConvBlock(512, 256),
        #     nn.Dropout2d(p=0.5)
        # )

        self.mem = nn.Parameter(torch.FloatTensor(1, self.final_dim, self.part_num).normal_(0.0, variance))
        # self.cls_mem = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))
        # self.mu_mem = nn.Parameter(torch.FloatTensor(1, 256, 64).normal_(0.0, variance))
        # self.sigma_mem = nn.Parameter(torch.FloatTensor(1, 256, 64).normal_(0.0, variance))

        self.den_head = ConvBlock(self.final_dim, 1, kernel_size=1, padding=0)

        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=0.5),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def compute_sim_loss(self, x, y):
        b, k, h, w = x.shape
        x_ = x.view(b, k, -1)
        y_ = y.view(b, k, -1)
        sim = torch.bmm(y_.transpose(1, 2), x_)
        sim_gt = torch.linspace(0, h * w - 1, h * w).unsqueeze(0).repeat(b, 1).to(x.device)
        loss = F.cross_entropy(sim, sim_gt.long())
        return loss

    def forward_mem(self, y, mem):
        b, k, h, w = y.shape
        m = mem.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        y_ = y.view(b, k, -1)
        logits = torch.bmm(m_key, y_) / sqrt(k)
        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        y_new_ = y_new.view(b, k, h, w)

        # calculate rec loss
        # recon_sim = torch.bmm(y_new.transpose(1, 2), y_)
        # sim_gt = torch.linspace(0, y.shape[2] * y.shape[3] - 1,
        #                         y.shape[2] * y.shape[3]).unsqueeze(0).repeat(y.shape[0], 1).to(y.device)
        # sim_loss = F.cross_entropy(recon_sim, sim_gt.long())

        return y_new_, logits

    def forward_fe(self, x):
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

        return y_cat, x3

    def forward(self, x, c_gt=None, raw=True):
        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        if raw:
            y_den_new, _ = self.forward_mem(y_den, self.mem)
        else:
            with torch.no_grad():
                y_den_new, _ = self.forward_mem(y_den, self.mem)

        # y_cls = self.cls_dec(x3)
        # y_cls_new, loss_cls_sim = self.forward_mem(y_cls, self.cls_mem)

        # c = self.cls_head(y_cls_new)
        c = self.cls_head(x3)
        if c_gt is not None:
            new_c = c_gt
        else:
            new_c = c.clone().detach()
            new_c[c<self.thrs] = 0
            new_c[c>=self.thrs] = 1
        resized_c = upsample(new_c, scale_factor=4, mode='nearest')
        d = self.den_head(y_den_new)
        dc = d * resized_c
        dc = upsample(dc, scale_factor=4)

        # return dc, (d, c, y_den_new, y_cls_new, loss_den_sim, loss_cls_sim)
        return dc, (d, c, y_den, y_den_new, x3)

    def forward_pair(self, img1, img2, c_gt=None):
        y_cat1, x3_1 = self.forward_fe(img1)
        y_cat2, x3_2 = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)
        e_y = torch.abs(y_in1 - y_in2)

        e_mask = (e_y < 0.2).detach()
        y_den_masked1 = F.dropout2d(y_den1, 0.5) * e_mask
        y_den_masked2 = F.dropout2d(y_den2, 0.5) * e_mask

        loss_err = F.l1_loss(y_in1, y_in2) # if (~e_mask).sum() > 0 else 0

        y_den_new1, logits1 = self.forward_mem(y_den_masked1, self.mem)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2, self.mem)

        # JSD
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        log_p1 = F.log_softmax(logits1, dim=1)
        log_p2 = F.log_softmax(logits2, dim=1)
        pm = (p1 + p2) / 2
        loss_kl = 0.5 / logits1.shape[2] * (F.kl_div(log_p1, pm, reduction='batchmean') + \
                  F.kl_div(log_p2, pm, reduction='batchmean'))


        # loss_kl = F.kl_div(log_p2, log_p1, reduction='batchmean', log_target=True) / logits1.shape[2]

        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)

        if c_gt is not None:
            c_new1 = c_gt
            c_new2 = c_gt
        else:
            c_new1 = c1.clone().detach()
            c_new1[c1<self.thrs] = 0
            c_new1[c1>=self.thrs] = 1
            c_new2 = c2.clone().detach()
            c_new2[c2<self.thrs] = 0
            c_new2[c2>=self.thrs] = 1
        c_resized1 = upsample(c_new1, scale_factor=4, mode='nearest')
        c_resized2 = upsample(c_new2, scale_factor=4, mode='nearest')

        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)
        dc1 = upsample(d1 * c_resized1, scale_factor=4)
        dc2 = upsample(d2 * c_resized2, scale_factor=4)

        loss_f_sim = -F.cosine_similarity(y_den1, y_den2, dim=1).mean()
        loss_fnew_sim = F.mse_loss(y_den_new1, y_den_new2) #-F.cosine_similarity(y_den_new1, y_den_new2, dim=1).mean()

        return dc1, dc2, c1, c2, loss_kl, loss_fnew_sim, loss_err

class DensityRegressorBase(nn.Module):
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

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            nn.Dropout2d(p=0.5)
        )

        self.den_head = ConvBlock(256, 1, kernel_size=1, padding=0)

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

        y_den = self.den_dec(y_cat)
        d = self.den_head(y_den)
        dc = upsample(d, scale_factor=4)

        return dc
    
class DensityRegressorBaseCls(nn.Module):
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

        self.den_dec = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            nn.Dropout2d(p=0.5)
        )

        self.cls_dec = nn.Sequential(
            ConvBlock(512, 256),
            nn.Dropout2d(p=0.5)
        )

        self.den_head = ConvBlock(256, 1, kernel_size=1, padding=0)

        self.cls_head = nn.Sequential(
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x, c_gt=None):
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

        y_den = self.den_dec(y_cat)
        y_cls = self.cls_dec(x3)
        c = self.cls_head(y_cls)
        if c_gt is not None:
            new_c = c_gt
        else:
            new_c = c.clone().detach()
            new_c[c<0.5] = 0
            new_c[c>=0.5] = 1
        resized_c = upsample(new_c, scale_factor=4, mode='nearest')
        d = self.den_head(y_den)
        dc = d * resized_c
        dc = upsample(dc, scale_factor=4)

        return dc, (d, c)

def get_models():
    gen = Generator()
    reg = DensityRegressorM()
    return gen, reg

def get_basemodel():
    return DensityRegressorBase()