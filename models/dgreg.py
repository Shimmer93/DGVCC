import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from enum import Enum
from losses.triplet import triplet_loss
from queue import Queue
from random import randint

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

class FeatureMemory(nn.Module):
    def __init__(self, dim, mem_size):
        super().__init__()
        self.dim = dim
        self.mem_size = mem_size
        self.memory = nn.Parameter(torch.randn(1, dim, mem_size))

    def forward(self, x):
        b, c, h, w = x.shape
        m = self.memory.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        x = x.view(b, c, -1)
        logits = torch.bmm(m_key, x)
        x = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        x = x.view(b, c, h, w)
        return x

class MemoryQueue(nn.Module):
    def __init__(self, dim, mem_size):
        super().__init__()
        self.register_buffer('q', torch.zeros(mem_size, dim))
        self.num_items = 512
        self.max_size = mem_size

    def add(self, x):
        b = x.size(0)
        if self.num_items + b <= self.max_size:
            self.q[self.num_items:self.num_items+b] = x.clone().detach()
            self.num_items += b
        else:
            r = self.num_items + b - self.max_size
            self.q[:self.max_size-b] = self.q[r:r+self.max_size-b].clone()
            self.q[self.max_size-b:] = x.clone().detach()
            self.num_items = self.max_size

    def get_all(self):
        return self.q[:self.num_items]
    
    def is_empty(self):
        return self.num_items == 0
    
    def size(self):
        return self.num_items

class StyleEncodingModule(nn.Module):
    def __init__(self, encoder, mem_dim, mem_size):
        super().__init__()
        self.encoder = encoder
        self.memory = FeatureMemory(mem_dim, mem_size)

    def get_style(self, x):
        mu = torch.mean(x, dim=(2, 3))
        sigma = torch.std(x, dim=(2, 3))
        return mu.clone(), sigma.clone()

    def forward(self, x, s_new=None):
        f = self.encoder(x)
        mu_orig, sigma_orig = self.get_style(f)
        if s_new is not None:
            mu_new, sigma_new = s_new
            # print(mu_new.shape, sigma_new.shape, mu_orig.shape, sigma_orig.shape)
            f = (f - mu_orig.unsqueeze(-1).unsqueeze(-1)) / (sigma_orig.unsqueeze(-1).unsqueeze(-1) + 1e-5) * sigma_new.unsqueeze(-1).unsqueeze(-1) + mu_new.unsqueeze(-1).unsqueeze(-1)
        y = self.memory(f)
        return y, mu_orig, sigma_orig

def rbf_loss(x, y, gamma=1.0):
    # loss based on RBF kernel
    # x: N x C
    # y: M x C
    # gamma: float
    # return: N
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    dist = torch.sum((x - y) ** 2, dim=2)
    loss = torch.mean(torch.exp(-gamma * dist), dim=1)
    return loss

class DensityRegressor(nn.Module):
    def __init__(self, pretrained=True, feat_mem_size=512, sty_mem_size=512, rbf_gamma=1.0, rbf_thrs=0.1, dropout=0.5):
        super().__init__()
        self.feat_mem_size = feat_mem_size
        self.sty_mem_size = sty_mem_size
        self.rbf_gamma = rbf_gamma
        self.rbf_thrs = rbf_thrs
        self.dropout = dropout

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        vgg_feat_list = list(vgg.features.children())

        self.stage1 = StyleEncodingModule(nn.Sequential(*vgg_feat_list[:23]), 256, feat_mem_size)
        self.stage2 = StyleEncodingModule(nn.Sequential(*vgg_feat_list[23:33]), 512, feat_mem_size)
        self.stage3 = StyleEncodingModule(nn.Sequential(*vgg_feat_list[33:43]), 512, feat_mem_size)

        self.q1_mu = MemoryQueue(256, sty_mem_size)
        self.q1_sigma = MemoryQueue(256, sty_mem_size)
        self.q2_mu = MemoryQueue(512, sty_mem_size)
        self.q2_sigma = MemoryQueue(512, sty_mem_size)

        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True),
            FeatureMemory(512, feat_mem_size)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256, bn=True),
            FeatureMemory(256, feat_mem_size)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True),
            FeatureMemory(128, feat_mem_size)
        )

        self.to_inv = nn.Sequential(
            ConvBlock(512+256+128, 256, kernel_size=1, padding=0),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            FeatureMemory(256, feat_mem_size)
        )

        self.final_layer = nn.Sequential(
            nn.Dropout2d(p=dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0)
        )

    def select_style(self, s_cur, s_new):
        # print(s_cur.shape, s_new.shape)
        l_rbf = rbf_loss(s_new, s_cur, gamma=self.rbf_gamma)
        # print(l_rbf)
        s_nov = s_new[l_rbf.argmin(),:].unsqueeze(0)
        return s_nov
    
    def update_queues(self, mu1, sigma1, mu2, sigma2):
        if self.q1_mu.is_empty():
            self.q1_mu.add(mu1[0:1])
            self.q1_sigma.add(sigma1[0:1])
            self.q2_mu.add(mu2[0:1])
            self.q2_sigma.add(sigma2[0:1])

        mu1_cur = self.q1_mu.get_all()
        sigma1_cur = self.q1_sigma.get_all()
        mu2_cur = self.q2_mu.get_all()
        sigma2_cur = self.q2_sigma.get_all()

        mu1_good = self.select_style(mu1_cur, mu1)
        sigma1_good = self.select_style(sigma1_cur, sigma1)
        mu2_good = self.select_style(mu2_cur, mu2)
        sigma2_good = self.select_style(sigma2_cur, sigma2)

        print(f'Added good styles: {mu1_good.shape[0]}, {sigma1_good.shape[0]}, {mu2_good.shape[0]}, {sigma2_good.shape[0]}')

        if mu1_good.shape[0] > 0:
            self.q1_mu.add(mu1_good)
        if sigma1_good.shape[0] > 0:
            self.q1_sigma.add(sigma1_good)
        if mu2_good.shape[0] > 0:
            self.q2_mu.add(mu2_good)
        if sigma2_good.shape[0] > 0:
            self.q2_sigma.add(sigma2_good)

        # mu1_rand = mu1 + torch.randn_like(mu1) * sigma1
        # sigma1_rand = sigma1 + torch.randn_like(sigma1) * sigma1
        # mu2_rand = mu2 + torch.randn_like(mu2) * sigma2
        # sigma2_rand = sigma2 + torch.randn_like(sigma2) * sigma2

        # mu1_nov = self.select_style(mu1_cur, mu1_rand)
        # sigma1_nov = self.select_style(sigma1_cur, sigma1_rand)
        # mu2_nov = self.select_style(mu2_cur, mu2_rand)
        # sigma2_nov = self.select_style(sigma2_cur, sigma2_rand)

        # print(f'Added novel styles: {mu1_nov.shape[0]}, {sigma1_nov.shape[0]}, {mu2_nov.shape[0]}, {sigma2_nov.shape[0]}')

        # if mu1_nov.shape[0] > 0:
        #     self.q1_mu.add(mu1_nov)
        # if sigma1_nov.shape[0] > 0:
        #     self.q1_sigma.add(sigma1_nov)
        # if mu2_nov.shape[0] > 0:
        #     self.q2_mu.add(mu2_nov)
        # if sigma2_nov.shape[0] > 0:
        #     self.q2_sigma.add(sigma2_nov)

    def encode(self, x, transfer=False):
        if transfer:
            mu1_rand = self.q1_mu.get_all()[randint(0, self.q1_mu.size()-1)].unsqueeze(0).clone()
            sigma1_rand = self.q1_sigma.get_all()[randint(0, self.q1_sigma.size()-1)].unsqueeze(0).clone()
            mu2_rand = self.q2_mu.get_all()[randint(0, self.q2_mu.size()-1)].unsqueeze(0).clone()
            sigma2_rand = self.q2_sigma.get_all()[randint(0, self.q2_sigma.size()-1)].unsqueeze(0).clone()
            x1, mu1, sigma1 = self.stage1(x, (mu1_rand, sigma1_rand))
            x2, mu2, sigma2 = self.stage2(x1, (mu2_rand, sigma2_rand))
            x3, _, _ = self.stage3(x2)
        else:
            x1, mu1, sigma1 = self.stage1(x)
            x2, mu2, sigma2 = self.stage2(x1)
            x3, _, _ = self.stage3(x2)

        if self.training:
            self.update_queues(mu1, sigma1, mu2, sigma2)

        return x1, x2, x3
    
    def decode(self, x1, x2, x3):
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

        return y

    def forward(self, x, transfer=False):
        if self.training:
           
            x1, x2, x3 = self.encode(x)
            x1_t, x2_t, x3_t = self.encode(x.clone(), transfer=True)
            x1_cat = torch.cat([x1, x1_t], dim=0)
            x2_cat = torch.cat([x2, x2_t], dim=0)
            x3_cat = torch.cat([x3, x3_t], dim=0)
            y_cat = self.decode(x1_cat, x2_cat, x3_cat)
            y, y_t = torch.chunk(y_cat, 2, dim=0)

            return y, y_t
        else:
            x1, x2, x3 = self.encode(x, transfer=transfer)
            y = self.decode(x1, x2, x3)
            return y