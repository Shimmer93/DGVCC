import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import models
import yaml
import time
from rich.progress import track
import argparse
from math import sqrt
from enum import Enum
from PIL import Image
from glob import glob

from trainers.trainer import Trainer
from losses.bl import BL
from utils.misc import denormalize, divide_img_into_patches

class DGRegTrainer(Trainer):
    def __init__(self, seed, version, device, log_para):
        super().__init__(seed, version, device)

        self.log_para = log_para

    def compute_count_loss(self, loss: nn.Module, pred_dmaps, gt_datas):
        if loss.__class__.__name__ == 'MSELoss':
            _, gt_dmaps = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            loss_value = loss(pred_dmaps, gt_dmaps * self.log_para)

        elif loss.__class__.__name__ == 'BL':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            targs = [targ.to(self.device) for targ in targs]
            st_sizes = st_sizes.to(self.device)
            loss_value = loss(gts, st_sizes, targs, pred_dmaps)

        else:
            raise ValueError('Unknown loss: {}'.format(loss))
        
        return loss_value
    
    def predict(self, model, img):
        h, w = img.shape[2:]
        patch_size = 1440
        if h >= patch_size or w >= patch_size:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, patch_size)
            for patch in img_patches:
                pred = model(patch, True)
                pred_count += torch.sum(pred).cpu().item() / self.log_para
        else:
            pred_dmap = model(img, True)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para

        return pred_count

    def train_step(self, model, loss, optimizer, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            optimizer.zero_grad()
            dmaps, dmaps_trans = model(imgs)
            loss_den = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_den_trans = self.compute_count_loss(loss, dmaps_trans, gt_datas)
            loss_sim = F.mse_loss(dmaps, dmaps_trans)
            loss_total = loss_den + loss_den_trans + loss_sim
            loss_total.backward()
            optimizer.step()

        return loss_total.item()

    def val_step(self, model, batch):
        img, gt, _, _ = batch
        img = img.to(self.device)
        pred_count = self.predict(model, img)
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        return mae

    def test_step(self, model, batch):
        img, gt, _, _ = batch
        img = img.to(self.device)
        pred_count = self.predict(model, img)
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        return {'mae': mae, 'mse': mse}

    def vis_step(self, model, batch):
        img, gt, name, _ = batch
        vis_dir = os.path.join(self.log_dir, 'vis')

        img = img.to(self.device)
        pred_dmaps = [model(img, True), model(img, True), model(img, True)]
        pred_count = [d.sum().cpu().item() / self.log_para for d in pred_dmaps]
        gt_count = gt.shape[1]
        img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
        pred_dmap = [d[0,0].detach().cpu().numpy() for d in pred_dmaps]

        fig = plt.figure(figsize=(20, 10))
        ax_img = fig.add_subplot(1, 4, 1)
        ax_img.set_title('GT: {}'.format(gt_count))
        ax_img.imshow(img)
        for i in range(3):
            ax_den = fig.add_subplot(1, 4, i+2)
            ax_den.set_title('Pred: {}'.format(pred_count[i]))
            ax_den.imshow(pred_dmap[i])

        plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0])))
        plt.close()