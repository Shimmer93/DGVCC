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

class DGTrainer(Trainer):
    def __init__(self, seed, version, device, mode, log_para):
        super().__init__(seed, version, device)

        self.mode = mode
        self.log_para = log_para

    def load_ckpt(self, model, path):
        if self.mode == 'regression':
            super().load_ckpt(model.reg, path)
        elif self.mode == 'generation':
            super().load_ckpt(model.gen, path)
        elif self.mode == 'joint':
            super().load_ckpt(model, path)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))
        
    def save_ckpt(self, model, path):
        if self.mode == 'regression':
            super().save_ckpt(model.reg, path)
        elif self.mode == 'generation':
            super().save_ckpt(model.gen, path)
        elif self.mode == 'joint':
            super().save_ckpt(model, path)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

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
    
    def compute_count_loss_with_aug(self, loss, pred_dmaps, gt_datas, num_aug_samples=3):
        num_dmaps = 1 + num_aug_samples
        if loss.__class__.__name__ == 'MSELoss':
            if len(gt_datas) == 2:
                gt_dmaps = gt_datas[1]
            else:
                gt_dmaps = gt_datas[2]
            gt_dmaps = gt_dmaps.to(self.device)
            gt_dmaps = gt_dmaps.repeat(num_dmaps, 1, 1, 1)
            loss_value = loss(pred_dmaps, gt_dmaps * self.log_para)

        elif loss.__class__.__name__ == 'BL':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            gts = gts * num_dmaps
            targs = [targ.to(self.device) for targ in targs]
            targs = targs * num_dmaps
            st_sizes = st_sizes.repeat(num_dmaps)
            loss_value = loss(gts, st_sizes, targs, pred_dmaps)

        else:
            raise ValueError('Unknown loss: {}'.format(loss))
        
        return loss_value

    def train_step(self, model, loss, optimizer, batch):
        imgs, gt_datas = batch

        optimizer.zero_grad()

        if self.mode == 'regression':
            imgs = imgs.to(self.device)
            pred_dmaps = model.forward_reg(imgs)
            loss_value = self.compute_count_loss(loss, pred_dmaps, gt_datas)
        elif self.mode == 'generation':
            imgs_cot, imgs_sty = imgs
            imgs_cot = imgs_cot.to(self.device)
            imgs_sty = imgs_sty.to(self.device)
            imgs_new, loss_value = model.forward_gen(imgs_cot, imgs_sty)
        elif self.mode == 'joint':
            imgs_cot, imgs_sty = imgs
            imgs_cot = imgs_cot.to(self.device)
            imgs_sty = imgs_sty.to(self.device)
            pred_dmaps, loss_gen = model(imgs_cot, imgs_sty)
            loss_reg = self.compute_count_loss_with_aug(loss, pred_dmaps, gt_datas, 2)
            loss_value = loss_reg + loss_gen
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
        loss_value.backward()
        optimizer.step()

        return loss_value.item()
    
    def val_step(self, model, batch):
        img, gt, _, _ = batch

        if self.mode == 'regression':
            img = img.to(self.device)
            pred_dmap = model.forward_reg(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            return mae
        elif self.mode == 'generation':
            img_cot, img_sty = img
            img_cot = img_cot.to(self.device)
            img_sty = img_sty.to(self.device)
            _, loss = model.forward_gen(img_cot, img_sty)
            loss = loss.cpu().item()
            return loss
        elif self.mode == 'joint':
            img_cot, img_sty = img
            img_cot = img_cot.to(self.device)
            pred_dmap = model.forward_reg(img_cot)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt[0].shape[1]
            mae = np.abs(pred_count - gt_count)
            return mae
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
    def test_step(self, model, batch):
        img, gt, _, _ = batch

        if self.mode == 'regression':
            img = img.to(self.device)
            h, w = img.shape[2:]
            patch_size = 1440
            if h >= patch_size or w >= patch_size:
                pred_count = 0
                img_patches, _, _ = divide_img_into_patches(img, patch_size)
                for patch in img_patches:
                    pred = model.forward_reg(patch)
                    pred_count += torch.sum(pred).cpu().item() / self.log_para
            else:
                pred_dmap = model.forward_reg(img)
                pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        elif self.mode == 'generation':
            img_cot, img_sty = img
            img_cot = img_cot.to(self.device)
            img_sty = img_sty.to(self.device)
            _, loss = model.forward_gen(img_cot, img_sty)
            loss = loss.cpu().item()
            return {'loss': loss}
        elif self.mode == 'joint':
            img_cot, img_sty = img
            img_cot = img_cot.to(self.device)
            img_sty = img_sty.to(self.device)
            h, w = img_cot.shape[2:]
            patch_size = 1440
            if h >= patch_size or w >= patch_size:
                pred_count = 0
                img_patches, _, _ = divide_img_into_patches(img_cot, patch_size)
                for patch in img_patches:
                    pred = model.forward_reg(patch)
                    pred_count += torch.sum(pred).cpu().item() / self.log_para
            else:
                pred_dmap = model.forward_reg(img_cot)
                pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt[0].shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
    def vis_step(self, model, batch):
        img, gt, name, _ = batch

        vis_dir = os.path.join(self.log_dir, 'vis')

        if self.mode == 'regression':
            img = img.to(self.device)
            pred_dmap = model.forward_reg(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_dmap = pred_dmap[0,0].detach().cpu().numpy()

            fig = plt.figure(figsize=(20, 10))
            ax_img = fig.add_subplot(1, 2, 1)
            ax_img.set_title('GT: {}'.format(gt_count))
            ax_img.imshow(img)
            ax_den = fig.add_subplot(1, 2, 2)
            ax_den.set_title('Pred: {}'.format(pred_count))
            ax_den.imshow(pred_dmap)

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0])))
            plt.close()

        elif self.mode == 'generation':
            img_cot, img_sty = img
            img_cot = img_cot.to(self.device)
            img_sty = img_sty.to(self.device)
            img_new, _ = model.forward_gen(img_cot, img_sty)
            img_cot = denormalize(img_cot.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_sty = denormalize(img_sty.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_new = denormalize(img_new.detach())[0].cpu().permute(1, 2, 0).numpy()

            fig = plt.figure(figsize=(20, 10))
            ax_cot = fig.add_subplot(1, 3, 1)
            ax_cot.set_title('Content')
            ax_cot.imshow(img_cot)
            ax_sty = fig.add_subplot(1, 3, 2)
            ax_sty.set_title('Style')
            ax_sty.imshow(img_sty)
            ax_new = fig.add_subplot(1, 3, 3)
            ax_new.set_title('New')
            ax_new.imshow(img_new)

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0][0])))

        elif self.mode == 'joint':
            img_cot, img_sty = img
            img_cot = img_cot.to(self.device)
            img_sty = img_sty.to(self.device)
            img_new, img_nov, _ = model.forward_gen(img_cot, img_sty)
            dmap_cot = model.forward_reg(img_cot)
            dmap_new = model.forward_reg(img_new)
            dmap_nov = model.forward_reg(img_nov)
            pred_count_cot = dmap_cot.sum().cpu().item() / self.log_para
            pred_count_new = dmap_new.sum().cpu().item() / self.log_para
            pred_count_nov = dmap_nov.sum().cpu().item() / self.log_para
            gt_count = gt[0].shape[1]

            img_cot = denormalize(img_cot.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_sty = denormalize(img_sty.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_new = denormalize(img_new.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_nov = denormalize(img_nov.detach())[0].cpu().permute(1, 2, 0).numpy()
            dmap_cot = dmap_cot[0,0].detach().cpu().numpy()
            dmap_new = dmap_new[0,0].detach().cpu().numpy()
            dmap_nov = dmap_nov[0,0].detach().cpu().numpy()

            data = [img_cot, img_new, img_nov, img_sty, dmap_cot, dmap_new, dmap_nov]
            label = [f'Content: {gt_count}', 'New', 'novel', 'Style', f'Pred_cotent: {pred_count_cot:.4f}', f'Pred_new: {pred_count_new:.4f}', f'Pred_noval: {pred_count_nov:.4f}']
            fig = plt.figure(figsize=(20, 10))
            for i in range(7):
                ax = fig.add_subplot(2, 4, i+1)
                ax.set_title(label[i])
                ax.imshow(data[i])

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0][0])))

        else:
            raise NotImplementedError

    