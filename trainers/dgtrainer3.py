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

class DGTrainer3(Trainer):
    def __init__(self, seed, version, device, mode, log_para):
        super().__init__(seed, version, device)

        self.mode = mode
        self.log_para = log_para

    def load_ckpt(self, model, path):
        if path is None:
            return
        self.log('Load checkpoint from {}'.format(path))
        if self.mode == 'regression':
            super().load_ckpt(model.reg, path)
        elif self.mode == 'generation':
            super().load_ckpt(model.gen, path)
        elif self.mode == 'joint' or self.mode == 'final':
            super().load_ckpt(model, path)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))
        
    def save_ckpt(self, model, path):
        if self.mode == 'regression':
            super().save_ckpt(model.reg, path)
        elif self.mode == 'generation':
            super().save_ckpt(model.gen, path)
        elif self.mode == 'joint' or self.mode == 'final':
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

        if self.mode == 'regression':
            optimizer.zero_grad()
            imgs = imgs.to(self.device)
            pred_dmaps = model.forward_reg(imgs)
            loss_value = self.compute_count_loss(loss, pred_dmaps, gt_datas)
            loss_value.backward()
            optimizer.step()
        elif self.mode == 'generation':
            optimizer.zero_grad()
            imgs_cot, imgs_sty = imgs
            imgs_cot = imgs_cot.to(self.device)
            imgs_sty = imgs_sty.to(self.device)
            imgs_new, loss_value = model.forward_gen(imgs_cot, imgs_sty)
            loss_value.backward()
            optimizer.step()
        elif self.mode == 'joint':
            optimizer.zero_grad()
            imgs = imgs.to(self.device)
            z1 = torch.randn(imgs.shape[0], 64).to(self.device)
            z2 = torch.randn(imgs.shape[0], 64).to(self.device)
            den_cat, loss_add = model.forward_joint(imgs, z1, z2)
            loss_count = self.compute_count_loss_with_aug(loss, den_cat, gt_datas, 2)
            loss_value = loss_count + loss_add
            loss_value.backward()
            optimizer.step()
        elif self.mode == 'final':
            optimizer.zero_grad()
            imgs = imgs.to(self.device)
            den = model.forward_final(imgs)
            loss_value = self.compute_count_loss(loss, den, gt_datas)
            loss_value.backward()
            optimizer.step()            
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
        return loss_value.item()
    
    def val_step(self, model, batch):
        img, gt, _, _ = batch

        if self.mode == 'regression' or self.mode == 'joint':
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
        elif self.mode == 'final':
            img = img.to(self.device)
            h, w = img.shape[2:]
            patch_size = 1440
            if h >= patch_size or w >= patch_size:
                pred_count = 0
                img_patches, _, _ = divide_img_into_patches(img, patch_size)
                for patch in img_patches:
                    pred, _, _ = model.forward_test(patch)
                    pred_count += torch.sum(pred).cpu().item() / self.log_para
            else:
                pred_dmap, _, _ = model.forward_test(img)
                pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            return mae
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
    def test_step(self, model, batch):
        img, gt, _, _ = batch

        if self.mode == 'regression' or self.mode == 'joint':
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
        elif self.mode == 'final':
            img = img.to(self.device)
            h, w = img.shape[2:]
            patch_size = 1440
            if h >= patch_size or w >= patch_size:
                pred_count = 0
                img_patches, _, _ = divide_img_into_patches(img, patch_size)
                for patch in img_patches:
                    pred, _, _ = model.forward_test(patch)
                    pred_count += torch.sum(pred).cpu().item() / self.log_para
            else:
                pred_dmap, _, _ = model.forward_test(img)
                pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
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
            plt.close()

        elif self.mode == 'joint' or self.mode == 'final':
            img = img.to(self.device)
            dmap, img_list, dmap_list = model.forward_test(img)

            img_list.insert(0, img)
            dmap_list.insert(0, dmap)
            img_list = [denormalize(i.detach())[0].cpu().permute(1, 2, 0).numpy() for i in img_list]
            dmap_list = [d[0,0].detach().cpu().numpy() for d in dmap_list]
            pred_count_list = [d.sum() / self.log_para for d in dmap_list]
            gt_count = gt.shape[1]

            img_label_list = [f'GT: {gt_count}'] + [f'Img {i+1}' for i in range(len(img_list)-1)]
            dmap_label_list = [f'Pred: {pred_count_list[0]:.2f}'] + [f'Dmap {i+1}: {pred_count_list[i+1]:.2f}' for i in range(len(dmap_list)-1)]

            fig = plt.figure(figsize=(5*len(img_list), 10))
            for i in range(len(img_list)):
                ax_img = fig.add_subplot(2, len(img_list), i+1)
                ax_img.set_title(img_label_list[i])
                ax_img.imshow(img_list[i])
                ax_den = fig.add_subplot(2, len(img_list), i+1+len(img_list))
                ax_den.set_title(dmap_label_list[i])
                ax_den.imshow(dmap_list[i])

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0])))
            plt.close()

        else:
            raise NotImplementedError

    