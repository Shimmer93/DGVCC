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

class DGClsTrainer(Trainer):
    def __init__(self, seed, version, device, log_para, mode):
        super().__init__(seed, version, device)

        self.log_para = log_para
        self.mode = mode

    def compute_count_loss(self, loss: nn.Module, pred_dmaps, gt_datas):
        if loss.__class__.__name__ == 'MSELoss':
            _, gt_dmaps, _ = gt_datas
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
                pred, _ = model.reg(patch)
                pred_count += torch.sum(pred).cpu().item() / self.log_para
        else:
            pred_dmap, _ = model.reg(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para

        return pred_count

    def train_step(self, model, loss, optimizer, batch, epoch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)

        if self.mode == 'regression':
            optimizer.zero_grad()
            dmaps, bmaps = model.reg(imgs)
            loss_den = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_cls = F.binary_cross_entropy(bmaps, gt_datas[-1].to(self.device))
            loss_total = 10 * loss_cls
            loss_total.backward()
            optimizer.step()
        elif self.mode == 'generation':
            optimizer.zero_grad()
            imgs_rec = model.gen(imgs)
            loss_total = F.mse_loss(imgs_rec, imgs)
            loss_total.backward()
            optimizer.step()
        else:
            opt_gen, opt_reg = optimizer

            if epoch % 2 == 0:
                opt_gen.zero_grad()
                b, c, h, w = imgs.shape
                noise_input = torch.randn(b, 128, h//8, w//8).to(self.device)
                dmaps_noisy, bmaps_noisy, imgs_new = model.forward_gen(imgs, noise_input)
                loss_den = self.compute_count_loss(loss, dmaps_noisy, gt_datas)
                loss_cls = F.binary_cross_entropy(bmaps_noisy, 1-gt_datas[-1].to(self.device))
                # loss_n0 = torch.mean(torch.abs(noise0))
                loss_rec = F.mse_loss(imgs_new, imgs)
                print(f'loss_den: {loss_den.item():.4f}, loss_cls: {loss_cls.item():.4f}, loss_rec: {loss_rec.item():.4f}')
                loss_total = 10 * loss_cls
                loss_total.backward()
                opt_gen.step()
            else:
                opt_reg.zero_grad()
                b, c, h, w = imgs.shape
                noise_input = torch.randn(b, 128, h//8, w//8).to(self.device)
                dmaps, dmaps_noisy, bmaps, bmaps_noisy = model.forward_reg(imgs, noise_input)
                loss_den = self.compute_count_loss(loss, dmaps, gt_datas)
                loss_cls = F.binary_cross_entropy(bmaps, gt_datas[-1].to(self.device))
                loss_den_noisy = self.compute_count_loss(loss, dmaps_noisy, gt_datas)
                loss_cls_noisy = F.binary_cross_entropy(bmaps_noisy, gt_datas[-1].to(self.device))
                print(f'loss_den: {loss_den.item():.4f}, loss_cls: {loss_cls.item():.4f}, loss_den_noisy: {loss_den_noisy.item():.4f}, loss_cls_noisy: {loss_cls_noisy.item():.4f}')
                loss_total = loss_den + 10 * loss_cls + loss_den_noisy + 10 * loss_cls_noisy
                loss_total.backward()
                opt_reg.step()

            # loss_total = loss_total_gen + loss_total_reg

        return loss_total.item()

    def val_step(self, model, batch):
        if self.mode == 'regression':
            img, gt, _, _ = batch
            img = img.to(self.device)
            pred_count = self.predict(model, img)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            return mae
        elif self.mode == 'generation':
            img, gt, _, _ = batch
            img = img.to(self.device)
            img_rec = model.gen(img)
            loss = F.mse_loss(img_rec, img)
            return loss.detach().item()
        else:
            img, gt, _, _ = batch
            img = img.to(self.device)
            z = torch.randn(img.shape[0], 128, img.shape[2]//8, img.shape[3]//8).to(self.device)
            img_noisy = model.forward_test(img, z)
            pred_count = self.predict(model, img)
            noisy_count = self.predict(model, img_noisy)
            gt_count = gt.shape[1]
            mae = np.abs((pred_count+noisy_count)/2 - gt_count)
            return mae

    def test_step(self, model, batch):
        if self.mode == 'generation':
            img, gt, _, _ = batch
            img = img.to(self.device)
            img_rec = model.gen(img)
            loss = F.mse_loss(img_rec, img)
            return {'loss': loss.detach().item()}
        else:
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

        if self.mode == 'regression':

            pred_dmap, pred_bmap = model.reg(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_dmap = pred_dmap[0,0].detach().cpu().numpy() / self.log_para
            pred_bmap = pred_bmap[0,0].detach().cpu().numpy()

            fig = plt.figure(figsize=(20, 6))
            ax_img = fig.add_subplot(1, 3, 1)
            ax_img.set_title('GT: {}'.format(gt_count))
            ax_img.imshow(img)
            ax_den = fig.add_subplot(1, 3, 2)
            ax_den.set_title('Pred: {}'.format(pred_count))
            ax_den.imshow(pred_dmap)
            ax_cls = fig.add_subplot(1, 3, 3)
            ax_cls.set_title('Cls')
            ax_cls.imshow(pred_bmap)

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0])))
            plt.close()

        elif self.mode == 'generation':
            img_rec = model.gen(img)
            img_rec = denormalize(img_rec.detach())[0].cpu().permute(1, 2, 0).numpy()
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()

            fig = plt.figure(figsize=(12, 6))
            ax_img = fig.add_subplot(1, 2, 1)
            ax_img.set_title('GT')
            ax_img.imshow(img)
            ax_den = fig.add_subplot(1, 2, 2)
            ax_den.set_title('Rec')
            ax_den.imshow(img_rec)

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0])))
            plt.close()

        else:
            dmap, dmap_noisy, bmap, bmap_noisy, img_noisy = model.forward_vis(img)
            pred_count = dmap.sum().cpu().item() / self.log_para
            noisy_count = dmap_noisy.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_noisy = denormalize(img_noisy.detach())[0].cpu().permute(1, 2, 0).numpy()
            dmap = dmap[0,0].detach().cpu().numpy() / self.log_para
            dmap_noisy = dmap_noisy[0,0].detach().cpu().numpy() / self.log_para
            bmap = bmap[0,0].detach().cpu().numpy()
            bmap_noisy = bmap_noisy[0,0].detach().cpu().numpy()

            datas = [img, dmap, bmap, img_noisy, dmap_noisy, bmap_noisy]
            labels = ['GT: {}'.format(gt_count), 'Pred: {}'.format(pred_count), 'Cls', 'Noisy', 'Pred: {}'.format(noisy_count), 'Cls']

            fig = plt.figure(figsize=(15, 6))
            for i in range(6):
                ax = fig.add_subplot(2, 3, i+1)
                ax.set_title(labels[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, '{}.png'.format(name[0])))
            plt.close()

