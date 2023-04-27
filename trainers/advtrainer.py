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

class AdvTrainer(Trainer):
    def __init__(self, seed, version, device, log_para, mode):
        super().__init__(seed, version, device)

        self.log_para = log_para
        self.mode = mode

    def load_ckpt(self, model, path):
        if isinstance(model, list):
            if path is not None:
                super().load_ckpt(model[0], path[0])
                super().load_ckpt(model[1], path[1])
        else:
            super().load_ckpt(model, path)

    def save_ckpt(self, model, path):
        if isinstance(model, list):
            super().save_ckpt(model[0], path.replace('.pth', '_gen.pth'))
            super().save_ckpt(model[1], path.replace('.pth', '_reg.pth'))
        else:
            super().save_ckpt(model, path)

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
                pred = model(patch)[0]
                pred_count += torch.sum(pred).cpu().item() / self.log_para
        else:
            pred_dmap = model(img)[0]
            pred_count = pred_dmap.sum().cpu().item() / self.log_para

        return pred_count
    
    def get_visualized_results(self, model, img):
        h, w = img.shape[2:]
        patch_size = 1024
        if h >= patch_size or w >= patch_size:
            dmap = torch.zeros(1, 1, h, w)
            bmap = torch.zeros(1, 3, h//32, w//32)
            img_patches, nh, nw = divide_img_into_patches(img, patch_size)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i*nw+j]
                    pred_dmap, _, pred_bmap = model(patch)
                    dmap[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred_dmap
                    bmap[:, :, i*patch_size//32:(i+1)*patch_size//32, j*patch_size//32:(j+1)*patch_size//32] = pred_bmap
        else:
            dmap, _, bmap = model(img)

        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()
        bmap = bmap[0].cpu().detach().numpy().transpose(1, 2, 0)

        return dmap, bmap
    
    def train_step(self, model, loss, optimizer, batch, epoch):
        imgs1, imgs2, gt_datas = batch
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)

        if self.mode == 'regression':
            optimizer.zero_grad()
            dmaps, _, bmaps = model(imgs1)
            loss_den = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_cls = F.cross_entropy(bmaps, gt_datas[-1].to(self.device).long())
            loss_total = loss_den + 10 * loss_cls
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'generation':
            optimizer.zero_grad()
            imgs_rec = model(imgs1)
            loss_total = F.mse_loss(imgs_rec, imgs1)
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'joint':
            model_gen, model_reg = model
            opt_gen, opt_reg = optimizer

            opt_gen.zero_grad()
            imgs_noisy = model_gen(imgs2)
            dmaps, dmaps_raw, bmaps = model_reg(imgs1)
            gt_bmaps_resized = F.interpolate(gt_datas[-1].to(self.device).float().unsqueeze(1), mode='nearest', scale_factor=32)
            imgs_new = imgs2 * (gt_bmaps_resized > 0).float() + imgs_noisy * (gt_bmaps_resized == 0).float()
            dmaps_noisy, dmaps_raw_noisy, bmaps_noisy = model_reg(imgs_new)
            loss_dmap_raw = F.mse_loss(dmaps_raw, dmaps_raw_noisy)
            loss_cls_noisy = F.cross_entropy(bmaps_noisy, (gt_datas[-1].to(self.device).long() + 1) % 3) + \
                             F.cross_entropy(bmaps_noisy, (gt_datas[-1].to(self.device).long() + 2) % 3)
            loss_rec = F.mse_loss(imgs_noisy, imgs2)
            loss_gen = 10 * loss_dmap_raw + 10 * loss_cls_noisy + 1000 * loss_rec
            # loss_gen = 10 * loss_cls_noisy + 10 * loss_rec
            print(f'loss_dmap_raw: {loss_dmap_raw.item():.4f}, loss_cls_noisy: {loss_cls_noisy.item():.4f}, loss_rec: {loss_rec.item():.4f}')
            # print(f'loss_cls_noisy: {loss_cls_noisy.item():.4f}, loss_rec: {loss_rec.item():.4f}')
            loss_gen.backward()
            opt_gen.step()

            opt_reg.zero_grad()
            imgs_noisy = model_gen(imgs2)
            dmaps, dmaps_raw, bmaps = model_reg(imgs1)
            gt_bmaps_resized = F.interpolate(gt_datas[-1].to(self.device).float().unsqueeze(1), mode='nearest', scale_factor=32)
            imgs_new = imgs2 * (gt_bmaps_resized > 0).float() + imgs_noisy * (gt_bmaps_resized == 0).float()
            dmaps_noisy, dmaps_raw_noisy, bmaps_noisy = model_reg(imgs_new)
            loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_dmap_noisy = self.compute_count_loss(loss, dmaps_noisy, gt_datas)
            loss_cls = F.cross_entropy(bmaps, gt_datas[-1].to(self.device).long())
            loss_cls_noisy = F.cross_entropy(bmaps_noisy, gt_datas[-1].to(self.device).long())
            loss_dmap_sim = F.mse_loss(dmaps_raw, dmaps_raw_noisy)
            loss_cls_sim = F.cross_entropy(bmaps_noisy, bmaps)
            print(f'loss_dmap: {loss_dmap.item():.4f}, loss_dmap_noisy: {loss_dmap_noisy.item():.4f}, loss_cls: {loss_cls.item():.4f}, loss_cls_noisy: {loss_cls_noisy.item():.4f}, loss_dmap_sim: {loss_dmap_sim.item():.4f}, loss_cls_sim: {loss_cls_sim.item():.4f}')
            loss_reg = loss_dmap + loss_dmap_noisy + loss_dmap_sim + 10 * (loss_cls + loss_cls_noisy + loss_cls_sim)
            loss_reg.backward()
            opt_reg.step()

            loss_total = loss_reg + loss_gen

        else:
            model_gen, model_reg = model
            opt_gen, opt_reg = optimizer

            opt_reg.zero_grad()
            imgs_noisy = model_gen(imgs2)
            dmaps, dmaps_raw, bmaps = model_reg(imgs1)
            gt_bmaps_resized = F.interpolate(gt_datas[-1].to(self.device).float().unsqueeze(1), mode='nearest', scale_factor=32)
            imgs_new = imgs2 * (gt_bmaps_resized > 0).float() + imgs_noisy * (gt_bmaps_resized == 0).float()
            dmaps_noisy, dmaps_raw_noisy, bmaps_noisy = model_reg(imgs_new)
            loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_dmap_noisy = self.compute_count_loss(loss, dmaps_noisy, gt_datas)
            loss_cls = F.cross_entropy(bmaps, gt_datas[-1].to(self.device).long())
            loss_cls_noisy = F.cross_entropy(bmaps_noisy, gt_datas[-1].to(self.device).long())
            loss_dmap_sim = F.mse_loss(dmaps_raw, dmaps_raw_noisy)
            loss_cls_sim = F.cross_entropy(bmaps_noisy, bmaps)
            # print(f'loss_dmap: {loss_dmap.item():.4f}, loss_dmap_noisy: {loss_dmap_noisy.item():.4f}, loss_cls: {loss_cls.item():.4f}, loss_cls_noisy: {loss_cls_noisy.item():.4f}, loss_dmap_sim: {loss_dmap_sim.item():.4f}, loss_cls_sim: {loss_cls_sim.item():.4f}')
            loss_total = loss_dmap + loss_dmap_noisy + loss_dmap_sim + 10 * (loss_cls + loss_cls_noisy + loss_cls_sim)
            loss_total.backward()
            opt_reg.step()

        return loss_total.detach().item()

    def val_step(self, model, batch):
        img, gt, _, _ = batch
        img = img.to(self.device)

        if self.mode == 'regression':
            pred_count = self.predict(model, img)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            return mae
        
        elif self.mode == 'generation':
            img_rec = model(img)
            loss = F.mse_loss(img_rec, img)
            return loss.detach().item()
        
        else:
            gen, reg = model
            # img_noisy = gen(img)
            pred_count = self.predict(reg, img)
            # noisy_count = self.predict(reg, img_noisy)
            gt_count = gt.shape[1]
            # mae = np.abs((pred_count+noisy_count)/2 - gt_count)
            mae = np.abs(pred_count - gt_count)
            return mae
        
    def test_step(self, model, batch):
        img, gt, _, _ = batch
        img = img.to(self.device)

        if self.mode == 'regression':
            pred_count = self.predict(model, img)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        
        elif self.mode == 'generation':
            img, gt, _, _ = batch
            img = img.to(self.device)
            img_rec = model(img)
            loss = F.mse_loss(img_rec, img)
            return {'loss': loss.detach().item()}
        
        else:
            _, reg = model
            pred_count = self.predict(reg, img)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        
    def vis_step(self, model, batch):
        img, gt, name, _ = batch
        vis_dir = os.path.join(self.log_dir, 'vis')
        img = img.to(self.device)

        if self.mode == 'regression':
            pred_dmap, pred_bmap = self.get_visualized_results(model, img)
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count = pred_dmap.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img, pred_dmap, pred_bmap]
            titles = [f'GT: {gt_count}', f'Pred: {pred_count}', 'Cls']

            fig = plt.figure(figsize=(20, 6))
            for i in range(3):
                ax = fig.add_subplot(1, 3, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

        elif self.mode == 'generation':
            img_rec = model(img)
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_rec = denormalize(img_rec.detach())[0].cpu().permute(1, 2, 0).numpy()

            datas = [img, img_rec]
            titles = ['Input', 'Reconstruction']

            fig = plt.figure(figsize=(20, 6))
            for i in range(2):
                ax = fig.add_subplot(1, 2, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

        else:
            gen, reg = model
            img_noisy = gen(img)
            pred_dmap, pred_bmap = self.get_visualized_results(reg, img)
            noisy_dmap, noisy_bmap = self.get_visualized_results(reg, img_noisy)
            img = denormalize(img.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_noisy = denormalize(img_noisy.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count = pred_dmap.sum() / self.log_para
            noisy_count = noisy_dmap.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img, pred_dmap, pred_bmap, img_noisy, noisy_dmap, noisy_bmap]
            titles = [f'GT: {gt_count}', f'Pred: {pred_count}', 'Cls', 'Rec', f'Noisy: {noisy_count}', 'Noisy_Cls']

            fig = plt.figure(figsize=(20, 6))
            for i in range(6):
                ax = fig.add_subplot(2, 3, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()
            