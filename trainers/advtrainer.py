import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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
from copy import deepcopy

from trainers.trainer import Trainer
from losses.bl import BL
from losses.sim import sim_loss
from losses.ortho import ortho_loss
from losses.lw import lw_loss
from utils.misc import denormalize, divide_img_into_patches, patchwise_random_rotate, seed_everything, get_current_datetime

class AdvTrainer(Trainer):
    def __init__(self, seed, version, device, log_para, mode):
        super().__init__(seed, version, device)

        self.log_para = log_para
        self.mode = mode
        self.augment_transforms = T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1)
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def augment(self, imgs):
        imgs_aug = denormalize(imgs)
        imgs_aug = self.augment_transforms(imgs_aug)
        imgs_aug = self.normalize(imgs_aug)
        return imgs_aug

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

    def compute_count_loss(self, loss: nn.Module, pred_dmaps, gt_datas, weights=None):
        if loss.__class__.__name__ == 'MSELoss':
            _, gt_dmaps, _ = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            if weights is not None:
                pred_dmaps = pred_dmaps * weights
                gt_dmaps = gt_dmaps * weights
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
        patch_size = 720
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
        patch_size = 720
        if h >= patch_size or w >= patch_size:
            dmap = torch.zeros(1, 1, h, w)
            dmap_raw = torch.zeros(1, 1, h//4, w//4)
            bmap = torch.zeros(1, 3, h//16, w//16)
            img_patches, nh, nw = divide_img_into_patches(img, patch_size)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i*nw+j]
                    pred_dmap, (pred_dmap_raw, pred_bmap, _, _, _, _, _) = model(patch)
                    dmap[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred_dmap
                    dmap_raw[:, :, i*patch_size//4:(i+1)*patch_size//4, j*patch_size//4:(j+1)*patch_size//4] = pred_dmap_raw
                    bmap[:, :, i*patch_size//16:(i+1)*patch_size//16, j*patch_size//16:(j+1)*patch_size//16] = pred_bmap
        else:
            dmap, (dmap_raw, bmap, _, _, _, _, _) = model(img)

        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()
        dmap_raw = dmap_raw[0, 0].cpu().detach().numpy().squeeze()
        bmap = bmap[0, 0].cpu().detach().numpy().squeeze()

        return dmap, dmap_raw, bmap
    
    def train_step(self, model, loss, optimizer, batch, epoch):
        imgs1, imgs2, gt_datas = batch
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        gt_bmaps = gt_datas[-1].to(self.device)

        if self.mode == 'regression':
            optimizer.zero_grad()
            dmaps, (_, bmaps, feats, feats_rec) = model(imgs1, gt_bmaps)
            loss_den = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_cls = F.binary_cross_entropy(bmaps, gt_bmaps)
            loss_sim = sim_loss(feats, feats_rec)
            print(f'loss_den: {loss_den}, loss_cls: {loss_cls}, loss_sim_den: {loss_sim}')
            loss_total = loss_den + 10 * loss_cls + loss_sim
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'generation':
            optimizer.zero_grad()
            imgs_rec = model(imgs1)
            loss_total = F.mse_loss(imgs_rec, imgs1)
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'final':
            optimizer.zero_grad()
            dmaps1, dmaps2, bmaps1, bmaps2, loss_logits, loss_con2, loss_err = model.forward_pair(imgs1, imgs2, gt_bmaps)
            loss_dmap = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_cls = F.binary_cross_entropy(bmaps1, gt_bmaps) + F.binary_cross_entropy(bmaps2, gt_bmaps)
            # bmap_err = (bmaps1.round() != bmaps2.round()).detach()
            # loss_sim = F.mse_loss(bmaps2[bmap_err], bmaps1[bmap_err])
            # print(f'loss_logits: {loss_logits:.3f}, loss_err: {loss_err:.3f}')
            # print(f'loss_f_sim: {loss_f_sim:.3f}, loss_fnew_sim: {loss_fnew_sim:.3f}')
            # print(f'loss_con1: {loss_con1:.3f}, loss_con2: {loss_con2:.3f}')
            loss_total = loss_dmap + 10 * loss_cls + loss_logits # + loss_err # loss_con1 + loss_con2 # + (loss_f_sim + loss_fnew_sim)

            loss_total.backward()
            optimizer.step()

        elif self.mode == 'emmm':
            optimizer.zero_grad()
            dmaps, (dmaps_raw, bmaps, feats, feats_rec, loss_bg_sim, loss_fg_sim, loss_bg_dissim, loss_fg_dissim, loss_dissim, fg_maps) = model(imgs1, gt_bmaps)
            dmaps_aug, (dmaps_raw_aug, bmaps_aug, feats_aug, feats_rec_aug, loss_bg_sim_aug, loss_fg_sim_aug, loss_bg_dissim_aug, loss_fg_dissim_aug, loss_dissim_aug, fg_maps_aug) = model(imgs2, gt_bmaps, raw=False)

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(denormalize(imgs1.detach())[0].cpu().permute(1, 2, 0).numpy())
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(fg_maps[0].detach().cpu().numpy().squeeze())
            plt.savefig(os.path.join(self.log_dir, f'bg_map_{epoch}.png'))
            plt.close()

            loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_dmap_aug = self.compute_count_loss(loss, dmaps_aug, gt_datas)
            loss_dmap_con = F.mse_loss(dmaps, dmaps_aug)
            loss_cls = F.binary_cross_entropy(bmaps, gt_bmaps)
            loss_cls_aug = F.binary_cross_entropy(bmaps_aug, gt_bmaps)
            loss_cls_con = F.mse_loss(bmaps, bmaps_aug)

            loss_f_con = F.mse_loss(feats, feats_aug)
            loss_frec_con = F.mse_loss(feats_rec, feats_rec_aug)

            mem_bg_mean = model.mem_bg.squeeze().mean(dim=-1)
            mem_fg_mean = model.mem_fg.squeeze().mean(dim=-1)
            loss_mem = -F.mse_loss(mem_bg_mean, mem_fg_mean)

            print(f'bg_sim: {loss_bg_sim:.3f}, fg_sim: {loss_fg_sim:.3f}, bg_dissim: {loss_bg_dissim:.3f}, fg_dissim: {loss_fg_dissim:.3f}, f_con: {loss_f_con:.3f}, frec_con: {loss_frec_con:.3f}')

            loss_total = (loss_dmap + loss_dmap_aug) + 10 * (loss_cls + loss_cls_aug)  + \
                1 * (loss_bg_sim + loss_bg_sim_aug + loss_fg_sim + loss_fg_sim_aug) - \
                1 * (loss_bg_dissim + loss_bg_dissim_aug + loss_fg_dissim + loss_fg_dissim_aug) + \
                10 * (loss_f_con + loss_frec_con)
            # 1 * (loss_dissim + loss_dissim_aug) - \

            loss_total.backward()
            optimizer.step()

        else:
            optimizer.zero_grad()
            dmaps, (dmaps_raw, bmaps, feats, feats_rec, xs) = model(imgs1, gt_bmaps)
            dmaps_aug, (dmaps_raw_aug, bmaps_aug, feats_aug, feats_rec_aug, xs_aug) = model(imgs2, gt_bmaps, raw=False)

            # loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            # loss_dmap_aug = self.compute_count_loss(loss, dmaps_aug, gt_datas)
            # loss_dmap_sim = F.l1_loss(dmaps, dmaps_aug)
            # loss_cls = F.binary_cross_entropy(bmaps, gt_bmaps)
            # loss_cls_aug = F.binary_cross_entropy(bmaps_aug, gt_bmaps)
            # loss_cls_sim = F.l1_loss(bmaps, bmaps_aug)
            # loss_sim = sim_loss(feats, feats_rec)
            # loss_sim_aug = sim_loss(feats_aug, feats_rec_aug)
            # loss_ortho = ortho_loss(model_reg.mem.squeeze().t(), model_reg.mem.squeeze().t()) / (1024*1024)
            # # loss_trans = F.mse_loss(feats_rec, feats_rec_aug) + F.mse_loss(feats, feats_aug)
            # print(f'loss_sim: {loss_sim:.4f}, loss_sim_aug: {loss_sim_aug:.4f}, loss_dmap_sim: {loss_dmap_sim:.4f}, loss_cls_sim: {loss_cls_sim:.4f}, loss_ortho: {loss_ortho:.4f}')
            # # print(f'loss_dmap: {loss_dmap:.4f}, loss_dmap_aug: {loss_dmap_aug:.4f}, loss_cls: {loss_cls:.4f}, loss_cls_aug: {loss_cls_aug:.4f}, loss_sim: {loss_sim:.4f}, loss_trans: {loss_trans:.4f}')
            # loss_total = loss_dmap + loss_dmap_aug + loss_dmap_sim + 10 * (loss_cls + loss_cls_aug + loss_cls_sim) + (loss_sim) + loss_ortho
            # loss_total.backward()

            loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_dmap_aug = self.compute_count_loss(loss, dmaps_aug, gt_datas)
            loss_dmap_con = F.mse_loss(dmaps, dmaps_aug)
            loss_cls = F.binary_cross_entropy(bmaps, gt_bmaps)
            loss_cls_aug = F.binary_cross_entropy(bmaps_aug, gt_bmaps)
            loss_cls_con = F.mse_loss(bmaps, bmaps_aug)

            incon_dmap = ((dmaps_raw - dmaps_raw_aug).detach().abs() > 0.1).repeat(1, 256, 1, 1)
            incon_map = (bmaps.round() != bmaps_aug.round()).detach().repeat(1, 512, 1, 1)
            if incon_map.sum() > 0:
                loss_lw = F.mse_loss(xs[incon_map], xs_aug[incon_map])
            else:
                loss_lw = 0
            
            loss_sim = sim_loss(feats, feats_rec) + sim_loss(feats.clone().detach(), feats_rec_aug)

            # print(f'dmap_con: {loss_dmap_con:.3f}, cls_con: {loss_cls_con:.3f}, lw: {loss_lw:.3f}')

            if epoch < 40:
                loss_total = (loss_dmap + loss_dmap_aug) + 10 * (loss_cls + loss_cls_aug)
            else:
                loss_total = (loss_dmap + loss_dmap_aug + loss_dmap_con) + \
                    10 * (loss_cls + loss_cls_aug + loss_cls_con)
            loss_total.backward()

            optimizer.step()

        return loss_total.detach().item()

    def val_step(self, model, batch):
        img1, img2, gt, _, _ = batch
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'generation':
            img_rec = model(img1)
            loss = F.mse_loss(img_rec, img1)
            return loss.detach().item()
        
        else:
            pred_count = self.predict(model, img1)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return mae, {'mse': mse}

    def test_step(self, model, batch):
        img1, img2, gt, _, _ = batch
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'generation':
            img_rec = model(img1)
            loss = F.mse_loss(img_rec, img1)
            return {'loss': loss.detach().item()}
    
        else:
            pred_count = self.predict(model, img1)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        
    def vis_step(self, model, batch):
        img1, img2, gt, name, _ = batch
        vis_dir = os.path.join(self.log_dir, 'vis')
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'regression':
            pred_dmap, pred_dmap_raw, pred_bmap = self.get_visualized_results(model, img1)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_raw_count = pred_dmap_raw.sum() / self.log_para * 16
            pred_count = pred_dmap.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img1, pred_dmap_raw, pred_dmap, pred_bmap]
            titles = [f'GT: {gt_count}', f'Pred_raw: {pred_raw_count}', f'Pred: {pred_count}', 'Cls']

            fig = plt.figure(figsize=(20, 8))
            for i in range(4):
                ax = fig.add_subplot(1, 4, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

        elif self.mode == 'generation':
            img_rec = model(img1)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_rec = denormalize(img_rec.detach())[0].cpu().permute(1, 2, 0).numpy()

            datas = [img1, img_rec]
            titles = ['Input', 'Reconstruction']

            fig = plt.figure(figsize=(20, 6))
            for i in range(2):
                ax = fig.add_subplot(1, 2, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

        else:
            pred_dmap, _, pred_bmap = self.get_visualized_results(model, img1)
            noisy_dmap, _, noisy_bmap = self.get_visualized_results(model, img2)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy()
            # img_noisy = denormalize(img_noisy.detach())[0].cpu().permute(1, 2, 0).numpy()
            # res = denormalize(res.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count = pred_dmap.sum() / self.log_para
            noisy_count = noisy_dmap.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img1, pred_dmap, pred_bmap, img2, noisy_dmap, noisy_bmap]
            titles = [f'GT: {gt_count}', f'Pred: {pred_count}', 'Cls', 'Rec', f'Noisy: {noisy_count}', 'Noisy_Cls']

            fig = plt.figure(figsize=(15, 6))
            for i in range(6):
                ax = fig.add_subplot(2, 3, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

    def train_and_test_epoch(self, model, loss, train_dataloader, val_dataloader, test_dataloader, \
                             optimizer, scheduler, epoch, best_criterion, best_epoch):
        best_criterion, best_epoch = self.train_epoch(model, loss, train_dataloader, val_dataloader, \
                                                      optimizer, scheduler, epoch, best_criterion, best_epoch)
        self.test(model, test_dataloader)
        return best_criterion, best_epoch
    
    def train_and_test(self, model, loss, train_dataloader, val_dataloader, test_dataloader, \
                       optimizer, scheduler, checkpoint=None, num_epochs=100):
        self.log('Start training and testing at {}'.format(get_current_datetime()))
        self.load_ckpt(model, checkpoint)

        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]
        loss = loss.to(self.device)
        
        best_criterion = 1e10
        best_epoch = -1

        for epoch in range(num_epochs):
            best_criterion, best_epoch = self.train_and_test_epoch(
                model, loss, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, epoch, best_criterion, best_epoch)

        self.log('Best epoch: {}, best criterion: {}'.format(best_epoch, best_criterion))
        self.log('Training results saved to {}'.format(self.log_dir))
        self.log('End training and testing at {}'.format(get_current_datetime()))
