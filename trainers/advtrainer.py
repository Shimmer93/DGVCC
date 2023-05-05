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
        patch_size = 1600
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
                    pred_dmap, (pred_dmap_raw, pred_bmap, _, _) = model(patch)
                    dmap[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred_dmap
                    dmap_raw[:, :, i*patch_size//4:(i+1)*patch_size//4, j*patch_size//4:(j+1)*patch_size//4] = pred_dmap_raw
                    bmap[:, :, i*patch_size//16:(i+1)*patch_size//16, j*patch_size//16:(j+1)*patch_size//16] = pred_bmap
        else:
            dmap, (dmap_raw, bmap, _, _) = model(img)

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

        elif self.mode == 'joint':
            model_gen, model_reg = model
            opt_gen, opt_reg = optimizer

            # imgs2 = patchwise_random_rotate(imgs2, gt_bmaps)

            opt_gen.zero_grad()

            # model_gen_clone = deepcopy(model_gen)
            imgs_noisy = model_gen(imgs2 + torch.randn_like(imgs2) * 0.1)
            imgs_noisy2 = model_gen(imgs2 + torch.randn_like(imgs2) * 0.1)
            loss_div = -F.l1_loss(imgs_noisy, imgs_noisy2)
            resized_bmaps = F.interpolate(gt_bmaps, scale_factor=16, mode='nearest')
            imgs_noisy = imgs_noisy * (1 - resized_bmaps) + imgs2 * resized_bmaps
            # dmaps, (_, bmaps, feats, feats_rec) = model_reg(imgs2, gt_bmaps)
            dmaps_noisy, (_, bmaps_noisy, feats_noisy, feats_rec_noisy) = model_reg(imgs_noisy, gt_bmaps)
            # confused_bmaps = bmaps_noisy * (1 - gt_bmaps) + gt_bmaps * (1 - bmaps_noisy)
            loss_cls_confuse = F.binary_cross_entropy(bmaps_noisy, 1-gt_bmaps)
            loss_sim_confuse = -sim_loss(feats_noisy, feats_rec_noisy)
            loss_rec = F.mse_loss(imgs_noisy[(resized_bmaps==0).repeat(1, 3, 1, 1)], imgs2[(resized_bmaps==0).repeat(1, 3, 1, 1)])
            loss_gen = loss_cls_confuse + loss_sim_confuse + 100 * loss_rec + 10 * loss_div
            print(f'loss_cls_confuse: {loss_cls_confuse}, loss_sim_confuse: {loss_sim_confuse}, loss_rec: {loss_rec}, loss_div: {loss_div}')
            loss_gen.backward()
            opt_gen.step()

            # with torch.no_grad():
            #     alpha = np.exp(-1000*loss_rec.detach().item())
            #     for param, param_clone in zip(model_gen.parameters(), model_gen_clone.parameters()):
            #         param.data = param.data * alpha + param_clone.data * (1 - alpha)

            opt_reg.zero_grad()

            # model_reg_clone = deepcopy(model_reg)
            imgs_noisy = model_gen(imgs2 + torch.randn_like(imgs2) * 0.1)
            resized_bmaps = F.interpolate(gt_bmaps, scale_factor=16, mode='nearest')
            imgs_noisy = imgs_noisy * (1 - resized_bmaps) + imgs2 * resized_bmaps
            dmaps, (_, bmaps, feats, feats_rec) = model_reg(imgs1, gt_bmaps)
            dmaps_noisy, (_, bmaps_noisy, feats_noisy, feats_rec_noisy) = model_reg(imgs_noisy, gt_bmaps)
            loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_dmap_noisy = self.compute_count_loss(loss, dmaps_noisy, gt_datas)
            # loss_dmap_sim = F.mse_loss(dmaps, dmaps_noisy)
            loss_cls = F.binary_cross_entropy(bmaps, gt_bmaps)
            loss_cls_noisy = F.binary_cross_entropy(bmaps_noisy, gt_bmaps)
            # loss_cls_sim = F.mse_loss(bmaps, bmaps_noisy)
            loss_sim = sim_loss(feats, feats_rec)
            loss_sim_noisy = sim_loss(feats_noisy, feats_rec_noisy)
            loss_trans = sim_loss(feats_rec, feats_rec_noisy)
            loss_trans_noisy = sim_loss(feats_rec_noisy, feats_rec)
            print(f'loss_dmap: {loss_dmap:.4f}, loss_dmap_noisy: {loss_dmap_noisy:.4f}, loss_cls: {loss_cls:.4f}, loss_cls_noisy: {loss_cls_noisy:.4f}, loss_sim: {loss_sim:.4f}, loss_sim_noisy: {loss_sim_noisy:.4f}')
            loss_reg = loss_dmap + loss_dmap_noisy + 10 * (loss_cls + loss_cls_noisy) + (loss_sim + loss_sim_noisy + loss_trans + loss_trans_noisy)
            loss_reg.backward()
            opt_reg.step()

            # with torch.no_grad():
            #     alpha = np.exp(-1000*loss_rec.detach().item())
            #     for param, param_clone in zip(model_reg.parameters(), model_reg_clone.parameters()):
            #         param.data = param.data * alpha + param_clone.data * (1 - alpha)

            loss_total = loss_reg + loss_gen

        else:
            model_gen, model_reg = model
            opt_gen, opt_reg = optimizer

            opt_reg.zero_grad()
            # imgs_noisy = model_gen(imgs2 + torch.randn_like(imgs2) * 0.1)
            # resized_bmaps = F.interpolate(gt_bmaps, scale_factor=16, mode='nearest')
            # imgs_noisy = imgs_noisy * (1 - resized_bmaps) + imgs2 * resized_bmaps
            # model_reg.mem.requires_grad_(False)
            dmaps, (_, bmaps, feats, feats_rec) = model_reg(imgs1, gt_bmaps)
            dmaps_noisy, (_, bmaps_noisy, feats_noisy, feats_rec_noisy) = model_reg(imgs2, gt_bmaps)
            loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas)
            loss_dmap_noisy = self.compute_count_loss(loss, dmaps_noisy, gt_datas)
            # loss_dmap_sim = F.mse_loss(dmaps, dmaps_noisy)
            loss_cls = F.binary_cross_entropy(bmaps, gt_bmaps)
            loss_cls_noisy = F.binary_cross_entropy(bmaps_noisy, gt_bmaps)
            # loss_cls_sim = F.mse_loss(bmaps, bmaps_noisy)
            loss_sim = sim_loss(feats, feats_rec)
            loss_trans = F.mse_loss(feats_rec, feats_rec_noisy) + F.mse_loss(feats, feats_noisy)
            # loss_trans_noisy = F.mse_loss(feats_rec_noisy, feats_rec)
            print(f'loss_dmap: {loss_dmap:.4f}, loss_dmap_noisy: {loss_dmap_noisy:.4f}, loss_cls: {loss_cls:.4f}, loss_cls_noisy: {loss_cls_noisy:.4f}, loss_sim: {loss_sim:.4f}, loss_trans: {loss_trans:.4f}')
            loss_total = loss_dmap + loss_dmap_noisy + 10 * (loss_cls + loss_cls_noisy) + (loss_sim + 100 * loss_trans)
            loss_total.backward()
            opt_reg.step()

            # dmaps, dmaps_raw, bmaps = model_reg(imgs1, gt_bmaps)
            # dmaps_noisy, dmaps_raw_noisy, bmaps_noisy = model_reg(imgs2, gt_bmaps)
            # loss_dmap_sim = F.l1_loss(dmaps.clone(), dmaps_noisy.clone(), reduction='none').detach() + 1
            # loss_dmap = self.compute_count_loss(loss, dmaps, gt_datas, loss_dmap_sim)
            # loss_dmap_noisy = self.compute_count_loss(loss, dmaps_noisy, gt_datas, loss_dmap_sim)
            # loss_cls = F.mse_loss(bmaps, gt_bmaps)
            # loss_cls_noisy = F.mse_loss(bmaps_noisy, gt_bmaps)
            # print(f'loss_dmap: {loss_dmap.item():.4f}, loss_dmap_noisy: {loss_dmap_noisy.item():.4f}, loss_cls: {loss_cls.item():.4f}, loss_cls_noisy: {loss_cls_noisy.item():.4f}, loss_dmap_sim: {loss_dmap_sim.item():.4f}, loss_cls_sim: {loss_cls_sim.item():.4f}')
            # loss_total = loss_dmap + loss_dmap_noisy + 10 * (loss_cls + loss_cls_noisy)
            # loss_total.backward()
            # opt_reg.step()

        return loss_total.detach().item()

    def val_step(self, model, batch):
        img1, img2, gt, _, _ = batch
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'regression':
            pred_count = self.predict(model, img1)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return mae, {'mse': mse}
        
        elif self.mode == 'generation':
            img_rec = model(img1)
            loss = F.mse_loss(img_rec, img1)
            return loss.detach().item()
        
        else:
            gen, reg = model
            # img_noisy = gen(img1)
            pred_count = self.predict(reg, img1)
            noisy_count = self.predict(reg, img2)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            mae_noisy = np.abs(noisy_count - gt_count)
            res = np.abs(pred_count - noisy_count)
            return mae + mae_noisy + res, {'mae': mae, 'mae_noisy': mae_noisy}
        
    def test_step(self, model, batch):
        img1, img2, gt, _, _ = batch
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'regression':
            pred_count = self.predict(model, img1)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        
        elif self.mode == 'generation':
            img_rec = model(img1)
            loss = F.mse_loss(img_rec, img1)
            return {'loss': loss.detach().item()}
        
        else:
            _, reg = model
            pred_count = self.predict(reg, img1)
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
            gen, reg = model
            pred_dmap, _, pred_bmap = self.get_visualized_results(reg, img1)
            # img2 = patchwise_random_rotate(img2, torch.from_numpy(pred_bmap).unsqueeze(0).unsqueeze(0).to(self.device))
            # img_noisy = self.augment(gen(img1))
            img_noisy = gen(img1 + torch.randn_like(img1) * 0.1)
            resized_bmap = F.interpolate(torch.from_numpy(pred_bmap).unsqueeze(0).unsqueeze(0).to(self.device), scale_factor=16, mode='nearest')
            img_noisy = img_noisy * (1 - resized_bmap) + img1 * resized_bmap
            res = img_noisy - img1
            noisy_dmap, _, noisy_bmap = self.get_visualized_results(reg, img_noisy)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy()
            img_noisy = denormalize(img_noisy.detach())[0].cpu().permute(1, 2, 0).numpy()
            res = denormalize(res.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count = pred_dmap.sum() / self.log_para
            noisy_count = noisy_dmap.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img1, pred_dmap, pred_bmap, img_noisy, noisy_dmap, noisy_bmap, img2, res]
            titles = [f'GT: {gt_count}', f'Pred: {pred_count}', 'Cls', 'Rec', f'Noisy: {noisy_count}', 'Noisy_Cls', 'Aug', 'Res']

            fig = plt.figure(figsize=(20, 9))
            for i in range(8):
                ax = fig.add_subplot(3, 3, i+1)
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
