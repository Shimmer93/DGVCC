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

from models.dgvccnet2 import DGVCCNet2
from losses.bl import BL
from datasets.den_dataset import DensityMapDataset
from datasets.bay_dataset import BayesianDataset

from utils.misc import divide_img_into_patches, denormalize, AverageMeter, seed_everything

class DGVCCTrainer():
    def __init__(self, config, device):
        with open(config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # seed_everything(cfg['seed'])

        self.device = torch.device(device)
        self.model = DGVCCNet2(**cfg['model'])
        self.model.to(self.device)

        self.method = cfg['method']
        if self.method == 'Density':
            self.den_loss = nn.MSELoss()
            train_dataset = DensityMapDataset(method='train', **cfg['train_dataset'])
            val_dataset = DensityMapDataset(method='test', **cfg['train_dataset'])
            test_dataset = DensityMapDataset(method='test', **cfg['test_dataset'])
            collate = DensityMapDataset.collate
        elif self.method == 'Bayesian':
            self.den_loss = BL()
            train_dataset = BayesianDataset(method='train', **cfg['train_dataset'])
            val_dataset = BayesianDataset(method='test', **cfg['train_dataset'])
            test_dataset = BayesianDataset(method='test', **cfg['test_dataset'])
            collate = BayesianDataset.collate
            
        self.train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)

        self.downsample = cfg['train_dataset']['downsample']
        self.num_epochs = cfg['num_epochs']
        self.log_para = cfg['log_para']
        self.patch_size = cfg['patch_size']
        self.version = cfg['version']
        self.log_dir = os.path.join('logs', self.version)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        os.system('cp {} {}'.format(config, self.log_dir))

        self.opt_gen = torch.optim.AdamW(self.model.gen.parameters(), **cfg['optimizer'])
        self.opt_gen_cyc = torch.optim.AdamW(self.model.gen_cyc.parameters(), **cfg['optimizer'])
        self.opt_reg = torch.optim.AdamW(self.model.reg.parameters(), **cfg['optimizer'])

        self.sch_gen = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_gen, **cfg['scheduler'])
        self.sch_gen_cyc = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_gen_cyc, **cfg['scheduler'])
        self.sch_reg = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_reg, **cfg['scheduler'])

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write(msg + '\n')

    def compute_count_loss(self, pred_dmaps, gt_datas):
        if self.method == 'Density':
            _, gt_dmaps = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            loss = self.den_loss(pred_dmaps, gt_dmaps * self.log_para)

        elif self.method == 'Bayesian':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            targs = targs.to(self.device)
            loss = self.den_loss(gts, targs, st_sizes, pred_dmaps)
        
        else:
            raise NotImplementedError

        return loss
    
    def compute_count_loss_final(self, pred_dmaps, gt_datas):
        if self.method == 'Density':
            _, gt_dmaps = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            gt_dmaps = gt_dmaps.repeat(2, 1, 1, 1)
            loss = self.den_loss(pred_dmaps, gt_dmaps * self.log_para)
        
        else:
            raise NotImplementedError

        return loss
    
    def train_den_step(self, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)

        self.opt_reg.zero_grad()

        pred_dmaps = self.model.forward_den(imgs)
        loss = self.compute_count_loss(pred_dmaps, gt_datas)
        loss.backward()

        self.opt_reg.step()

        return loss.item()
    
    def train_rec_step(self, batch):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        self.opt_gen.zero_grad()

        _, loss = self.model.forward_rec(imgs)
        loss.backward()

        self.opt_gen.step()

        return loss.item()
    
    def train_final_step(self, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)
        z1 = torch.randn(imgs.size(0), 64, device=self.device)
        z2 = torch.randn(imgs.size(0), 64, device=self.device)

        self.opt_reg.zero_grad()

        d_cat = self.model.forward_dg(imgs, z1, z2, mode='reg_final')
        loss = self.compute_count_loss_final(d_cat, gt_datas)

        loss.backward()

        self.opt_reg.step()

        return loss.item()
    
    def train_step(self, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)
        z1 = torch.randn(imgs.size(0), 64, device=self.device)
        z2 = torch.randn(imgs.size(0), 64, device=self.device)

        # train generator
        self.opt_gen.zero_grad()
        self.opt_gen_cyc.zero_grad()

        d_gen, loss_cyc, loss_div, loss_ortho = self.model.forward_dg(imgs, z1, z2, mode='gen')
        loss_den_gen2 = self.compute_count_loss(d_gen, gt_datas)
        loss_gen = loss_den_gen2 + 10 * loss_cyc + 100 * loss_div + 100 * loss_ortho
        loss_gen.backward()

        self.opt_gen.step()
        self.opt_gen_cyc.step()

        # train regressor
        self.opt_reg.zero_grad()

        # d, d_gen, loss_sim = self.model.forward_dg(imgs, z1, z2, mode='reg')
        # loss_den = self.compute_count_loss(d, gt_datas)
        # loss_den_gen = self.compute_count_loss(d_gen, gt_datas)
        # loss_reg = 20 * loss_den + 10 * loss_den_gen + 100 * loss_sim
        d_cat, loss_sim = self.model.forward_dg(imgs, z1, z2, mode='reg')
        loss_den_cat = self.compute_count_loss_final(d_cat, gt_datas)
        loss_reg = loss_den_cat + 100 * loss_sim
        loss_reg.backward()

        self.opt_reg.step()

        return loss_gen.item(), loss_reg.item()
    
    def val_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b, _, h, w = img.shape
        # z1 = torch.randn(img.size(0), 64, device=self.device)
        # z2 = torch.randn(img.size(0), 64, device=self.device)

        assert b == 1, 'batch size should be 1 in validation'

        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)

            for patch in img_patches:
                # _, pred, _, _, _ = self.model.forward_dg(patch, z1, z2, mode='test')
                pred = self.model.forward_den(patch)
                pred_count += pred.sum().cpu().item() / self.log_para

        else:
            # _, pred, _, _, _ = self.model.forward_dg(img, z1, z2, mode='test')
            pred = self.model.forward_den(img)
            pred_count = pred.sum().cpu().item() / self.log_para

        gt_count = gt.shape[1]

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2

        return mae, mse
    
    def val_rec_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        ps = self.patch_size
        if h >= ps or w >= ps:
            loss = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)

            for patch in img_patches:
                _, loss_patch = self.model.forward_rec(patch)
                loss += loss_patch.cpu().item()

        else:
            _, loss = self.model.forward_rec(img)
            loss = loss.cpu().item()

        return loss
    
    def visualize_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        z1 = torch.randn(img.size(0), 64, device=self.device)
        z2 = torch.randn(img.size(0), 64, device=self.device)

        b, _, h, w = img.shape
        ds = self.downsample

        assert b == 1, 'batch size should be 1 in validation'

        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            pred_count_gen = 0
            pred_dmap = torch.zeros((1, 1, h//ds, w//ds), device=self.device)
            pred_dmap_gen = torch.zeros((1, 1, h//ds, w//ds), device=self.device)
            img_gen = torch.zeros((1, 3, h, w), device=self.device)
            img_gen2 = torch.zeros((1, 3, h, w), device=self.device)
            img_cyc = torch.zeros((1, 3, h, w), device=self.device)

            img_patches, nh, nw = divide_img_into_patches(img, ps)

            for i, patch in enumerate(img_patches):
                pred, pred_gen, patch_gen, patch_gen2, patch_cyc = self.model.forward_dg(patch, z1, z2, mode='test')
                pred_count += pred.sum().cpu().item() / self.log_para
                pred_count_gen += pred_gen.sum().cpu().item() / self.log_para

                pred_dmap[:,:,i//nw*ps//ds:(i//nw+1)*ps//ds,i%nw*ps//ds:(i%nw+1)*ps//ds] += pred
                pred_dmap_gen[:,:,i//nw*ps//ds:(i//nw+1)*ps//ds,i%nw*ps//ds:(i%nw+1)*ps//ds] += pred_gen
                img_gen[:,:,i//nw*ps:(i//nw+1)*ps,i%nw*ps:(i%nw+1)*ps] += patch_gen
                img_gen2[:,:,i//nw*ps:(i//nw+1)*ps,i%nw*ps:(i%nw+1)*ps] += patch_gen2
                img_cyc[:,:,i//nw*ps:(i//nw+1)*ps,i%nw*ps:(i%nw+1)*ps] += patch_cyc

        else:
            pred_dmap, pred_dmap_gen, img_gen, img_gen2, img_cyc = self.model.forward_dg(img, z1, z2, mode='test')
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            pred_count_gen = pred_dmap_gen.sum().cpu().item() / self.log_para
            
        gt_count = gt.shape[1]

        img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
        pred_dmap = pred_dmap[0,0].cpu().numpy()
        pred_dmap_gen = pred_dmap_gen[0,0].cpu().numpy()
        img_gen = denormalize(img_gen)[0].cpu().permute(1, 2, 0).numpy()
        img_gen2 = denormalize(img_gen2)[0].cpu().permute(1, 2, 0).numpy()
        img_cyc = denormalize(img_cyc)[0].cpu().permute(1, 2, 0).numpy()

        datas = [img, pred_dmap, pred_dmap_gen, img_gen, img_gen2, img_cyc]
        titles = [f'Original: {gt_count}', f'Pred: {pred_count:.2f}', f'Pred_gen: {pred_count_gen:.2f}', 'Generated', 'Generated2', 'Cycled']

        fig = plt.figure(figsize=(20, 10))
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1)
            ax.set_title(titles[i])
            ax.imshow(datas[i])

        plt.savefig(os.path.join(self.vis_dir, '{}.png'.format(name[0])))
        plt.close()

    def visualize_reg_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b, _, h, w = img.shape
        ds = self.downsample

        assert b == 1, 'batch size should be 1 in validation'

        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            pred_dmap = torch.zeros((1, 1, h//ds, w//ds), device=self.device)
            img_patches, nh, nw = divide_img_into_patches(img, ps)

            for i, patch in enumerate(img_patches):
                pred = self.model(patch)
                pred_count += pred.sum().cpu().item() / self.log_para
                pred_dmap[:,:,i//nw*ps//ds:(i//nw+1)*ps//ds,i%nw*ps//ds:(i%nw+1)*ps//ds] += pred

        else:
            pred = self.model(img)
            pred_count = pred.sum().cpu().item() / self.log_para
            pred_dmap = pred

        gt_count = gt.shape[1]

        fig = plt.figure(figsize=(20, 10))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_img.set_title('GT: {}'.format(gt_count))
        ax_img.imshow(img[0].cpu().permute(1, 2, 0).numpy())
        ax_den = fig.add_subplot(1, 2, 2)
        ax_den.set_title('Pred: {}'.format(pred_count))
        ax_den.imshow(pred_dmap[0,0].cpu().numpy())
        plt.savefig(os.path.join(self.vis_dir, '{}.png'.format(name[0])))
        plt.close()

    def train_epoch(self, epoch, mode='dg'):
        start_time = time.time()

        # training
        self.model.train()
        loss_gen_meter = AverageMeter()
        loss_reg_meter = AverageMeter()
        for batch in track(self.train_loader, description='Epoch: {}, Training...'.format(epoch), complete_style='dim cyan', total=len(self.train_loader)):
            if mode == 'den':
                loss = self.train_den_step(batch)
                loss_reg_meter.update(loss)
            elif mode == 'rec':
                loss = self.train_rec_step(batch)
                loss_gen_meter.update(loss)
            elif mode == 'final':
                loss = self.train_final_step(batch)
                loss_reg_meter.update(loss)
            else:
                losses = self.train_step(batch)
                loss_gen_meter.update(losses[0])
                loss_reg_meter.update(losses[1])
        if mode in ['den', 'final']:
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss_reg_meter.avg))
        elif mode == 'rec':
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss_gen_meter.avg))
        else:
            print('Epoch: {}, Loss_gen: {:.4f}, Loss_reg: {:.4f}'.format(epoch, loss_gen_meter.avg, loss_reg_meter.avg))

        self.sch_reg.step()
        self.sch_gen.step()
        self.sch_gen_cyc.step()

        # validation
        self.model.eval()
        if mode == 'rec':
            loss_rec_meter = AverageMeter()
            with torch.no_grad():
                for batch in track(self.val_loader, description='Epoch: {}, Validating...'.format(epoch), complete_style='dim cyan', total=len(self.val_loader)):
                    loss = self.val_rec_step(batch)
                    loss_rec_meter.update(loss)
            loss_rec = loss_rec_meter.avg

            duration = time.time() - start_time
            self.log('Epoch: {}, Loss_rec: {:.4f}, Time: {:.2f}s'.format(epoch, loss_rec, duration))
            return loss_rec
        
        else:
            mae_meter = AverageMeter()
            mse_meter = AverageMeter()
            with torch.no_grad():
                for batch in track(self.val_loader, description='Epoch: {}, Validating...'.format(epoch), complete_style='dim cyan', total=len(self.val_loader)):
                    mae, mse = self.val_step(batch)
                    mae_meter.update(mae)
                    mse_meter.update(mse)

            mae = mae_meter.avg
            mse = sqrt(mse_meter.avg)

            duration = time.time() - start_time
            self.log('Epoch: {}, MAE: {:.2f}, MSE: {:.2f}, Time: {:.2f}s'.format(epoch, mae, mse, duration))
            return mae, mse
    
    def train(self, mode='dg', gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        self.log('Start training at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        self.load_ckpts(gen_ckpt, reg_ckpt, all_ckpt)

        if mode == 'rec':
            best_loss = 1e10
            best_epoch = 0
            for epoch in range(self.num_epochs):
                loss = self.train_epoch(epoch, mode)
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    self.model.save_sd(os.path.join(self.log_dir, 'best.pth'), mode='gen')
                    self.log('Epoch: {}, Best model saved.'.format(epoch))
                self.model.save_sd(os.path.join(self.log_dir, 'last.pth'), mode='gen')
            self.log('Best Loss: {:.4f} at epoch {}'.format(best_loss, best_epoch))

        else:
            best_mae = 1e10
            best_epoch = 0
            for epoch in range(self.num_epochs):
                mae, _ = self.train_epoch(epoch, mode)
                if mae < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    self.model.save_sd(os.path.join(self.log_dir, 'best.pth'), mode='reg' if mode == 'den' else 'all')
                    self.log('Epoch: {}, Best model saved.'.format(epoch))
                self.model.save_sd(os.path.join(self.log_dir, 'last.pth'), mode='reg' if mode == 'den' else 'all')
            self.log('Best MAE: {:.2f} at epoch {}'.format(best_mae, best_epoch))

        self.log('End training at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    def test(self, mode, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        self.log('Testing...')
        self.load_ckpts(gen_ckpt, reg_ckpt, all_ckpt)

        self.model.eval()
        if mode == 'rec':

            loss_rec_meter = AverageMeter()
            with torch.no_grad():
                for batch in track(self.test_loader, description='Testing...', complete_style='dim cyan', total=len(self.test_loader)):
                    loss = self.val_rec_step(batch)
                    loss_rec_meter.update(loss)
            loss_rec = loss_rec_meter.avg

            self.log('Test Loss: {:.4f}'.format(loss_rec))
        
        else:
            mae_meter = AverageMeter()
            mse_meter = AverageMeter()
            with torch.no_grad():
                for batch in track(self.test_loader, description='Testing...', complete_style='dim cyan', total=len(self.test_loader)):
                    mae, mse = self.val_step(batch)
                    mae_meter.update(mae)
                    mse_meter.update(mse)

            mae = mae_meter.avg
            mse = sqrt(mse_meter.avg)

            self.log('Test MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))

    def visualize(self, mode, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        self.log('Visualizing...')
        self.load_ckpts(gen_ckpt, reg_ckpt, all_ckpt)

        if mode == 'rec':
            raise NotImplementedError

        self.vis_dir = os.path.join(self.log_dir, 'vis')
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.model.eval()
        with torch.no_grad():
            for batch in track(self.test_loader, description='Visualizing...', complete_style='dim cyan', total=len(self.test_loader)):
                if mode in ['den', 'final']:
                    self.visualize_reg_step(batch)
                else:
                    self.visualize_step(batch)

        self.log('Visualized results saved to {}'.format(self.vis_dir))

    def load_ckpts(self, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        if gen_ckpt is not None:
            self.model.load_sd(gen_ckpt, 'gen', device=self.device)
            self.model.load_sd(gen_ckpt, 'gen_cyc', device=self.device)
            self.log('Generator loaded from {}'.format(gen_ckpt))
        if reg_ckpt is not None:
            self.model.load_sd(reg_ckpt, 'reg', device=self.device)
            self.log('Density Regressor loaded from {}'.format(reg_ckpt))
        if all_ckpt is not None:
            self.model.load_sd(all_ckpt, 'all', device=self.device)
            self.log('Model loaded from {}'.format(all_ckpt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--config', type=str, metavar='PATH')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mode', type=str, default='dg', choices=['rec', 'den', 'dg', 'final'])
    parser.add_argument('--gen_ckpt', type=str, metavar='PATH')
    parser.add_argument('--reg_ckpt', type=str, metavar='PATH')
    parser.add_argument('--all_ckpt', type=str, metavar='PATH')
    args = parser.parse_args()

    trainer = DGVCCTrainer(args.config, args.device)
    if args.train:
        trainer.train(args.mode, args.gen_ckpt, args.reg_ckpt, args.all_ckpt)
    if args.test:
        trainer.test(args.mode, args.gen_ckpt, args.reg_ckpt, args.all_ckpt)
    if args.vis:
        trainer.visualize(args.mode, args.gen_ckpt, args.reg_ckpt, args.all_ckpt)