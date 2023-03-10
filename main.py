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

from models.dgvccnet import DGVCCNet
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
        self.model = DGVCCNet(**cfg['model'])
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

        self.sch_reg = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_reg, **cfg['scheduler'])

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write(msg + '\n')

    def compute_loss(self, pred_dmaps, gt_datas):
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

    def train_reg_step(self, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)

        self.opt_reg.zero_grad()

        pred_dmaps = self.model(imgs)
        loss = self.compute_loss(pred_dmaps, gt_datas)
        loss.backward()

        self.opt_reg.step()

        return loss.item()
    
    def train_step(self, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)
        z1 = torch.randn(imgs.size(0), 64, device=self.device)
        z2 = torch.randn(imgs.size(0), 64, device=self.device)

        # train regressor
        self.opt_reg.zero_grad()

        d, d_gen, loss_sim = self.model.forward_dg(imgs, z1, z2, mode='reg')
        loss_den = self.compute_loss(d, gt_datas)
        loss_den_gen = self.compute_loss(d_gen, gt_datas)
        loss_reg = loss_den + loss_den_gen + 100 * loss_sim
        loss_reg.backward()

        self.opt_reg.step()

        # train generator
        self.opt_gen.zero_grad()
        self.opt_gen_cyc.zero_grad()

        d_gen, loss_cyc, loss_div, loss_ortho = self.model.forward_dg(imgs, z1, z2, mode='gen')
        loss_den_gen2 = self.compute_loss(d_gen, gt_datas)
        loss_gen = 10 * loss_den_gen2 + 100 * loss_cyc + 100 * loss_div + 10 * loss_ortho
        loss_gen.backward()

        self.opt_gen.step()
        self.opt_gen_cyc.step()

        return loss_reg.item(), loss_gen.item()
    
    def val_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)

            for patch in img_patches:
                pred = self.model(patch)
                pred_count += pred.sum().cpu().item() / self.log_para

        else:
            pred = self.model(img)
            pred_count = pred.sum().cpu().item() / self.log_para

        gt_count = gt.shape[1]

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2

        return mae, mse
    
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

    def train_epoch(self, epoch, only_reg=False):
        start_time = time.time()

        # training
        self.model.train()
        loss_reg_meter = AverageMeter()
        loss_gen_meter = AverageMeter()
        for batch in track(self.train_loader, description='Epoch: {}, Training...'.format(epoch), complete_style='dim cyan', total=len(self.train_loader)):
            if only_reg:
                loss = self.train_reg_step(batch)
                loss_reg_meter.update(loss)
            else:
                losses = self.train_step(batch)
                loss_reg_meter.update(losses[0])
                loss_gen_meter.update(losses[1])
        if only_reg:
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss_reg_meter.avg))
        else:
            print('Epoch: {}, Loss_reg: {:.4f}, Loss_gen: {:.4f}'.format(epoch, loss_reg_meter.avg, loss_gen_meter.avg))

        self.sch_reg.step()

        # validation
        self.model.eval()
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
    
    def train(self, only_reg=False, ckpt=None):
        self.log('Start training at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        self.load_ckpt(ckpt)

        best_mae = 1e10
        best_epoch = 0
        for epoch in range(self.num_epochs):
            mae, mse = self.train_epoch(epoch, only_reg=only_reg)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best.pth'))
                self.log('Epoch: {}, Best model saved.'.format(epoch))
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'last.pth'))

        self.log('Best MAE: {:.2f} at epoch {}'.format(best_mae, best_epoch))
        self.log('End training at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    def test(self, ckpt=None):
        self.load_ckpt(ckpt)

        self.model.eval()
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

    def visualize(self, only_reg=False, ckpt=None):
        print('Visualizing...')
        self.load_ckpt(ckpt)
        print('What is the problem?')

        self.vis_dir = os.path.join(self.log_dir, 'vis')
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.model.eval()
        with torch.no_grad():
            for batch in track(self.test_loader, description='Visualizing...', complete_style='dim cyan', total=len(self.test_loader)):
                if only_reg:
                    self.visualize_reg_step(batch)
                else:
                    self.visualize_step(batch)

        self.log('Visualized results saved to {}'.format(self.vis_dir))

    def load_ckpt(self, ckpt):
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=True)
            self.log('Model loaded from {}'.format(ckpt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--config', type=str, metavar='PATH')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, metavar='PATH')
    parser.add_argument('--only_reg', action='store_true', default=False)
    args = parser.parse_args()

    trainer = DGVCCTrainer(args.config, args.device)
    if args.train:
        trainer.train(only_reg=args.only_reg, ckpt=args.ckpt)
    elif args.test:
        trainer.test(ckpt=args.ckpt)
    elif args.vis:
        trainer.visualize(only_reg=args.only_reg, ckpt=args.ckpt)