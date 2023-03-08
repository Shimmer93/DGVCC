import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import models
import yaml
import time
from rich.progress import track
import argparse

from models.dgvccnet import DGVCCNet
from datasets.den_dataset import DensityMapDataset

from utils.misc import divide_img_into_patches, denormalize, AverageMeter, seed_everything

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    dmaps = torch.stack(transposed_batch[2], 0)
    return images, points, dmaps

class DGVCCTrainer():
    def __init__(self, config, device):
        with open(config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

        seed_everything(cfg['seed'])

        self.device = torch.device(device)
        self.model = DGVCCNet(**cfg['model'])
        self.model.to(self.device)

        train_dataset = DensityMapDataset(method='train', **cfg['train_dataset'])
        val_dataset = DensityMapDataset(method='val', **cfg['train_dataset'])
        test_dataset = DensityMapDataset(method='test', **cfg['test_dataset'])
        self.train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, collate_fn=train_collate)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)

        self.den_loss = nn.MSELoss()

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

    def train_reg_step(self, batch):
        imgs, gts, dmaps = batch
        imgs = imgs.to(self.device)
        dmaps = dmaps.to(self.device)

        self.opt_reg.zero_grad()

        pred_dmaps = self.model(imgs)
        loss = self.den_loss(pred_dmaps, dmaps * self.log_para)
        loss.backward()

        self.opt_reg.step()

        return loss.item()
    
    def train_step(self, batch):
        imgs, gts, dmaps = batch
        imgs = imgs.to(self.device)
        dmaps = dmaps.to(self.device)
        z1 = torch.randn(imgs.size(0), 64, device=self.device)
        z2 = torch.randn(imgs.size(0), 64, device=self.device)

        # train regressor
        self.opt_reg.zero_grad()

        d, d_gen, loss_sim = self.model.forward_dg(imgs, z1, z2, mode='reg')
        loss_den = self.loss(d, dmaps * self.log_para)
        loss_den_gen = self.loss(d_gen, dmaps * self.log_para)
        loss_reg = loss_den + loss_den_gen + 10 * loss_sim
        loss_reg.backward()

        self.opt_reg.step()

        # train generator
        self.opt_gen.zero_grad()
        self.opt_gen_cyc.zero_grad()

        d_gen, loss_cyc, loss_div, loss_ortho = self.model.forward_dg(imgs, z1, z2, mode='gen')
        loss_den_gen2 = self.loss(d_gen, dmaps * self.log_para)
        loss_gen = loss_den_gen2 + 10 * loss_cyc + 100 * loss_div + 0.001 * loss_ortho
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
        mse = mse_meter.avg

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
            for batch in track(self.test_loader, description='Testing...'):
                mae, mse = self.val_step(batch)
                mae_meter.update(mae)
                mse_meter.update(mse)

        mae = mae_meter.avg
        mse = mse_meter.avg

        self.log('MAE: {:.2f}, MSE: {:.2f}'.format(mae, mse))

    def load_ckpt(self, ckpt):
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.log('Model loaded from {}'.format(ckpt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
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