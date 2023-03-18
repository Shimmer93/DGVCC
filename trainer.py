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

from models.dgvcc import DGVCCModel, ModelComponent
from losses.bl import BL
from datasets.den_dataset import DensityMapDataset
from datasets.bay_dataset import BayesianDataset

from utils.misc import divide_img_into_patches, denormalize, AverageMeter, seed_everything, get_current_datetime

class DGMode(Enum):
    GENERATOR = 1
    REGRESSOR = 2
    JOINT = 3
    AUGMENTED = 4

class DGVCCTrainer():
    def __init__(self, config, device):
        with open(config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

        seed_everything(cfg['seed'])

        self.device = torch.device(device)
        self.model = DGVCCModel(**cfg['model'])
        self.model.to(self.device)

        self.fixed_z1 = torch.randn(1, 64, device=self.device)
        self.fixed_z2 = torch.randn(1, 64, device=self.device)
        self.fixed_z3 = torch.randn(1, 64, device=self.device)

        self.mode = DGMode[cfg['mode']]

        if self.mode == DGMode.GENERATOR:
            self.component = ModelComponent.GENERATOR
        elif self.mode == DGMode.REGRESSOR:
            self.component = ModelComponent.REGRESSOR
        else:
            self.component = ModelComponent.ALL

        self.method = cfg['method']
        if self.method == 'Density':
            self.den_loss = nn.MSELoss()
            DatasetType = DensityMapDataset
        elif self.method == 'Bayesian':
            self.den_loss = BL(**cfg['bl_loss'])
            DatasetType = BayesianDataset
        else:
            raise NotImplementedError
        
        train_dataset = DatasetType(method='train', **cfg['train_dataset'])
        val_dataset = DatasetType(method='test', **cfg['train_dataset'])
        test_dataset = DatasetType(method='test', **cfg['test_dataset'])
        collate_fn = DatasetType.collate
            
        self.train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)

        self.downsample = cfg['train_dataset']['downsample']
        self.num_epochs = cfg['num_epochs']
        self.log_para = cfg['log_para']

        self.version = cfg['version']
        self.log_dir = os.path.join('logs', self.version)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        os.system('cp {} {}'.format(config, self.log_dir))

        self.opt = torch.optim.AdamW(self.model.get_params(ModelComponent.REGRESSOR if self.mode == DGMode.AUGMENTED else self.component), **cfg['optimizer'])
        self.sch = torch.optim.lr_scheduler.StepLR(self.opt, **cfg['scheduler'])

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write(msg + '\n')

    def load_ckpts(self, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        if gen_ckpt is not None:
            self.model.load_sd(gen_ckpt, ModelComponent.GENERATOR, device=self.device)
            self.model.load_sd(gen_ckpt, ModelComponent.GENERATOR_CYCLIC, device=self.device)
            self.log('Generator loaded from {}'.format(gen_ckpt))
        if reg_ckpt is not None:
            self.model.load_sd(reg_ckpt, ModelComponent.REGRESSOR, device=self.device)
            self.log('Density Regressor loaded from {}'.format(reg_ckpt))
        if all_ckpt is not None:
            self.model.load_sd(all_ckpt, ModelComponent.ALL, device=self.device)
            self.log('Model loaded from {}'.format(all_ckpt))

    def compute_count_loss(self, pred_dmaps, gt_datas):
        if self.method == 'Density':
            _, gt_dmaps = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            loss = self.den_loss(pred_dmaps, gt_dmaps * self.log_para)

        elif self.method == 'Bayesian':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            targs = [targ.to(self.device) for targ in targs]
            st_sizes = st_sizes.to(self.device)
            loss = self.den_loss(gts, st_sizes, targs, pred_dmaps)
        
        else:
            raise NotImplementedError

        return loss
    
    def compute_count_loss_with_aug(self, pred_dmaps, gt_datas, num_aug_samples=3):
        num_dmaps = 1 + num_aug_samples
        if self.method == 'Density':
            _, gt_dmaps = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            gt_dmaps = gt_dmaps.repeat(num_dmaps, 1, 1, 1)
            loss = self.den_loss(pred_dmaps, gt_dmaps * self.log_para)
        elif self.method == 'Bayesian':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            gts = gts * num_dmaps
            targs = [targ.to(self.device) for targ in targs]
            targs = targs * num_dmaps
            st_sizes = st_sizes.repeat(num_dmaps)
            loss = self.den_loss(gts, st_sizes, targs, pred_dmaps)
        else:
            raise NotImplementedError
        
        return loss
    
    def train_step(self, batch):
        imgs, gt_datas = batch
        imgs = imgs.to(self.device)

        self.opt.zero_grad()

        if self.mode == DGMode.GENERATOR:
            _, loss = self.model.forward_generator(imgs)
        elif self.mode == DGMode.REGRESSOR:
            pred_dmaps = self.model.forward_regressor(imgs)
            loss = self.compute_count_loss(pred_dmaps, gt_datas)
        elif self.mode == DGMode.JOINT:
            z1 = torch.randn(imgs.size(0), 64, device=self.device)
            z2 = torch.randn(imgs.size(0), 64, device=self.device)
            d_cat, loss_cyc, loss_div, loss_sim, loss_dissim = self.model.forward_joint(imgs, z1, z2)
            loss_den_cat = self.compute_count_loss_with_aug(d_cat, gt_datas, num_aug_samples=1)
            loss = loss_den_cat + 10 * loss_cyc + 10 * loss_div + 1000 * loss_sim + 100 * loss_dissim
        elif self.mode == DGMode.AUGMENTED:
            z1 = torch.randn(imgs.size(0), 64, device=self.device)
            z2 = torch.randn(imgs.size(0), 64, device=self.device)
            z3 = torch.randn(imgs.size(0), 64, device=self.device)
            _, d_cat = self.model.forward_augmented(imgs, z1, z2, z3)
            loss = self.compute_count_loss_with_aug(d_cat, gt_datas, num_aug_samples=3)
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def val_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        if self.mode == DGMode.GENERATOR:
            _, loss = self.model.forward_generator(img)
            loss = loss.cpu().item()
            return {'loss': loss}
        elif self.mode == DGMode.REGRESSOR or self.mode == DGMode.JOINT:
            pred_dmap = self.model.forward_regressor(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        elif self.mode == DGMode.AUGMENTED:
            _, pred_dmap_cat = self.model.forward_augmented(img, self.fixed_z1, self.fixed_z2, self.fixed_z3)
            pred_count_avg = pred_dmap_cat.sum().cpu().item() / (self.log_para * 4)
            gt_count = gt.shape[1]
            mae = np.abs(pred_count_avg - gt_count)
            mse = (pred_count_avg - gt_count) ** 2
            return {'mae': mae, 'mse': mse}
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))
        
    def test_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in testing'

        if self.mode == DGMode.GENERATOR:
            _, loss = self.model.forward_generator(img)
            loss = loss.cpu().item()
            return {'loss': loss}
        else:
            pred_dmap = self.model.forward_regressor(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            mae = np.abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            return {'mae': mae, 'mse': mse}

    def vis_step(self, batch):
        img, gt, name = batch
        img = img.to(self.device)
        b = img.shape[0]

        assert b == 1, 'batch size should be 1 in visualization'

        if self.mode == DGMode.GENERATOR:
            rec_img, _ = self.model.forward_generator(img)
            img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
            rec_img = denormalize(rec_img)[0].cpu().permute(1, 2, 0).numpy()
            
            fig = plt.figure(figsize=(20, 10))
            ax_orig = fig.add_subplot(1, 2, 1)
            ax_orig.set_title('GT')
            ax_orig.imshow(img)
            ax_rec = fig.add_subplot(1, 2, 2)
            ax_rec.set_title('Rec')
            ax_rec.imshow(rec_img)

            plt.savefig(os.path.join(self.vis_dir, '{}.png'.format(name[0])))
            plt.close()

        elif self.mode == DGMode.REGRESSOR:
            pred_dmap = self.model.forward_regressor(img)
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            gt_count = gt.shape[1]
            img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
            pred_dmap = pred_dmap[0,0].cpu().numpy()

            fig = plt.figure(figsize=(20, 10))
            ax_img = fig.add_subplot(1, 2, 1)
            ax_img.set_title('GT: {}'.format(gt_count))
            ax_img.imshow(img)
            ax_den = fig.add_subplot(1, 2, 2)
            ax_den.set_title('Pred: {}'.format(pred_count))
            ax_den.imshow(pred_dmap)

            plt.savefig(os.path.join(self.vis_dir, '{}.png'.format(name[0])))
            plt.close()
        
        elif self.mode == DGMode.JOINT:
            pred_dmap, pred_dmap_gen, img_gen, img_gen2, img_cyc = self.model.forward_test(img)
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

        elif self.mode == DGMode.AUGMENTED:
            img_cat, pred_dmap_cat = self.model.forward_augmented(img, self.fixed_z1, self.fixed_z2, self.fixed_z3)
            img_cat = denormalize(img_cat)
            img_orig = img_cat[0].cpu().permute(1, 2, 0).numpy()
            img_gen1 = img_cat[1].cpu().permute(1, 2, 0).numpy()
            img_gen2 = img_cat[2].cpu().permute(1, 2, 0).numpy()
            img_gen3 = img_cat[3].cpu().permute(1, 2, 0).numpy()
            pred_dmap_orig = pred_dmap_cat[0,0].cpu().numpy()
            pred_dmap_gen1 = pred_dmap_cat[1,0].cpu().numpy()
            pred_dmap_gen2 = pred_dmap_cat[2,0].cpu().numpy()
            pred_dmap_gen3 = pred_dmap_cat[3,0].cpu().numpy()
            pred_count_orig = pred_dmap_orig.sum() / self.log_para
            pred_count_gen1 = pred_dmap_gen1.sum() / self.log_para
            pred_count_gen2 = pred_dmap_gen2.sum() / self.log_para
            pred_count_gen3 = pred_dmap_gen3.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img_orig, img_gen1, img_gen2, img_gen3, pred_dmap_orig, pred_dmap_gen1, pred_dmap_gen2, pred_dmap_gen3]
            titles = [f'Original: {gt_count}', 'Generated1', 'Generated2', 'Generated3', f'Pred: {pred_count_orig:.2f}', 
                      f'Pred: {pred_count_gen1:.2f}', f'Pred: {pred_count_gen2:.2f}', f'Pred: {pred_count_gen3:.2f}']
            
            fig = plt.figure(figsize=(25, 10))
            for i in range(8):
                ax = fig.add_subplot(2, 4, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])
            
            plt.savefig(os.path.join(self.vis_dir, '{}.png'.format(name[0])))
            plt.close()

    def train_epoch(self, epoch):
        start_time = time.time()

        self.model.train()
        loss_meter = AverageMeter()
        for batch in track(self.train_loader, description='Epoch: {}, Training...'.format(epoch), \
                           complete_style='dim cyan', total=len(self.train_loader)):
            loss = self.train_step(batch)
            loss_meter.update(loss)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, loss_meter.avg))

        self.sch.step()

        self.model.eval()
        if self.mode == DGMode.GENERATOR:
            val_loss_meter = AverageMeter()
        else:
            mae_meter = AverageMeter()
            mse_meter = AverageMeter()

        with torch.no_grad():
            for batch in track(self.val_loader, description='Epoch: {}, Validating...'.format(epoch), \
                            complete_style='dim cyan', total=len(self.val_loader)):
                result = self.val_step(batch)

                if self.mode == DGMode.GENERATOR:
                    val_loss_meter.update(result['loss'])
                else:
                    mae_meter.update(result['mae'])
                    mse_meter.update(result['mse'])
                duration = time.time() - start_time

        if self.mode == DGMode.GENERATOR:
            result = {'loss': val_loss_meter.avg}
            self.log('Epoch: {}, Val Loss: {:.4f}, Time: {:.2f}'.format(epoch, result['loss'], duration))
        else:
            result = {'mae': mae_meter.avg, 'mse': np.sqrt(mse_meter.avg)}
            self.log('Epoch: {}, Val MAE: {:.2f}, MSE: {:.2f}, Time: {:.2f}'.format(epoch, result['mae'], result['mse'], duration))

        return result
    
    def train(self, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        self.log('Start training at {}'.format(get_current_datetime()))
        self.load_ckpts(gen_ckpt, reg_ckpt, all_ckpt)

        best_criterion = 1e10
        best_epoch = -1

        for epoch in range(self.num_epochs):
            result = self.train_epoch(epoch)
            if self.mode == DGMode.GENERATOR:
                criterion = result['loss']
            else:
                criterion = result['mae']
            
            if criterion < best_criterion:
                best_criterion = criterion
                best_epoch = epoch
                
                self.model.save_sd(os.path.join(self.log_dir, 'best.pth'), self.component)
                self.log('Epoch: {}, Best model saved!'.format(epoch))
            self.model.save_sd(os.path.join(self.log_dir, 'last.pth'), self.component)

        if self.mode == DGMode.GENERATOR:
            self.log('Best epoch: {}, Best Val Loss: {:.4f}'.format(best_epoch, best_criterion))
        else:
            self.log('Best epoch: {}, Best Val MAE: {:.2f}'.format(best_epoch, best_criterion))

        self.log('Training results saved to {}'.format(self.log_dir))
        self.log('End training at {}'.format(get_current_datetime()))

    def test(self, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        self.log('Start testing at {}'.format(get_current_datetime()))
        self.load_ckpts(gen_ckpt, reg_ckpt, all_ckpt)

        self.model.eval()
        if self.mode == DGMode.GENERATOR:
            test_loss_meter = AverageMeter()
        else:
            mae_meter = AverageMeter()
            mse_meter = AverageMeter()

        with torch.no_grad():
            for batch in track(self.test_loader, description='Testing...', \
                            complete_style='dim cyan', total=len(self.test_loader)):
                result = self.test_step(batch)

                if self.mode == DGMode.GENERATOR:
                    test_loss_meter.update(result['loss'])
                else:
                    mae_meter.update(result['mae'])
                    mse_meter.update(result['mse'])

        if self.mode == DGMode.GENERATOR:
            result = {'loss': test_loss_meter.avg}
            self.log('Test Loss: {:.4f}'.format(result['loss']))
        else:
            result = {'mae': mae_meter.avg, 'mse': np.sqrt(mse_meter.avg)}
            self.log('Test MAE: {:.2f}, MSE: {:.2f}'.format(result['mae'], result['mse']))

        self.log('Testing results saved to {}'.format(self.log_dir))
        self.log('End testing at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    def visualize(self, gen_ckpt=None, reg_ckpt=None, all_ckpt=None):
        self.log('Start Visualizing at {}'.format(get_current_datetime()))
        self.load_ckpts(gen_ckpt, reg_ckpt, all_ckpt)

        self.vis_dir = os.path.join(self.log_dir, 'vis')
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.model.eval()
        with torch.no_grad():
            for batch in track(self.test_loader, description='Visualizing...', 
                               complete_style='dim cyan', total=len(self.test_loader)):
                self.vis_step(batch)

        self.log('Visualized results saved to {}'.format(self.vis_dir))
        self.log('End Visualizing at {}'.format(get_current_datetime()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--config', type=str, metavar='PATH')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_ckpt', type=str, metavar='PATH')
    parser.add_argument('--reg_ckpt', type=str, metavar='PATH')
    parser.add_argument('--all_ckpt', type=str, metavar='PATH')
    args = parser.parse_args()

    trainer = DGVCCTrainer(args.config, args.device)
    if args.train:
        trainer.train(args.gen_ckpt, args.reg_ckpt, args.all_ckpt)
    if args.test:
        trainer.test(args.gen_ckpt, args.reg_ckpt, args.all_ckpt)
    if args.vis:
        trainer.visualize(args.gen_ckpt, args.reg_ckpt, args.all_ckpt)