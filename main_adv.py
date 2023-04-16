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

from trainers.dgtrainer import DGTrainer
from trainers.dgtrainer2 import DGTrainer2
from models.dgvcc_new import DGNet
from models.dgvcc_new2 import DGNet2
from losses.bl import BL
from datasets.den_dataset import DensityMapDataset
from datasets.bay_dataset import BayesianDataset
from datasets.dual_dataset import DualDataset
from utils.misc import divide_img_into_patches, denormalize, AverageMeter, DictAvgMeter, seed_everything, get_current_datetime

def get_model(name, params):
    model = DGNet2(**params)
    return model

def get_loss(name, params):
    if name == 'bl':
        loss = BL(**params)
    elif name == 'mse':
        loss = nn.MSELoss()
    else:
        raise ValueError('Unknown loss: {}'.format(name))
    return loss

def get_dataset(name, params, method):
    if name == 'den':
        dataset = DensityMapDataset(method=method, **params)
    elif name == 'bay':
        dataset = BayesianDataset(method=method, **params)
    elif name == 'dual':
        dataset = DualDataset(method=method, **params)
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return dataset

def get_optimizer(name, params, model):
    if name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **params)
    elif name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError('Unknown optimizer: {}'.format(name))
    return optimizer

def get_scheduler(name, params, optimizer):
    if name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif name == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)
    else:
        raise ValueError('Unknown scheduler: {}'.format(name))
    return scheduler

def load_config(config_path, task):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    init_params = {}
    task_params = {}

    init_params['seed'] = cfg['seed']
    init_params['version'] = cfg['version']
    init_params['device'] = cfg['device']
    init_params['mode'] = cfg['mode']
    init_params['log_para'] = cfg['log_para']

    task_params['model'] = get_model(cfg['model']['name'], cfg['model']['params'])
    task_params['checkpoint'] = cfg['checkpoint']
    
    if task == 'train':
        task_params['loss'] = get_loss(cfg['loss']['name'], cfg['loss']['params'])
        train_dataset = get_dataset(cfg['train_dataset']['name'], cfg['train_dataset']['params'], method='train')
        task_params['train_dataloader'] = DataLoader(train_dataset, collate_fn=train_dataset.__class__.collate, **cfg['train_loader'])
        val_dataset = get_dataset(cfg['val_dataset']['name'], cfg['val_dataset']['params'], method='val')
        task_params['val_dataloader'] = DataLoader(val_dataset, **cfg['val_loader'])

        if cfg['mode'] == 'joint':
            opt_reg = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'].reg)
            opt_gen = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'].gen)
            opt_cyc = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'].gen_cyc)
            task_params['optimizer'] = [opt_reg, opt_gen, opt_cyc]
            sch_reg = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], opt_reg)
            sch_gen = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], opt_gen)
            sch_cyc = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], opt_cyc)
            task_params['scheduler'] = [sch_reg, sch_gen, sch_cyc]
        else:
            task_params['optimizer'] = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'])
            task_params['scheduler'] = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], task_params['optimizer'])
        
        task_params['num_epochs'] = cfg['num_epochs']

    else:
        test_dataset = get_dataset(cfg['test_dataset']['name'], cfg['test_dataset']['params'], method='test')
        task_params['test_dataloader'] = DataLoader(test_dataset, **cfg['test_loader'])

    return init_params, task_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dg.yaml', help='path to config file')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'vis'], help='task to perform')
    args = parser.parse_args()

    init_params, task_params = load_config(args.config, args.task)

    trainer = DGTrainer2(**init_params)
    os.system(f'cp {args.config} {trainer.log_dir}')

    if args.task == 'train':
        trainer.train(**task_params)
    elif args.task == 'test':
        trainer.test(**task_params)
    elif args.task == 'vis':
        trainer.vis(**task_params)
