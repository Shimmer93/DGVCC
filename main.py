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

from trainers.advtrainer import AdvTrainer
from models.models import get_models
from losses.bl import BL
from datasets.den_dataset import DensityMapDataset
from datasets.den_cls_dataset import DenClsDataset
from datasets.bay_dataset import BayesianDataset
from datasets.jhu_domain_dataset import JHUDomainDataset
from datasets.jhu_domain_cls_dataset import JHUDomainClsDataset
from utils.misc import divide_img_into_patches, denormalize, AverageMeter, DictAvgMeter, seed_everything, get_current_datetime

def get_model(name, params, mode):
    gen, reg = get_models()
    if mode == 'generation':
        return gen
    else:
        return reg

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
        collate = DensityMapDataset.collate
    elif name == 'den_cls':
        dataset = DenClsDataset(method=method, **params)
        collate = DenClsDataset.collate
    elif name == 'bay':
        dataset = BayesianDataset(method=method, **params)
        collate = BayesianDataset.collate
    elif name == 'jhu_domain':
        dataset = JHUDomainDataset(method=method, **params)
        collate = JHUDomainDataset.collate
    elif name == 'jhu_domain_cls':
        dataset = JHUDomainClsDataset(method=method, **params)
        collate = JHUDomainClsDataset.collate
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return dataset, collate

def get_optimizer(name, params, model):
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), **params)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), **params)
    elif name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError('Unknown optimizer: {}'.format(name))

def get_scheduler(name, params, optimizer):
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif name == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)
    else:
        raise ValueError('Unknown scheduler: {}'.format(name))

def load_config(config_path, task):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    init_params = {}
    task_params = {}

    init_params['seed'] = cfg['seed']
    init_params['version'] = cfg['version']
    init_params['device'] = cfg['device']
    init_params['log_para'] = cfg['log_para']
    init_params['mode'] = cfg['mode']

    gen, reg = get_models()
    if cfg['mode'] == 'generation':
        task_params['model'] = gen
    else:
        task_params['model'] = reg

    task_params['checkpoint'] = cfg['checkpoint']
    
    if task == 'train' or task == 'train_test':
        task_params['loss'] = get_loss(cfg['loss']['name'], cfg['loss']['params'])
        train_dataset, collate = get_dataset(cfg['train_dataset']['name'], cfg['train_dataset']['params'], method='train')
        task_params['train_dataloader'] = DataLoader(train_dataset, collate_fn=collate, **cfg['train_loader'])
        val_dataset, _ = get_dataset(cfg['val_dataset']['name'], cfg['val_dataset']['params'], method='val')
        task_params['val_dataloader'] = DataLoader(val_dataset, **cfg['val_loader'])
        task_params['optimizer'] = get_optimizer(cfg['optimizer']['name'], cfg['optimizer']['params'], task_params['model'])
        task_params['scheduler'] = get_scheduler(cfg['scheduler']['name'], cfg['scheduler']['params'], task_params['optimizer'])
        task_params['num_epochs'] = cfg['num_epochs']

    if task != 'train':
        test_dataset, _ = get_dataset(cfg['test_dataset']['name'], cfg['test_dataset']['params'], method='test')
        task_params['test_dataloader'] = DataLoader(test_dataset, **cfg['test_loader'])

    return init_params, task_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dg.yaml', help='path to config file')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'vis', 'train_test'], help='task to perform')
    args = parser.parse_args()

    init_params, task_params = load_config(args.config, args.task)

    trainer = AdvTrainer(**init_params)
    os.system(f'cp {args.config} {trainer.log_dir}')

    if args.task == 'train':
        trainer.train(**task_params)
    elif args.task == 'test':
        trainer.test(**task_params)
    elif args.task == 'vis':
        trainer.vis(**task_params)
    elif args.task == 'train_test':
        trainer.train_and_test(**task_params)
