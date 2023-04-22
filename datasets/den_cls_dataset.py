import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import random
import os

import sys
sys.path.append('..')

from datasets.den_dataset import DensityMapDataset
from utils.misc import random_crop, get_padding

class DenClsDataset(DensityMapDataset):
    def collate(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        bmaps = torch.stack(transposed_batch[3], 0)
        return images, (points, dmaps, bmaps)

    def __init__(self, root, crop_size, downsample, method, is_grey, unit_size, pre_resize=1, roi_map_path=None, gt_dir=None, gen_root=None):
        super().__init__(root, crop_size, downsample, method, is_grey, unit_size, pre_resize, roi_map_path, gt_dir, gen_root)

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img, img_ext = self._load_img(img_fn)

        basename = img_fn.split('/')[-1].split('.')[0]
        if img_fn.startswith(self.root):
            gt_fn = img_fn.replace(img_ext, '.npy')
        else:
            basename = basename[:-2]
            gt_fn = os.path.join(self.root, 'train', basename + '.npy')
        gt = self._load_gt(gt_fn)

        if self.method == 'train':
            if self.gt_dir is None:
                dmap_fn = gt_fn.replace(basename, basename + '_dmap2')
            else:
                dmap_fn = os.path.join(self.gt_dir, basename + '.npy')
            dmap = self._load_dmap(dmap_fn)
            img, gt, dmap = self._train_transform(img, gt, dmap)
            bmap = dmap.clone().reshape(1, dmap.shape[1]//16, 16, dmap.shape[2]//16, 16).sum(dim=(2, 4))
            bmap = (bmap > 0).float()
            return img, gt, dmap, bmap
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt, basename))
        
if __name__ == '__main__':
    dataset = DenClsDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/sta', 256, 1, 'train', False, 1)

    for i in range(len(dataset)):
        img, gt, dmap, bmap = dataset[i]
        print(bmap.sum())