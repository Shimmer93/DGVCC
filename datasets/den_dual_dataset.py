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

class DensityMapDualDataset(DensityMapDataset):
    def collate(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        return images, (points, dmaps)

    def __init__(self, root, crop_size, downsample, method, is_grey, unit_size, pre_resize=1, roi_map_path=None, gt_dir=None):
        super().__init__(root, crop_size, downsample, method, is_grey, unit_size, pre_resize, roi_map_path, gt_dir)

    def __getitem__(self, index):
        if self.method == 'train':
            another_index = random.randint(0, len(self.img_fns) - 1)
            img1, gt1, dmap1 = super().__getitem__(index)
            img2, gt2, dmap2 = super().__getitem__(another_index)
            img_cat = torch.cat((img1, img2), 1)
            gt_cat = [gt1, gt2]
            dmap_cat = torch.cat((dmap1, dmap2), 1)
            return img_cat, gt_cat, dmap_cat
        else:
            return super().__getitem__(index)