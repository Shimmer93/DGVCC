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

class DualDataset(DensityMapDataset):

    def collate(batch):
        transposed_batch = list(zip(*batch))
        images1 = torch.stack(transposed_batch[0], 0)
        images2 = torch.stack(transposed_batch[1], 0)
        points1 = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
        points2 = transposed_batch[3]  # the number of points is not fixed, keep it as a list of tensor
        dmaps1 = torch.stack(transposed_batch[4], 0)
        dmaps2 = torch.stack(transposed_batch[5], 0)
        return (images1, images2), (points1, points2, dmaps1, dmaps2)

    def __init__(self, root, crop_size, downsample, method, is_grey, unit_size, pre_resize=1, roi_map_path=None, gt_dir=None, gen_root=None):
        super().__init__(root, crop_size, downsample, method, is_grey, unit_size, pre_resize, roi_map_path, gt_dir, gen_root)

    def __getitem__(self, index):
        another_index = random.randint(0, len(self.img_fns) - 1)

        data1 = super().__getitem__(index)
        data2 = super().__getitem__(another_index)

        if self.method == 'train':
            img1, gt1, dmap1 = data1
            img2, gt2, dmap2 = data2
            return img1, img2, gt1, gt2, dmap1, dmap2
        else:
            img1, gt1, name1, padding1 = data1
            img2, gt2, name2, padding2 = data2
            return (img1, img2), (gt1, gt2), (name1, name2), (padding1, padding2)