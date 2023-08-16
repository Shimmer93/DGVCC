"""
Network Initializations
"""

import logging
import importlib
import torch
import torch.nn as nn

from . import Resnet
from .cov_settings import CovMatrix_ISW, CovMatrix_IRW
from .instance_whitening import instance_whitening_loss, get_covariance_matrix
from .mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights
# from .deepv3 import *

# class ISWCounter_ResNet(DeepV3Plus):
#     def __init__(self, variant='D', skip='m1', skip_num=48):
#         super(ISWCounter_ResNet, self).__init__(num_classes=1, trunk='resnet-50', 
#             criterion=nn.MSELoss().cuda(), criterion_aux=nn.MSELoss().cuda(), variant=variant, skip=skip, skip_num=skip_num, wt_layer=[0,0,2,2,2,0,0], use_wtloss=True, relax_denom=0)

class ISWCounter_ResNet(nn.Module):
    def __init__(self, criterion=nn.MSELoss().cuda(),
                variant='D', skip='m1', skip_num=48, wt_layer=[0,0,2,2,2,0,0], use_wtloss=True, relax_denom=2.0, clusters=3):
        super(ISWCounter_ResNet, self).__init__()

        self.criterion = criterion
        self.variant = variant
        self.wt_layer = wt_layer
        self.use_wtloss = use_wtloss
        self.relax_denom = relax_denom
        self.clusters = clusters

        self.eps = 1e-5
        self.whitening = False

        resnet = Resnet.resnet50(wt_layer=self.wt_layer)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = \
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=16)
        )

        in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]
        self.cov_matrix_layer = []
        self.cov_type = []
        for i in range(len(self.wt_layer)):
            if self.wt_layer[i] > 0:
                self.whitening = True
                if self.wt_layer[i] == 1:
                    self.cov_matrix_layer.append(CovMatrix_IRW(dim=in_channel_list[i], relax_denom=self.relax_denom))
                    self.cov_type.append(self.wt_layer[i])
                elif self.wt_layer[i] == 2:
                    self.cov_matrix_layer.append(CovMatrix_ISW(dim=in_channel_list[i], relax_denom=self.relax_denom, clusters=self.clusters))
                    self.cov_type.append(self.wt_layer[i])

    def set_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].set_mask_matrix()

    def reset_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].reset_mask_matrix()

    def forward(self, x, gts=None, cal_covstat=False, apply_wtloss=True):
        w_arr = []

        if cal_covstat:
            x = torch.cat(x, dim=0)

        x = self.layer0[0](x)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x, w = self.layer0[1](x)
            w_arr.append(w)
        else:
            x = self.layer0[1](x)
        x = self.layer0[2](x)
        x = self.layer0[3](x)

        x_tuple = self.layer1([x, w_arr])  # 400

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        # x_tuple = self.layer4(x_tuple)  # 100
        x = x_tuple[0]
        w_arr = x_tuple[1]

        if cal_covstat:
            for index, f_map in enumerate(w_arr):
                # Instance Whitening
                B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
                HW = H * W
                f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
                eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
                off_diag_elements = f_cor * reverse_eye
                #print("here", off_diag_elements.shape)
                self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))
            return 0
        
        main_out = self.head(x)

        if self.training:
            loss1 = self.criterion(main_out, gts * 1000)

            if self.use_wtloss:
                wt_loss = torch.FloatTensor([0]).cuda()
                if apply_wtloss:
                    for index, f_map in enumerate(w_arr):
                        eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                        loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)
                        wt_loss = wt_loss + loss
                wt_loss = wt_loss / len(w_arr)

            return [loss1, wt_loss]
        else:
            return main_out
        
if __name__ == '__main__':
    m = ISWCounter_ResNet()
    x = torch.randn(4, 3, 320, 320)
    y = m(x)