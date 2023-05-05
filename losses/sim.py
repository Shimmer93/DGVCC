import torch
import torch.nn as nn
import torch.nn.functional as F

def sim_loss(x, y):
    x_ = x.view(x.shape[0], x.shape[1], -1)
    y_ = y.view(y.shape[0], y.shape[1], -1)
    recon_sim = torch.bmm(y_.transpose(1, 2), x_)
    sim_gt = torch.linspace(0, y.shape[2] * y.shape[3] - 1,
                            y.shape[2] * y.shape[3]).unsqueeze(0).repeat(y.shape[0], 1).to(y.device)
    loss = F.cross_entropy(recon_sim, sim_gt.long())

    return loss
