import torch
import torch.nn as nn
import torch.nn.functional as F

def lw_loss(x, mask=None):
    # Instance Whitening Loss
    # x: (N, C, H, W)
    # mask: (N, 1, H, W)
    # return: scalar
    N, C, H, W = x.shape
    x = x.view(N, C, -1)
    x = x - torch.mean(x, dim=2, keepdim=True)
    x = x / torch.sqrt(torch.var(x, dim=2, keepdim=True) + 1e-5)
    if mask is not None:
        x = x * mask.view(N, 1, -1)
    gram_matrix = torch.matmul(x, x.transpose(1, 2))
    off_diag_entries = torch.triu(gram_matrix, diagonal=1)
    loss = torch.sum(torch.square(off_diag_entries))
    return loss