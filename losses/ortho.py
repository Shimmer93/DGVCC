import torch
import torch.nn as nn
import torch.nn.functional as F

def ortho_loss(x, y):
    # x: (C, P)
    # y: (C, P)
    # return: scalar
    gram_matrix = torch.matmul(x, y.t()) # (C, C)
    off_diag_entries = torch.triu(gram_matrix, diagonal=1) # (C, C)
    loss = torch.mean(torch.square(off_diag_entries))
    return loss