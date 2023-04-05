import torch

def triplet_loss(anchor, positive, negative, margin=0.5):
    """Compute the triplet loss given an anchor, a positive and a negative sample.
    Args:
        anchor (torch.Tensor): anchor sample.
        positive (torch.Tensor): positive sample.
        negative (torch.Tensor): negative sample.
        margin (float): margin for triplet loss.
    Returns:
        torch.Tensor: triplet loss.
    """
    d_p = torch.sum((anchor - positive) ** 2, dim=1)
    d_n = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.clamp(margin + d_p - d_n, min=0.0)
    return loss.mean()

class TripletLoss(torch.nn.Module):
    """Triplet loss.
    Args:
        margin (float): margin for triplet loss.
    """
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """Compute the triplet loss given an anchor, a positive and a negative sample.
        Args:
            anchor (torch.Tensor): anchor sample.
            positive (torch.Tensor): positive sample.
            negative (torch.Tensor): negative sample.
        Returns:
            torch.Tensor: triplet loss.
        """
        return triplet_loss(anchor, positive, negative, self.margin)