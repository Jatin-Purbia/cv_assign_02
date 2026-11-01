"""Pixel-wise binary cross entropy and explicitly defined soft Dice."""
import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   epsilon: float = 1e-6) -> torch.Tensor:
    p = logits.sigmoid().flatten(1)
    y = target.flatten(1)
    return (1 - (2*(p*y).sum(1) + epsilon)/(p.sum(1)+y.sum(1)+epsilon)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor,
                      dice_weight: float = 0.0) -> torch.Tensor:
    if dice_weight < 0:
        raise ValueError('Dice weight must be nonnegative')
    return F.binary_cross_entropy_with_logits(logits, target) + dice_weight*soft_dice_loss(logits, target)
