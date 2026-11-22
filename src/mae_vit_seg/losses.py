"""CE + classwise soft Dice without a large one-hot tensor."""
import torch
from torch.nn import functional as F


def cross_entropy_dice(logits: torch.Tensor, targets: torch.Tensor,
                       ignore_index: int = 255, dice_weight: float = 1.0):
    classes = logits.shape[1]
    valid = (targets != ignore_index) & (targets >= 0) & (targets < classes)
    if not valid.any():
        return logits.sum() * 0.0
    clean = targets.clamp(min=0, max=classes - 1).masked_fill(~valid, 0).long()
    ce = F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
    probs = logits.softmax(dim=1)
    sum_pred = (probs * valid[:, None]).sum(dim=(0, 2, 3))
    sum_gt = torch.bincount(clean[valid], minlength=classes).to(probs.dtype)
    p_true = probs.gather(1, clean[:, None]).squeeze(1)
    inter = probs.new_zeros((classes,))
    inter = inter.scatter_add(0, clean[valid], p_true[valid])
    dice = (2 * inter + 1e-6) / (sum_pred + sum_gt + 1e-6)
    return ce + dice_weight * (1 - dice[sum_gt > 0].mean())
