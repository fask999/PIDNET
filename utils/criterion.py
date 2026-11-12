# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if not isinstance(score, (list, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            weights = balance_weights
        elif len(score) == 1:
            weights = [sb_weights]
        else:
            raise ValueError("lengths of prediction and target are not identical!")

        if not (target != self.ignore_label).any():
            return sum(w * x[..., :0].sum() for w, x in zip(weights, score))
        return sum(w * self._forward(x, target) for w, x in zip(weights, score))

        


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):


        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        if pred.numel() == 0:
            return pixel_losses[mask].sum()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        selected = pixel_losses[pred < threshold]
        if selected.numel() < min(self.min_kept, pred.numel()):
            selected = pixel_losses[:min(self.min_kept, pred.numel())]
        return selected.mean()

    def forward(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target, single_class_bce_weight=0.0):
    log_p = bd_pre.permute(0, 2, 3, 1).reshape(1, -1)
    return _weighted_bce(log_p, target.reshape(1, -1), single_class_bce_weight)


def _weighted_bce(log_p, target_t, single_class_bce_weight=0.0):
    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    if single_class_bce_weight:
        # Only binary target positions receive weight, even without void masking.
        single_weight = (pos_index | neg_index).to(log_p.dtype) * single_class_bce_weight
        weight = torch.where((pos_num > 0) & (neg_num > 0), weight, single_weight)
    return F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0, dice_weight=0.0, mask_void=False,
                 single_class_bce_weight=0.0):
        super(BondaryLoss, self).__init__()
        if (isinstance(single_class_bce_weight, bool) or
                not isinstance(single_class_bce_weight, (int, float)) or
                not math.isfinite(single_class_bce_weight) or single_class_bce_weight < 0):
            raise ValueError('single_class_bce_weight must be finite and nonnegative')
        self.coeff_bce = coeff_bce
        self.dice_weight = dice_weight
        self.mask_void = mask_void
        self.single_class_bce_weight = float(single_class_bce_weight)

    def forward(self, bd_pre, bd_gt):
        if self.mask_void:
            target = bd_gt.reshape(-1)
            valid = target >= 0
            logits = bd_pre.permute(0, 2, 3, 1).reshape(-1)[valid]
            target = target[valid]
            if target.numel() == 0:
                return logits.sum()
            loss = self.coeff_bce * _weighted_bce(logits, target, self.single_class_bce_weight)
        else:
            logits = bd_pre
            target = bd_gt
            loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt, self.single_class_bce_weight)

        if self.dice_weight:
            prob = logits.float().reshape(-1).sigmoid()
            target = target.float().reshape(-1)
            dice = 1.0 - (2.0 * (prob * target).sum() + 1.0) / (
                prob.sum() + target.sum() + 1.0)
            loss = loss + self.dice_weight * dice
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        


