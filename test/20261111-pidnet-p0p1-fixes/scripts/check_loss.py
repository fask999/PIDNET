"""CPU numerical and gradient regression for boundary BCE and OHEM/BAS."""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import config
from utils.ablation_config import register_ablation_config, get_boundary_loss_kwargs
from utils.criterion import BondaryLoss, OhemCrossEntropy
from utils.utils import FullModel


torch.manual_seed(304)
torch.set_num_threads(2)


def close(actual, expected, name, atol=1e-9):
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=atol, msg=name)


def check_pair(actual, reference, x, xr, name, atol=1e-9):
    close(actual, reference, name + ' value', atol)
    ga = torch.autograd.grad(actual, x)[0]
    gr = torch.autograd.grad(reference, xr)[0]
    close(ga, gr, name + ' gradient', atol)


def boundary(logits, targets, mask=False, single=0., dice=0., coeff=7.):
    x = torch.tensor(logits, dtype=torch.float64).reshape(len(targets), 1, 1, -1).requires_grad_()
    y = torch.tensor(targets, dtype=torch.float64).reshape(len(targets), 1, -1)
    return x, y, BondaryLoss(coeff_bce=coeff, mask_void=mask, dice_weight=dice,
                              single_class_bce_weight=single)(x, y)


def boundary_reference(x, y, mask, single, dice=0., coeff=7.):
    logits = x.reshape(-1)
    target = y.reshape(-1)
    if mask:
        valid = target >= 0
        logits, target = logits[valid], target[valid]
    if not target.numel():
        return logits.sum()
    pos, neg = target == 1, target == 0
    npos, nneg = pos.sum().item(), neg.sum().item()
    if npos and nneg:
        weights = pos.to(x.dtype) * (nneg / (npos + nneg)) + neg.to(x.dtype) * (npos / (npos + nneg))
    else:
        weights = (pos | neg).to(x.dtype) * single
    loss = coeff * F.binary_cross_entropy_with_logits(logits, target, weight=weights)
    if dice:
        p, t = logits.float().sigmoid(), target.float()
        loss = loss + dice * (1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1))
    return loss


for cls in (0., 1.):
    for mask in (False, True):
        for single in (0., .3, 1.5):
            x, y, actual = boundary([[-1.2, .3, 2.]], [[cls] * 3], mask, single)
            xr = x.detach().clone().requires_grad_()
            expected = 7 * single * F.binary_cross_entropy_with_logits(xr.reshape(-1), y.reshape(-1))
            check_pair(actual, expected, x, xr, 'single class %s mask %s weight %s' % (cls, mask, single))

# The positive and negative counts are over the local batch, and the ordinary
# weighted mean remains unchanged even when single-class supervision is enabled.
for mask in (False, True):
    for single in (0., .4):
        for targets in ([[0., 0., 0.], [1., 0., 1.]], [[1., 0., 0., 1.]]):
            logits = [[-.8, .1, 1.1]] * 2 if len(targets) == 2 else [[-.8, .1, 1.1, .7]]
            x, y, actual = boundary(logits, targets, mask, single)
            xr = x.detach().clone().requires_grad_()
            check_pair(actual, boundary_reference(xr, y, mask, single), x, xr, 'mixed batch')

# Adding ignored pixels leaves valid-pixel loss and gradient intact under masking.
for single, valid in ((0.6, [0., 0., 0.]), (0.6, [1., 0., 1.])):
    xa, ya, la = boundary([[.1, -.4, 1.2]], [valid], True, single, dice=.7)
    xb, yb, lb = boundary([[.1, -.4, 1.2, 9., -4.]], [valid + [-1., -1.]], True, single, dice=.7)
    close(la, lb, 'fixed valid region', atol=1e-7)
    ga = torch.autograd.grad(la, xa)[0].reshape(-1)
    gb = torch.autograd.grad(lb, xb, retain_graph=True)[0].reshape(-1)
    close(ga, gb[:3], 'fixed valid gradient', atol=1e-7)
    close(gb[3:], torch.zeros_like(gb[3:]), 'void gradient')
    xr = xb.detach().clone().requires_grad_()
    check_pair(lb, boundary_reference(xr, yb, True, single, .7), xb, xr, 'dice and masked BCE', atol=1e-7)

for single in (0., .6):
    x, y, loss = boundary([[1., -2.]], [[-1., -1.]], True, single, dice=.7)
    close(loss, torch.zeros_like(loss), 'all void value')
    close(torch.autograd.grad(loss, x)[0], torch.zeros_like(x), 'all void gradient')

# Config registration, propagation, legacy/partial configs, and new parameter validation.
cfg = config.clone()
register_ablation_config(cfg)
assert get_boundary_loss_kwargs(cfg)['single_class_bce_weight'] == 0.
cfg.ABLATION.BOUNDARY.SINGLE_CLASS_BCE_WEIGHT = .35
assert get_boundary_loss_kwargs(cfg)['single_class_bce_weight'] == .35
assert get_boundary_loss_kwargs(SimpleNamespace())['single_class_bce_weight'] == 0.
for bad in (-.1, float('nan'), float('inf'), True, '1'):
    try:
        BondaryLoss(single_class_bce_weight=bad)
    except ValueError:
        pass
    else:
        raise AssertionError('constructor accepted %r' % (bad,))
    partial = SimpleNamespace(ABLATION=SimpleNamespace(BOUNDARY=SimpleNamespace(SINGLE_CLASS_BCE_WEIGHT=bad)))
    try:
        get_boundary_loss_kwargs(partial)
    except ValueError:
        pass
    else:
        raise AssertionError('config accepted %r' % (bad,))
print('Boundary BCE/Dice and configuration: passed')


def score_for(probs):
    z = torch.tensor([math.log(p / (1 - p)) for p in probs], dtype=torch.float64)
    return torch.stack((z, torch.zeros_like(z))).reshape(1, 2, 1, -1).requires_grad_()


# Explicit expected selected counts and indices; sort defines tie order.
cases = [
    ('one', [.95], 3, .9, 1),
    ('same score', [.95] * 4, 3, .9, 3),
    ('partial tie', [.91, .95, .95, .95, .98], 3, .9, 3),
    ('K > N', [.91, .95, .98], 5, .9, 3),
    ('threshold exceeds K', [.2, .3, .4, .5, .6], 2, .9, 5),
    ('ordinary threshold path', [.2, .4, .6, .95, .97], 3, .9, 3),
    ('rank threshold path', [.2, .4, .6, .95, .97], 2, .3, 2),
]
for name, probs, keep, threshold, count in cases:
    x = score_for(probs)
    y = torch.zeros((1, 1, len(probs)), dtype=torch.long)
    actual = OhemCrossEntropy(ignore_label=-1, min_kept=keep, thres=threshold)._ohem_forward(x, y)
    xr = x.detach().clone().requires_grad_()
    order = torch.softmax(xr.detach(), dim=1)[0, 0, 0].sort()[1]
    indices = order[:count]
    reference = F.cross_entropy(xr, y, reduction='none').reshape(-1)[indices].mean()
    check_pair(actual, reference, x, xr, name)

x = score_for([.2, .4, .95])
y = torch.tensor([[[-1, 0, 0]]])
actual = OhemCrossEntropy(ignore_label=-1, min_kept=5, thres=.9)._ohem_forward(x, y)
xr = x.detach().clone().requires_grad_()
reference = F.cross_entropy(xr, y, ignore_index=-1, reduction='none').reshape(-1)[1:].mean()
check_pair(actual, reference, x, xr, 'OHEM partial ignore')

x = score_for([.2, .8])
y = torch.full((1, 1, 2), -1, dtype=torch.long)
loss = OhemCrossEntropy(ignore_label=-1)._ohem_forward(x, y)
close(loss, torch.zeros_like(loss), 'OHEM all ignore')
close(torch.autograd.grad(loss, x)[0], torch.zeros_like(x), 'OHEM all ignore gradient')
print('OHEM selected loss and gradient: passed')


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.sem = nn.Parameter(score_for([.95, .5]).detach())
        self.bd = nn.Parameter(torch.tensor([[[[2., -2.]]]], dtype=torch.float64))

    def forward(self, x):
        return [self.sem, self.bd]


assert config.TRAIN.IGNORE_LABEL == -1
model = FullModel(Dummy(), OhemCrossEntropy(ignore_label=-1, thres=.9, min_kept=100000), BondaryLoss())
labels = torch.tensor([[[0, 0]]])
boundaries = torch.tensor([[[0., 1.]]], dtype=torch.float64)
total, _, _, parts = model(torch.zeros(1), labels, boundaries)
sem = model.model.sem.detach().clone().requires_grad_()
bd = model.model.bd.detach().clone().requires_grad_()
sem_ce = F.cross_entropy(sem, labels, reduction='none').reshape(-1)
ref_sem = config.LOSS.SB_WEIGHTS * sem_ce.mean()
ref_bas = config.LOSS.SB_WEIGHTS * sem_ce[0]
ref_bd = boundary_reference(bd, boundaries, False, 0., coeff=20.)
close(parts[0], ref_sem, 'FullModel semantic')
close(parts[1], ref_bd, 'FullModel boundary')
close(total.squeeze(), ref_sem + ref_bas + ref_bd, 'FullModel total including sparse BAS')
actual_grads = torch.autograd.grad(total.sum(), (model.model.sem, model.model.bd))
reference_grads = torch.autograd.grad(ref_sem + ref_bas + ref_bd, (sem, bd))
for actual, expected in zip(actual_grads, reference_grads):
    close(actual, expected, 'FullModel gradient')
# BAS is independently nonzero and contributes only at the selected semantic pixel.
bas_grad = torch.autograd.grad(config.LOSS.SB_WEIGHTS * F.cross_entropy(sem, labels, reduction='none').reshape(-1)[0], sem)[0]
assert ref_bas.item() > 0 and bas_grad[..., 0].abs().sum() > 0
close(bas_grad[..., 1], torch.zeros_like(bas_grad[..., 1]), 'sparse BAS unselected gradient')
print('FullModel sparse BAS value and gradients: passed; BAS=%.12g' % ref_bas.item())
print('All loss regression checks passed')
