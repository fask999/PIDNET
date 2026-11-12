import math
import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
from configs import config
from utils import criterion as current
from utils.utils import FullModel

old = {'__name__': 'head_criterion'}
exec(compile(subprocess.check_output(['git', 'show', '4c158cf:utils/criterion.py']).decode(), '4c158cf:utils/criterion.py', 'exec'), old)
print('torch', torch.__version__, 'device cpu', 'seed none (deterministic fixed inputs)')


def boundary(version, targets, mask=False, dice=0):
    target = torch.tensor(targets, dtype=torch.float64).reshape(len(targets), 1, -1)
    logit = torch.zeros((len(targets), 1, 1, len(targets[0])), dtype=torch.float64, requires_grad=True)
    loss = (old['BondaryLoss']() if version == 'HEAD' else current.BondaryLoss(mask_void=mask, dice_weight=dice))(logit, target)
    loss.backward()
    g = logit.grad.reshape(-1)
    t = target.reshape(-1)
    mean = lambda x: float(x.mean()) if x.numel() else float('nan')
    return dict(loss=loss.item(), pos_grad=mean(g[t == 1]), neg_grad=mean(g[t == 0]), void_grad=mean(g[t < 0]), l1=g.abs().sum().item(), l2=g.norm().item(), signed=g.sum().item())

N = 1000
print('\nBCE zero logits N=1000 coeff=20 (per-pixel signed grad, L1/L2 and signed sum)')
for k in (0, 1, 10, 20, 30, 1000):
    targets = [[1.] * k + [0.] * (N-k)]
    for version, mask in [('HEAD', False), ('current', False), ('current', True)]:
        print('positive=%d %s mask=%s %s' % (k, version, mask, boundary(version, targets, mask)))
print('all void current masked', boundary('current', [[-1.] * N], True))
print('all void HEAD unmasked', boundary('HEAD', [[-1.] * N]))
print('void comparison current mask: 10 valid [2 pos,8 neg] vs same with 4 void appended')
print('valid only', boundary('current', [[1.,1.] + [0.]*8], True))
print('plus void', boundary('current', [[1.,1.] + [0.]*8 + [-1.]*4], True))
print('void unmasked current', boundary('current', [[1.,1.] + [0.]*8 + [-1.]*4], False))
print('B2 negative+mixed', boundary('current', [[0.]*10, [1.,1.] + [0.]*8], True))
print('B1 negative', boundary('current', [[0.]*10], True))
print('B1 mixed', boundary('current', [[1.,1.] + [0.]*8], True))
for k in (0, 2):
    t = [[1.] * k + [0.] * (10-k)]
    print('dice=0 k=%d' % k, boundary('current', t, True, 0))
    print('dice=1 k=%d' % k, boundary('current', t, True, 1))


def score_for(ps):
    # Binary logits with true class 0, exact probabilities up to FP64 rounding.
    z = torch.tensor([math.log(p / (1-p)) for p in ps], dtype=torch.float64)
    return torch.stack((z, torch.zeros_like(z))).reshape(1,2,1,-1).requires_grad_()


def ohem(cls, ps, ignored, keep=3, thresh=.9):
    x = score_for(ps)
    y = torch.tensor([[-1 if i in ignored else 0 for i in range(len(ps))]], dtype=torch.long).reshape(1,1,-1)
    fun = cls(ignore_label=-1, thres=thresh, min_kept=keep)
    try:
        loss = fun._ohem_forward(x, y)
        loss.backward()
        return 'loss=%.12g grad_l1=%.12g grad=%s' % (loss.item(), x.grad.abs().sum().item(), x.grad[0,0,0].tolist())
    except Exception as e:
        return '%s: %s' % (type(e).__name__, str(e))

print('\nOHEM scores=true-class probabilities threshold=.9; compare raw _ohem_forward')
cases = [('one_high',[.95],[],3), ('all_tied_high',[.95]*4,[],3), ('all_ignore',[.95]*4,[0,1,2,3],3), ('ordinary',[.2,.4,.6,.95,.97],[],3), ('N_lt_K_distinct_high',[.91,.95,.98],[],5), ('partial_tie',[.91,.95,.95,.95,.98],[],3), ('below_threshold',[.2,.4,.6],[],5), ('threshold_more_than_K',[.2,.3,.4,.5,.6],[],2)]
for name, ps, ignored, keep in cases:
    print(name, 'N=%d K=%d p=%s ignored=%s' % (len(ps)-len(ignored), keep, ps, ignored))
    for label, cls in [('HEAD', old['OhemCrossEntropy']), ('current', current.OhemCrossEntropy)]:
        print(' ', label, ohem(cls, ps, ignored, keep))

# Real FullModel BAS dispatch: one boundary gate out of two and high-confidence target.
class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.sem = nn.Parameter(score_for([.95,.5]).detach())
        self.bd = nn.Parameter(torch.tensor([[[[2., -2.]]]], dtype=torch.float64))
    def forward(self, x):
        return [self.sem, self.bd]
class Spy(current.OhemCrossEntropy):
    def forward(self, x, y):
        out = super(Spy, self).forward(x, y)
        print(' FullModel sem_loss target=%s value=%.12g' % (y.reshape(-1).tolist(), out.item()))
        return out
print('\nFullModel OHEM BAS, threshold .9 keep 100000, sigmoid(2)>.8; SB_WEIGHT=%s' % config.LOSS.SB_WEIGHTS)
model = FullModel(Dummy(), Spy(ignore_label=-1, thres=.9, min_kept=100000), current.BondaryLoss())
total = model(torch.zeros(1), torch.tensor([[[0,0]]]), torch.tensor([[[0.,1.]]], dtype=torch.float64))[0]
total.sum().backward()
print(' FullModel total=',total.item(), 'sem_grad=',model.model.sem.grad.tolist())

print('\nRATIOS: current mask=True, zero logits, all-negative N=1000 / mixed N=1000')
base = boundary('current', [[0.]*N], True)
for k in (1,10,20,30):
    other = boundary('current', [[1.]*k + [0.]*(N-k)], True)
    print('positive=%d loss=%.9g negative_per_pixel=%.9g L1=%.9g L2=%.9g' % (k, *(base[key]/other[key] for key in ('loss','neg_grad','l1','l2'))))

print('\nOHEM fallback comparison, direct CE unweighted; same inputs and threshold .9')
for name, ps, ignored, keep in cases:
    if ignored: continue
    score = score_for(ps)
    pred = torch.softmax(score, dim=1)[0,0,0]
    losses = -pred.log()
    pred, idx = pred.sort()
    sorted_loss = losses[idx]
    threshold = max(pred[min(keep, pred.numel()-1)], .9)
    chosen = sorted_loss[pred < threshold]
    minimal = chosen if chosen.numel() else sorted_loss[:min(keep, pred.numel())]
    topk = sorted_loss[:min(keep, pred.numel())]
    guaranteed = sorted_loss[:max(min(keep, pred.numel()), chosen.numel())]
    print('%s threshold_count=%d minimal_empty_fallback=%d/%.12g topK=%d/%.12g threshold_plus_K=%d/%.12g' % (name, chosen.numel(), minimal.numel(), minimal.mean().item(), topk.numel(), topk.mean().item(), guaranteed.numel(), guaranteed.mean().item()))
