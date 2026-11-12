#!/usr/bin/env python3
"""Reproduce the historical synthetic step, then isolate BN/weight/input effects.

All generated images and lists go into this experiment's tests/bn-fixture.
The old fixture generator is imported only; its main is never executed.
Run from repository root using the command recorded in the audit plan.
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'test/20261111-external-review-audit'
OLD = ROOT / 'test/20261111-pidnet-ablation-switches'
FIXTURE_ROOT = EXP / 'tests/bn-fixture'
FIXTURE = FIXTURE_ROOT
CFG = OLD / 'configs/pidnet_medium_cityscapes.yaml'
SEED = 304
sys.path.insert(0, str(ROOT))

from configs import config, update_config
from datasets.cityscapes import Cityscapes
from models.pidnet import get_seg_model
from utils.ablation_config import (get_boundary_kwargs, get_boundary_loss_kwargs,
                                   register_ablation_config)
from utils.criterion import BondaryLoss, OhemCrossEntropy
from utils.utils import FullModel


def record(kind, **items):
    print(json.dumps(dict(kind=kind, **items), sort_keys=True, allow_nan=False), flush=True)


def old_fixture_generator():
    spec = importlib.util.spec_from_file_location('old_verify_ablation',
                                                  str(OLD / 'tests/verify_ablation.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic_dataset


def configure(variant):
    register_ablation_config(config)
    opts = []
    if variant == 'combo':
        opts += ['ABLATION.CONTEXT', 'denseaspp',
                 'ABLATION.I_ATROUS.MODE', 'fadc',
                 'ABLATION.BOUNDARY.TARGET', 'valid',
                 'ABLATION.BOUNDARY.MASK_VOID', 'True',
                 'ABLATION.BOUNDARY.DICE_WEIGHT', '1.0']
    elif variant != 'baseline':
        raise ValueError(variant)
    opts += ['DATASET.ROOT', str(FIXTURE / 'data') + os.sep,
             'OUTPUT_DIR', str(EXP / 'output'), 'LOG_DIR', str(EXP / 'logs'),
             'GPUS', '(0,)', 'WORKERS', '0', 'CUDNN.BENCHMARK', 'False',
             'MODEL.PRETRAINED', '', 'MODEL.NAME', 'pidnet_s',
             'TRAIN.IMAGE_SIZE', '[192, 128]', 'TRAIN.MULTI_SCALE', 'False',
             'TRAIN.FLIP', 'False', 'TRAIN.END_EPOCH', '1',
             'TEST.IMAGE_SIZE', '[192, 128]']
    update_config(config, argparse.Namespace(cfg=str(CFG), opts=opts))
    return opts


def dataset(is_train):
    kwargs = dict(root=config.DATASET.ROOT,
                  list_path=config.DATASET.TRAIN_SET if is_train else config.DATASET.TEST_SET,
                  num_classes=config.DATASET.NUM_CLASSES,
                  multi_scale=config.TRAIN.MULTI_SCALE if is_train else False,
                  flip=config.TRAIN.FLIP if is_train else False,
                  ignore_label=config.TRAIN.IGNORE_LABEL,
                  base_size=config.TRAIN.BASE_SIZE if is_train else config.TEST.BASE_SIZE,
                  crop_size=(config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
                  if is_train else (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]))
    if is_train:
        kwargs['scale_factor'] = config.TRAIN.SCALE_FACTOR
    kwargs.update(get_boundary_kwargs(config))
    return Cityscapes(**kwargs)


def snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def bn_layers(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)]


def bn_snapshot(model):
    return {name: (module.running_mean.detach().cpu().clone(),
                   module.running_var.detach().cpu().clone(),
                   module.num_batches_tracked.detach().cpu().clone())
            for name, module in bn_layers(model)}


def restore_bn(model, state):
    with torch.no_grad():
        for name, module in bn_layers(model):
            mean, var, count = state[name]
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            module.num_batches_tracked.copy_(count)


def bn_description(model, label, variant):
    pairs = bn_layers(model)
    means = torch.cat([module.running_mean.detach().cpu().view(-1) for _, module in pairs])
    variances = torch.cat([module.running_var.detach().cpu().view(-1) for _, module in pairs])
    counts = [int(module.num_batches_tracked) for _, module in pairs]
    by_layer = sorted([(float(module.running_var.mean()), name)
                       for name, module in pairs])
    def describe(values):
        return {'min': float(values.min()), 'median': float(values.median()),
                'mean': float(values.mean()), 'max': float(values.max())}
    record('bn_buffers', variant=variant, state=label, layers=len(pairs),
           channels=variances.numel(), batch_counts=[min(counts), max(counts)],
           momenta=sorted(set(float(module.momentum) for _, module in pairs)),
           mean_abs=describe(means.abs()), variance=describe(variances),
           layer_var_mean_min=by_layer[0], layer_var_mean_max=by_layer[-1])


def norm_description(model, initial, variant):
    # Four-dimensional weights cover Conv2d plus the FADC wrapper's learned kernel.
    current = dict(model.named_parameters())
    changes = []
    for name, weight in current.items():
        if weight.ndim == 4 and name.endswith('weight'):
            before = float(initial[name].double().norm())
            after = float(weight.detach().cpu().double().norm())
            changes.append((after - before, name, before, after))
    tol = 1e-12
    increases = sum(delta > tol for delta, _, _, _ in changes)
    decreases = sum(delta < -tol for delta, _, _, _ in changes)
    unchanged = len(changes) - increases - decreases
    record('conv_weight_norms', variant=variant, count=len(changes),
           increased=increases, decreased=decreases, unchanged=unchanged,
           largest_increase=sorted(changes)[-3:], largest_decrease=sorted(changes)[:3])


class LogitObserver:
    def __init__(self, model):
        self.current = None
        self.hook = model.register_forward_hook(self.capture)

    def capture(self, module, inputs, outputs):
        # Capture raw head values before FullModel interpolates them in place.
        self.current = []
        for name, value in zip(('aux_semantic', 'main_semantic', 'boundary'), outputs):
            v = value.detach().float()
            self.current.append(dict(head=name, mean_abs=float(v.abs().mean()),
                                     rms=float(v.square().mean().sqrt()),
                                     max_abs=float(v.abs().max())))

    def close(self):
        self.hook.remove()


def to_cuda(batch):
    return batch[0].cuda(), batch[1].long().cuda(), batch[2].float().cuda()


def unpack(result, observer):
    loss, _, _, pieces = result
    sem = float(pieces[0].mean().detach())
    boundary = float(pieces[1].mean().detach())
    total = float(loss.mean().detach())
    return dict(total=total, semantic=sem, boundary=boundary,
                bas=total-sem-boundary, logits=observer.current)


def set_bn_mode(model, batch_stats):
    # Keep every non-BN module in eval; toggle only BN on identical weights/inputs.
    model.eval()
    if batch_stats:
        for _, bn in bn_layers(model):
            bn.train()


def run_forward(dp, observer, x, y, b):
    with torch.no_grad():
        return unpack(dp(x, y, b), observer)


def clone_cpu_batch(batch):
    return tuple(t.detach().cpu().clone() for t in batch[:3])


def run_variant(variant):
    opts = configure(variant)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    # This follows tools/train.py's RNG-relevant creation order.
    core = get_seg_model(config, imgnet_pretrained=False)
    train_ds = dataset(True)
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE, num_workers=config.WORKERS,
        pin_memory=False, drop_last=True)
    val_ds = dataset(False)
    valloader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False, num_workers=config.WORKERS, pin_memory=False)
    sem = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                           thres=config.LOSS.OHEMTHRES,
                           min_kept=config.LOSS.OHEMKEEP,
                           weight=train_ds.class_weights)
    full = FullModel(core, sem, BondaryLoss(**get_boundary_loss_kwargs(config)))
    dp = nn.DataParallel(full, device_ids=[0]).cuda()
    optimizer = torch.optim.SGD(list(dict(dp.named_parameters()).values()),
                                lr=config.TRAIN.LR, momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD, nesterov=config.TRAIN.NESTEROV)
    observer = LogitObserver(core)
    initial = snapshot(core)
    virgin_bn = bn_snapshot(core)
    bn_description(core, 'virgin', variant)
    record('config', variant=variant, config=str(CFG), options=opts,
           seed=SEED, optimizer={'lr': config.TRAIN.LR, 'momentum': config.TRAIN.MOMENTUM,
                                 'weight_decay': config.TRAIN.WD, 'nesterov': config.TRAIN.NESTEROV},
           num_classes=config.DATASET.NUM_CLASSES, criterion='OhemCrossEntropy + BondaryLoss',
           balance_weights=list(config.LOSS.BALANCE_WEIGHTS), sb_weight=config.LOSS.SB_WEIGHTS,
           boundary_options=get_boundary_kwargs(config), boundary_loss_options=get_boundary_loss_kwargs(config))
    dp.train()
    train_batch = next(iter(trainloader))
    train_cpu = clone_cpu_batch(train_batch)
    train_cuda = to_cuda(train_batch)
    result = dp(*train_cuda)
    before_step = unpack(result, observer)
    dp.zero_grad()
    result[0].mean().backward()
    optimizer.step()  # Sole SGD update for this structure.
    updated = snapshot(core)
    one_forward_bn = bn_snapshot(core)
    bn_description(core, 'one_preupdate_train_forward', variant)
    norm_description(core, initial, variant)
    record('train_one_step', variant=variant, names=list(train_batch[4]), **before_step)
    del result

    # Original validation protocol: B=1 and the unweighted mean over the two samples.
    dp.eval()
    val_samples = []
    val_results = []
    for batch in valloader:
        val_samples.append(clone_cpu_batch(batch))
        out = run_forward(dp, observer, *to_cuda(batch))
        val_results.append(out)
        record('historical_val_b1', variant=variant, name=batch[4][0], **out)
    record('historical_val_b1_mean', variant=variant,
           **{key: sum(row[key] for row in val_results) / len(val_results)
              for key in ('total', 'semantic', 'boundary', 'bas')})
    val_cpu = tuple(torch.cat([row[i] for row in val_samples], dim=0) for i in range(3))
    data = {'train': train_cpu, 'val': val_cpu}

    # Each cell uses B=2, identical inputs and labels across BN mode / weight changes.
    # Crossing image and targets separates their contributions, still using real losses.
    for weights, model_state in [('W0', initial), ('W1', updated)]:
        for bn_name, bn_state, batch_stats in [('virgin', virgin_bn, False),
                                               ('one_forward', one_forward_bn, False),
                                               ('batch', one_forward_bn, True)]:
            for image_source, target_source in [('train', 'train'), ('val', 'val'),
                                                ('train', 'val'), ('val', 'train')]:
                core.load_state_dict(model_state, strict=True)
                restore_bn(core, bn_state)
                set_bn_mode(dp, batch_stats)
                image = data[image_source][0].cuda()
                labels = data[target_source][1].long().cuda()
                edges = data[target_source][2].float().cuda()
                values = run_forward(dp, observer, image, labels, edges)
                record('paired_b2', variant=variant, weights=weights, bn=bn_name,
                       images=image_source, targets=target_source, **values)

    # Re-estimate from TRAIN images only after freezing W1: one pass with momentum=1.
    # This is a causal probe, not a selected/validated deployment checkpoint.
    core.load_state_dict(updated, strict=True)
    restore_bn(core, virgin_bn)
    set_bn_mode(dp, True)
    bns = bn_layers(core)
    old_momenta = [bn.momentum for _, bn in bns]
    for _, bn in bns:
        bn.momentum = 1.0
    with torch.no_grad():
        dp(*to_cuda(train_cpu))
    for (_, bn), momentum in zip(bns, old_momenta):
        bn.momentum = momentum
    recalibrated_bn = bn_snapshot(core)
    unchanged_weights = all(torch.equal(value.detach().cpu(), updated[name])
                            for name, value in core.named_parameters())
    assert unchanged_weights
    bn_description(core, 'W1_train_batch_reestimated_momentum1', variant)
    for image_source, target_source in [('train', 'train'), ('val', 'val'),
                                        ('train', 'val'), ('val', 'train')]:
        core.load_state_dict(updated, strict=True)
        restore_bn(core, recalibrated_bn)
        set_bn_mode(dp, False)
        values = run_forward(dp, observer, data[image_source][0].cuda(),
                             data[target_source][1].long().cuda(),
                             data[target_source][2].float().cuda())
        record('paired_b2', variant=variant, weights='W1', bn='W1_train_reestimated',
               images=image_source, targets=target_source, **values)
    record('reestimate_check', variant=variant, train_batch_only=True,
           weight_parameters_unchanged=unchanged_weights, passes=1, momentum=1.0)
    observer.close()
    del dp, optimizer, full, core, train_ds, val_ds, train_cuda
    gc.collect()
    torch.cuda.empty_cache()


def main():
    global FIXTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='check configuration and model on CPU only')
    parser.add_argument('--fixture', default=str(FIXTURE_ROOT),
                        help='new fixture path, contained within this audit\'s tests/bn-fixture')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    fixture = Path(args.fixture).expanduser()
    if not fixture.is_absolute():
        fixture = ROOT / fixture
    fixture = fixture.resolve()
    try:
        fixture.relative_to(FIXTURE_ROOT.resolve())
    except ValueError:
        parser.error('--fixture must be inside {}'.format(FIXTURE_ROOT))
    FIXTURE = fixture
    if args.preflight:
        for variant in ('combo', 'baseline'):
            opts = configure(variant)
            torch.manual_seed(SEED)
            model = get_seg_model(config, imgnet_pretrained=False)
            record('cpu_preflight', variant=variant, options=opts, bn_layers=len(bn_layers(model)),
                   boundary=get_boundary_kwargs(config), batch_norm_cuda_initialized=torch.cuda.is_initialized())
            del model
        return
    if FIXTURE.exists():
        raise RuntimeError('Refusing to overwrite existing fixture: {}'.format(FIXTURE))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This probe requires exactly one visible CUDA device')
    record('environment', python=sys.version, torch=torch.__version__,
           cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
           fixture=str(FIXTURE), seed=SEED,
           historical_reference={'train_total': 14.690834, 'train_semantic': 6.061063,
                                 'train_boundary': 4.215402, 'train_bas': 4.414369,
                                 'val_total': 7575.002})
    # Only imported generator function is called, in our own empty fixture directory.
    old_fixture_generator()(FIXTURE / 'data', 'cityscapes')
    for variant in ('combo', 'baseline'):
        run_variant(variant)
    record('complete', structures=2, sgd_updates_per_structure=1)


if __name__ == '__main__':
    main()
