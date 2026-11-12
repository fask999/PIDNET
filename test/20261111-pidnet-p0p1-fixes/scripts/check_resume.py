"""Exercise the training entry point with real CPU SGD and temporary archives."""

from contextlib import ExitStack
from copy import deepcopy
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch
from torch import nn

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import train as entry
from utils.checkpoint import load_model_checkpoint


class TinyDataset(torch.utils.data.Dataset):
    class_weights = None

    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor([0.2, -1.0, 0.3]), torch.tensor([0.4, -0.5])


class CPUParallel(nn.Module):
    def __init__(self, module, device_ids):
        super().__init__()
        self.module = module

    def cuda(self):
        return self


def snapshot(model, optimizer):
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    buffers = {}
    for name, param in model.named_parameters():
        buffer = optimizer.state.get(param, {}).get('momentum_buffer')
        buffers[name] = None if buffer is None else buffer.detach().clone()
    return weights, buffers, optimizer.param_groups[0]['lr']


def assert_tensors(actual, expected):
    assert set(actual) == set(expected), (list(actual), list(expected))
    for name, value in expected.items():
        if value is None:
            assert actual[name] is None, name
        else:
            assert torch.equal(actual[name], value), name


def assert_snapshot(actual, expected):
    assert_tensors(actual[0], expected[0])
    assert_tensors(actual[1], expected[1])
    assert actual[2] == expected[2], (actual[2], expected[2])


def sgd_step(model, optimizer, features, targets):
    optimizer.zero_grad()
    (model(features) - targets).square().mean().backward()
    optimizer.step()


def run_main(directory, *, pretrained, resume, end_epoch=1, scores=None, error=None, momentum=0.9):
    logger = logging.Logger('resume-regression', level=logging.CRITICAL)
    argv = ['tools/train.py', '--cfg', str(ROOT / 'configs/cityscapes/pidnet_small_cityscapes.yaml'),
            '--seed', '304', 'GPUS', '(0,)', 'WORKERS', '0',
            'MODEL.PRETRAINED', str(pretrained), 'TRAIN.RESUME', str(resume),
            'TRAIN.END_EPOCH', str(end_epoch), 'TRAIN.MOMENTUM', str(momentum),
            'TRAIN.BATCH_SIZE_PER_GPU', '1', 'TEST.BATCH_SIZE_PER_GPU', '1',
            'OUTPUT_DIR', str(directory), 'LOG_DIR', str(directory)]
    calls = {'build': [], 'reads': [], 'train': [], 'validate': [], 'save': []}
    observations = {}
    real_save, real_load = torch.save, torch.load

    def build(cfg, imgnet_pretrained, *, load_pretrained=True):
        calls['build'].append((imgnet_pretrained, load_pretrained))
        model = nn.Linear(3, 2)
        if load_pretrained:
            load_model_checkpoint(model, cfg.MODEL.PRETRAINED, strict=False)
        return model

    def read(path, *args, **kwargs):
        calls['reads'].append((Path(path).name, kwargs.get('map_location')))
        return real_load(path, *args, **kwargs)

    def train(cfg, epoch, end_epoch, epoch_iters, base_lr, num_iters, loader, optimizer, parallel, writer):
        calls['train'].append(epoch)
        model = parallel.module.model
        assert type(optimizer) is torch.optim.SGD
        features, targets = next(iter(loader))
        before = snapshot(model, optimizer) if len(calls['train']) == 1 else None
        sgd_step(model, optimizer, features, targets)
        if epoch == 0:
            # Mimic a scheduled LR different from the initial config on resume.
            optimizer.param_groups[0]['lr'] = 0.027
        if before is not None:
            observations['first_before'] = before
            observations['first_after'] = snapshot(model, optimizer)

    def validate(cfg, loader, model, writer):
        epoch = calls['train'][-1]
        calls['validate'].append(epoch)
        score = np.float64(scores(epoch))
        return 0.5, score, np.array([score])

    def save(value, path, *args, **kwargs):
        name = Path(path).name
        calls['save'].append((name, value['best_mIoU'] if name == 'checkpoint.pth.tar' else None))
        return real_save(value, path, *args, **kwargs)

    with ExitStack() as stack:
        patches = (
            (sys, 'argv', argv), (entry, 'config', entry.config.clone()),
            (entry, 'create_logger', lambda *a: (logger, str(directory), str(directory))),
            (entry, 'SummaryWriter', mock.Mock()), (torch.cuda, 'device_count', lambda: 1),
            (nn, 'DataParallel', CPUParallel), (entry.models.pidnet, 'get_seg_model', build),
            (entry.datasets, 'cityscapes', TinyDataset), (entry, 'train', train),
            (entry, 'validate', validate), (torch, 'load', read), (torch, 'save', save),
        )
        for target, name, value in patches:
            stack.enter_context(mock.patch.object(target, name, value))
        if error is None:
            entry.main()
        else:
            exception_type, fragment = error
            try:
                entry.main()
            except exception_type as exc:
                assert fragment in str(exc), (fragment, str(exc))
                print('rejected before training: {}'.format(exc))
            else:
                raise AssertionError('expected {}: {}'.format(exception_type.__name__, fragment))
            assert calls['train'] == calls['validate'] == calls['save'] == [], calls
    return calls, observations


def check_resume(directory, pretrained, missing):
    path = directory / 'checkpoint.pth.tar'
    best_path = directory / 'best.pt'
    high = 0.65
    calls, fresh = run_main(directory, pretrained=pretrained, resume=False, scores=lambda epoch: high)
    assert calls['build'] == [(True, True)]
    assert calls['reads'] == [(pretrained.name, 'cpu')]
    assert calls['train'] == calls['validate'] == [0]
    assert calls['save'] == [('best.pt', None), ('checkpoint.pth.tar', high), ('final_state.pt', None)]
    assert_tensors(fresh['first_before'][0], torch.load(pretrained, map_location='cpu'))
    assert all(buffer is None for buffer in fresh['first_before'][1].values())
    assert all(buffer is not None for buffer in fresh['first_after'][1].values())
    original = torch.load(path, map_location='cpu')
    best = torch.load(best_path, map_location='cpu')
    best_bytes = best_path.read_bytes()
    assert set(original) == {'epoch', 'best_mIoU', 'state_dict', 'optimizer'}
    assert original['epoch'] == 1 and original['best_mIoU'] == high
    assert_tensors(best, original['state_dict'])  # best.pt remains raw FullModel weights.

    # A reference step from the original archive checks that restored momentum is used.
    model = nn.Linear(3, 2)
    load_model_checkpoint(model, original, strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.load_state_dict(original['optimizer'])
    features, targets = TinyDataset()[0]
    sgd_step(model, optimizer, features.unsqueeze(0), targets.unsqueeze(0))

    calls, resumed = run_main(directory, pretrained=missing, resume=True, end_epoch=108,
                              scores=lambda epoch: high if epoch == 5 else 0.4)
    assert calls['reads'] == [(path.name, 'cpu')]
    assert calls['build'] == [(True, False)]
    assert calls['train'] == list(range(1, 108))
    assert calls['validate'] == [1, 5] + list(range(8, 108)), calls['validate']
    assert calls['save'] == [(path.name, high)] * 107 + [('final_state.pt', None)]
    assert_snapshot(resumed['first_before'], fresh['first_after'])
    assert_snapshot(resumed['first_after'], snapshot(model, optimizer))
    latest = torch.load(path, map_location='cpu')
    assert latest['epoch'] == 108 and latest['best_mIoU'] == high
    assert not torch.equal(latest['state_dict']['model.weight'], best['model.weight'])
    assert best_path.read_bytes() == best_bytes
    print('resume restored model, LR and real SGD momentum; low/tied/unvalidated epochs kept best 0.65')
    return original


def check_invalid(directory, original, missing):
    path = directory / 'checkpoint.pth.tar'

    def rejects(exception, fragment, built=False):
        calls, _ = run_main(directory, pretrained=missing, resume=True, end_epoch=2,
                            error=(exception, fragment))
        assert calls['build'] == ([(True, False)] if built else []), calls
        assert calls['reads'] == [(path.name, 'cpu')], calls

    rejects(FileNotFoundError, 'TRAIN.RESUME requires checkpoint')
    path.write_bytes(b'broken archive')
    rejects(ValueError, 'Cannot read training checkpoint')
    for name, value in (('epoch', True), ('epoch', -1), ('epoch', 1.25),
                        ('best_mIoU', float('nan')), ('best_mIoU', 1.1),
                        ('state_dict', []), ('optimizer', None)):
        torch.save(dict(original, **{name: value}), path)
        rejects(ValueError, name + ' must be')
    torch.save([], path)
    rejects(ValueError, 'expected a dictionary')
    torch.save(dict(original, optimizer={}), path)
    rejects(ValueError, 'optimizer.state must be a dictionary')
    torch.save(dict(original, optimizer={'state': {}, 'param_groups': {}}), path)
    rejects(ValueError, 'optimizer.param_groups must be a list')

    def invalid_optimizer(change, fragment):
        saved = deepcopy(original)
        change(saved['optimizer'], saved['optimizer']['param_groups'][0])
        torch.save(saved, path)
        rejects(ValueError, fragment, built=True)

    invalid_optimizer(lambda opt, group: opt['param_groups'].append(deepcopy(group)), 'param_groups count')
    invalid_optimizer(lambda opt, group: group['params'].pop(), '.params do not match')
    invalid_optimizer(lambda opt, group: group['params'].reverse(), '.params do not match')
    invalid_optimizer(lambda opt, group: group['params'].__setitem__(1, 0), '.params do not match')
    invalid_optimizer(lambda opt, group: group.pop('lr'), '.lr is missing or invalid')
    invalid_optimizer(lambda opt, group: group.update(momentum='bad'), '.momentum is missing or invalid')
    invalid_optimizer(lambda opt, group: group.update(lr=1e300), '.lr is missing or invalid')
    invalid_optimizer(lambda opt, group: group.update(nesterov=True, dampening=0.2), '.nesterov requires')
    invalid_optimizer(lambda opt, group: opt['state'].update({2: {}}), 'does not refer to an SGD parameter')
    invalid_optimizer(lambda opt, group: opt['state'].update({0: {'exp_avg': torch.zeros(2, 3)}}),
                      'must contain only SGD momentum state')
    invalid_optimizer(lambda opt, group: opt['state'][0].update(momentum_buffer=torch.zeros(3)),
                      '.momentum_buffer shape (3,) does not match parameter shape (2, 3)')
    invalid_optimizer(lambda opt, group: opt['state'][0].update(momentum_buffer='bad'),
                      '.momentum_buffer must be a dense floating-point tensor')
    torch.save(dict(original, state_dict={'model.weight': torch.zeros(1)}), path)
    rejects(RuntimeError, 'Checkpoint does not match model', built=True)


def check_legacy(directory, original, missing):
    model = nn.Linear(3, 2)
    load_model_checkpoint(model, original, strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0)
    features, targets = TinyDataset()[0]
    sgd_step(model, optimizer, features.unsqueeze(0), targets.unsqueeze(0))
    expected = snapshot(model, optimizer)
    assert all(buffer is None for buffer in expected[1].values())
    legacy = optimizer.state_dict()
    for name in ('foreach', 'maximize', 'differentiable'):
        legacy['param_groups'][0].pop(name, None)
    torch.save({'epoch': 1, 'best_mIoU': np.float64(0.65),
                'state_dict': model.state_dict(), 'optimizer': legacy}, directory / 'checkpoint.pth.tar')
    calls, observed = run_main(directory, pretrained=missing, resume=True, end_epoch=2,
                               scores=lambda epoch: 0.4, momentum=0.0)
    assert calls['train'] == calls['validate'] == [1]
    assert_snapshot(observed['first_before'], expected)
    assert all(buffer is None for buffer in observed['first_after'][1].values())
    print('legacy NumPy best and zero-momentum SGD archive resumed with optional options absent')


def main():
    assert not torch.cuda.is_available(), 'run with CUDA_VISIBLE_DEVICES empty'
    torch.manual_seed(304)
    with TemporaryDirectory(prefix='resume-regression-', dir=str(EXPERIMENT / 'output')) as name:
        root = Path(name)
        pretrained = root / 'imagenet-tiny.pt'
        missing = root / 'imagenet-missing.pt'
        torch.save(nn.Linear(3, 2).state_dict(), pretrained)
        for subdir in ('current', 'invalid', 'legacy'):
            (root / subdir).mkdir()
        original = check_resume(root / 'current', pretrained, missing)
        check_invalid(root / 'invalid', original, missing)
        check_legacy(root / 'legacy', original, missing)
    assert not root.exists()
    print('resume regression passed (PyTorch {}, CPU SGD, seed 304)'.format(torch.__version__))


if __name__ == '__main__':
    main()
