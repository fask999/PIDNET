#!/usr/bin/env python3
"""Run real PIDNet-L behavior checks with synthetic inputs and weight fixtures."""
import argparse
from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import run_ablation as runner

ROOT, EXP = runner.ROOT, runner.EXPERIMENT
sys.path.insert(0, str(ROOT / 'tools'))
from configs import config
from datasets.base_dataset import BaseDataset
from models.context import DenseASPP, PMPM
from models.fadc import FrequencyAdaptiveConv2d
from models.model_utils import Bag, DAPPM, PAPPM
from models.pidnet import PIDNet, get_pred_model, get_seg_model
from utils.ablation_config import get_boundary_kwargs, get_boundary_loss_kwargs, get_model_kwargs
from utils.criterion import BondaryLoss
from utils.checkpoint import load_model_checkpoint

SEED = 304
BASE = '4c158cf24ce432f0a8cb43364fae38d93cee0dc3'


def raises(error, message, call):
    try:
        call()
    except error as exc:
        assert message in str(exc), str(exc)
        return
    raise AssertionError('Expected {} containing {!r}'.format(error.__name__, message))


def cfg_for(dataset, preset='baseline', *opts):
    args = runner.parse_args(['show', dataset, preset, '20261112-l-smoke', *opts])
    return runner.training_config(args, runner.TEST_ROOT / args.run)


def model_for(cfg):
    torch.manual_seed(SEED)
    return get_seg_model(cfg, imgnet_pretrained=True, load_pretrained=False)


def check_legacy():
    old = ModuleType('models._git_pidnet')
    old.__package__ = 'models'
    source = subprocess.check_output(['git', 'show', BASE + ':models/pidnet.py'], cwd=str(ROOT))
    exec(compile(source, BASE + ':models/pidnet.py', 'exec'), old.__dict__)
    models = (('pidnet_s', (2, 3, 32, 96, 128)),
              ('pidnet_m', (2, 3, 64, 96, 128)),
              ('pidnet_large', (3, 4, 64, 112, 256)))
    x = torch.randn(2, 3, 32, 64)
    for name, (m, n, width, ppm, head) in models:
        for augment in (False, True):
            cfg = config.clone()
            cfg.MODEL.NAME, cfg.MODEL.PRETRAINED = name, ''
            torch.manual_seed(SEED)
            expected = old.PIDNet(m=m, n=n, planes=width, ppm_planes=ppm,
                                  head_planes=head, num_classes=19, augment=augment).eval()
            torch.manual_seed(SEED)
            actual = (get_seg_model(cfg, True) if augment else get_pred_model(name, 19)).eval()
            actual_state, expected_state = actual.state_dict(), expected.state_dict()
            assert actual_state.keys() == expected_state.keys()
            assert all(torch.equal(t, actual_state[k]) for k, t in expected_state.items())
            with torch.no_grad():
                a, b = actual(x), expected(x)
            assert all(torch.equal(l, r) for l, r in zip(a, b)) if augment else torch.equal(a, b)
            del actual, expected, actual_state, expected_state
        print(name, 'default training/prediction keys, initial tensors and CPU outputs exactly match', BASE)


def check_protocol():
    for dataset, (classes, train_count, val_count, _) in runner.PROTOCOL.items():
        base = cfg_for(dataset)
        assert base.DATASET.NUM_CLASSES == classes
        if dataset == 'camvid':
            assert base.TRAIN.END_EPOCH == 121 and not base.MODEL.PRETRAINED
        train = runner.split_rows(ROOT / 'data' / base.DATASET.TRAIN_SET)
        val = runner.split_rows(ROOT / 'data' / base.DATASET.TEST_SET)
        for index in (0, 1):
            assert len({row[index] for row in train}) == train_count == len(train)
            assert len({row[index] for row in val}) == val_count == len(val)
            assert not {row[index] for row in train} & {row[index] for row in val}
        boundary_presets = {
            'valid-edge': ('valid', False, 0.), 'void-mask': ('legacy', True, 0.),
            'dice': ('legacy', False, 1.), 'boundary': ('valid', True, 1.)}
        i_modes = {'fixed1': ('fixed', 1), 'fixed2': ('fixed', 2), 'fadc': ('fadc', 2),
                   'combo-fixed': ('fixed', 2), 'combo-fadc': ('fadc', 2)}
        for preset in runner.PRESETS:
            cfg = cfg_for(dataset, preset)
            options, bd = cfg.ABLATION, cfg.ABLATION.BOUNDARY
            assert (bd.TARGET, bd.MASK_VOID, bd.DICE_WEIGHT) == boundary_presets.get(
                preset, ('legacy', False, 0.))
            assert (options.I_ATROUS.MODE, options.I_ATROUS.RATE) == i_modes.get(
                preset, ('none', 2))
            assert options.I_ATROUS.STAGE == 5 and options.I_ATROUS.BLOCK == 1
            if preset in boundary_presets:
                assert get_model_kwargs(cfg) == get_model_kwargs(base)
        for stage in (3, 4):
            assert get_model_kwargs(cfg_for(dataset, 'fixed2', '--stage', str(stage), '--block', '3'))[
                'atrous_block'] == 3
        raises(ValueError, 'non-first block',
               lambda: cfg_for(dataset, 'fadc', '--stage', '5', '--block', '3'))
        print(dataset, 'all 14 presets, independent boundary, canonical disjoint split',
              train_count, val_count)

    label = np.zeros((16, 16), dtype=np.uint8)
    label[:, 8:12] = 255
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    edges = {}
    for preset in ('baseline', 'valid-edge', 'void-mask', 'dice', 'boundary'):
        cfg = cfg_for('cityscapes', preset)
        kwargs = get_boundary_kwargs(cfg)
        ds = BaseDataset(ignore_label=255, **{k: v for k, v in kwargs.items() if k != 'bd_dilate_size'})
        edges[preset] = ds.gen_sample(image, label, False, False, False, kwargs['bd_dilate_size'])[2]
    assert edges['baseline'].sum() > 0 and not edges['valid-edge'].any()
    assert np.array_equal(edges['baseline'], edges['dice'])
    assert (edges['void-mask'][:, 8:12] == -1).all()
    assert (edges['boundary'][:, 8:12] == -1).all()
    prediction = torch.ones((1, 1, 2, 2), requires_grad=True)
    target = torch.tensor([[[-1., 1.], [0., 0.]]])
    bd = BondaryLoss(**get_boundary_loss_kwargs(cfg_for('cityscapes', 'boundary')))
    bd(prediction, target).backward()
    assert prediction.grad[0, 0, 0, 0] == 0 and torch.isfinite(prediction.grad).all()
    print('Synthetic void stripe: valid-edge removes void boundaries, mask excludes void gradients')


def check_network(device):
    variants = ('baseline', 'pappm', 'dappm', 'pmpm', 'denseaspp', 'fixed1', 'fixed2',
                'fadc', 'combo-fixed', 'combo-fadc') if device == 'cpu' else ('fadc', 'combo-fadc')
    contexts = {'pappm': PAPPM, 'pmpm': PMPM, 'denseaspp': DenseASPP,
                'combo-fixed': PMPM, 'combo-fadc': DenseASPP}
    for dataset, classes in (('cityscapes', 19), ('camvid', 11)):
        for preset in variants:
            opts = ('--stage', '3', '--block', '3') if preset == 'fadc' else ()
            cfg = cfg_for(dataset, preset, *opts)
            model = model_for(cfg).to(device).train()
            assert [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)] == [3, 3, 4, 4]
            assert model.conv1[0].out_channels == 64
            assert model.final_layer.conv1.out_channels == 256
            assert isinstance(model.dfm, Bag)
            assert isinstance(model.spp, contexts.get(preset, DAPPM))
            if isinstance(model.spp, (DAPPM, PAPPM)):
                assert model.spp.scale0[-1].out_channels == 112
            kw = get_model_kwargs(cfg)
            conv = getattr(model, 'layer{}'.format(kw['atrous_stage']))[kw['atrous_block']].conv2
            if kw['atrous_mode'] == 'fixed':
                assert conv.dilation == conv.padding == (kw['atrous_rate'],) * 2
            if kw['atrous_mode'] == 'fadc':
                assert isinstance(conv, FrequencyAdaptiveConv2d) and conv.nominal_rate == 2
            x = torch.randn(2, 3, 64, 64, device=device)
            semantic = torch.randint(classes, (2, 8, 8), device=device)
            edge = torch.rand(2, 8, 8, device=device).round()
            p, main, d = model(x)
            assert tuple(p.shape) == tuple(main.shape) == (2, classes, 8, 8)
            assert tuple(d.shape) == (2, 1, 8, 8)
            bd = BondaryLoss(**get_boundary_loss_kwargs(cfg))
            loss = F.cross_entropy(p, semantic) + F.cross_entropy(main, semantic) + bd(d, edge)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(param.grad is not None and torch.isfinite(param.grad).all()
                       for param in model.parameters())
            if isinstance(conv, FrequencyAdaptiveConv2d) and kw['atrous_stage'] == 3:
                assert torch.count_nonzero(conv.conv_offset.weight.grad)
                assert torch.count_nonzero(conv.conv_mask.weight.grad)
            print(device, dataset, preset, 'B2 64x64; finite gradients', flush=True)
            del model, x, loss, p, main, d, conv
    if device == 'cuda':
        torch.cuda.synchronize()


def parsed(module, path, *opts):
    with patch.object(module, 'config', config.clone()), patch.object(
            sys, 'argv', [module.__name__, '--cfg', str(path), *opts]):
        module.parse_args()
        return module.config


def check_loads(camvid_checkpoint=None):
    train_cli = importlib.import_module('train')
    eval_cli = importlib.import_module('eval')
    custom_cli = importlib.import_module('custom')
    city_cfg = EXP / 'configs/pidnet_large_cityscapes.yaml'
    cam_cfg = EXP / 'configs/pidnet_large_camvid.yaml'
    city = parsed(train_cli, city_cfg)
    cam = parsed(train_cli, cam_cfg, 'ABLATION.CONTEXT', 'pmpm', 'ABLATION.I_ATROUS.MODE', 'fadc')
    # Different source seed and a nondefault BN buffer expose missed loads even
    # for branches whose initial values are otherwise shared between variants.
    torch.manual_seed(SEED + 1)
    state = get_seg_model(city, True, load_pretrained=False).state_dict()
    state['layer4.3.bn2.running_var'].fill_(7.)
    backbone = ('conv1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.')
    image_state = {'module.' + name: v for name, v in state.items()
                   if name.startswith(backbone) and not name.endswith('num_batches_tracked')}
    city_state = {'model.' + name: v for name, v in state.items()}
    with tempfile.TemporaryDirectory(prefix='load-fixtures-', dir=str(EXP / 'logs')) as folder:
        folder = Path(folder)
        marker = folder / 'synthetic.state'
        marker.touch()
        raises(ValueError, 'explicit MODEL.PRETRAINED', lambda: get_seg_model(cam, False))
        raises(FileNotFoundError, 'does not exist', lambda: get_seg_model(city, True))
        city = parsed(train_cli, city_cfg, 'MODEL.PRETRAINED', str(marker))
        cam = parsed(train_cli, cam_cfg, 'MODEL.PRETRAINED', str(marker), 'ABLATION.CONTEXT', 'pmpm',
                     'ABLATION.I_ATROUS.MODE', 'fadc')
        with patch('torch.load', return_value={'state_dict': image_state}):
            torch.manual_seed(SEED)
            model = get_seg_model(city, True)
            assert torch.equal(model.layer3[3].conv2.weight, state['layer3.3.conv2.weight'])
            assert torch.equal(model.layer4[3].bn2.running_var, state['layer4.3.bn2.running_var'])
            del model

        class ReachedTrainingDataset(Exception):
            pass

        with patch.object(train_cli, 'config', config.clone()), patch.object(
                sys, 'argv', ['train.py', '--cfg', str(city_cfg), '--seed', '304',
                             'MODEL.PRETRAINED', str(marker), 'OUTPUT_DIR', str(folder / 'output'),
                             'LOG_DIR', str(folder / 'logs')]), patch.object(
                torch.cuda, 'device_count', return_value=1), patch('torch.load', return_value=image_state) as read, patch.object(
                train_cli.datasets, 'cityscapes', side_effect=ReachedTrainingDataset('fixture stops before data')) as ds:
            raises(ReachedTrainingDataset, 'fixture stops before data', train_cli.main)
            assert read.call_args[0] == (str(marker),)
            assert ds.call_args[1]['list_path'] == 'list/cityscapes/train.lst'
            assert ds.call_args[1]['num_classes'] == 19
        for dataset, path, profile in ((city, city_cfg, 'imagenet-l'), (cam, cam_cfg, 'cityscapes-l')):
            for small in ('pidnet_m', 'pidnet_s'):
                wrong = get_pred_model(small, 19).state_dict()
                if profile == 'imagenet-l':
                    wrong = {name: value for name, value in wrong.items() if name.startswith(backbone)}
                with patch('torch.load', return_value=wrong):
                    reason = 'layer4.3' if profile == 'imagenet-l' else '19-class PIDNet-L'
                    raises(RuntimeError, reason, lambda: get_seg_model(dataset, True))
                del wrong
            raises(ValueError, 'requires a PIDNet-L',
                   lambda: parsed(train_cli, path, 'MODEL.NAME', 'pidnet_m'))
        raises(ValueError, 'requires camvid with 11 classes',
               lambda: parsed(eval_cli, cam_cfg, 'DATASET.NUM_CLASSES', '19'))
        with patch('torch.load', return_value=city_state):
            raises(RuntimeError, 'cannot use an 11/19-class', lambda: get_seg_model(city, True))
        bad = dict(image_state)
        del bad['module.layer4.3.bn2.running_var']
        with patch('torch.load', return_value=bad):
            raises(RuntimeError, 'layer4.3.bn2.running_var', lambda: get_seg_model(city, True))
        bad = dict(city_state)
        bad['model.final_layer.conv2.weight'] = torch.zeros(11, 256, 1, 1)
        with patch('torch.load', return_value=bad):
            raises(RuntimeError, '19-class PIDNet-L', lambda: get_seg_model(cam, False))

        initialized = model_for(cam)
        with patch('torch.load', return_value={'state_dict': city_state}):
            torch.manual_seed(SEED)
            transferred = get_seg_model(cam, False)
        initial_state = initialized.state_dict()
        for name, value in transferred.state_dict().items():
            shared = (not name.startswith(('spp.', 'seghead_', 'final_layer.conv2.', '_i_atrous_signature_'))
                      and name in state)
            assert torch.equal(value, state[name] if shared else initial_state[name]), name
        print('Train config + real loader (in-memory weight fixtures): source checks and exactly shared tensors')
        del initialized, transferred, initial_state

        if camvid_checkpoint:
            local = parsed(train_cli, cam_cfg, 'MODEL.PRETRAINED', str(camvid_checkpoint))
            model = get_seg_model(local, False)
            assert model.final_layer.conv2.out_channels == 11
            assert torch.isfinite(model.layer4[3].conv2.weight).all()
            del model
            print('CamVid transfer reads the supplied 19-class L conversion; no training or split provenance claim:',
                  camvid_checkpoint)

        city = parsed(eval_cli, city_cfg, 'MODEL.PRETRAINED', 'missing-imagenet-l')
        model = get_seg_model(city, True, load_pretrained=False)
        assert load_model_checkpoint(model, {'state_dict': city_state}, strict=True) == len(state)
        fixed = parsed(eval_cli, city_cfg, 'ABLATION.I_ATROUS.MODE', 'fixed')
        raises(RuntimeError, '_i_atrous_signature_',
               lambda: load_model_checkpoint(model_for(fixed), city_state, strict=True))
        wrong_rate = get_pred_model('pidnet_large', 19, atrous_mode='fixed', atrous_rate=1)
        strict_fixed2 = get_pred_model('pidnet_large', 19, atrous_mode='fixed', atrous_rate=2)
        raises(RuntimeError, '_i_atrous_signature_',
               lambda: load_model_checkpoint(strict_fixed2, wrong_rate.state_dict(), prediction=True))
        del model, wrong_rate, strict_fixed2

        with patch.object(custom_cli, 'config', config.clone()), patch.object(
                sys, 'argv', ['custom.py', '--cfg', str(city_cfg)]), patch('sys.stderr', io.StringIO()):
            raises(SystemExit, '2', custom_cli.parse_args)
        with patch.object(custom_cli, 'config', config.clone()), patch.object(
                sys, 'argv', ['custom.py', '--cfg', str(city_cfg), 'TEST.MODEL_FILE', str(marker)]):
            assert custom_cli.parse_args().p == str(marker)
        with patch.object(sys, 'argv', ['custom.py']):
            assert custom_cli.parse_args().p.endswith('PIDNet_L_Cityscapes_test.pt')
        with patch.object(importlib.import_module('configs'), 'config', config.clone()), patch.object(
                sys, 'argv', ['custom.py', '--cfg', str(city_cfg), '--r', str(folder), '--p', str(marker)]), patch(
                'torch.load', return_value={'state_dict': city_state}), patch.object(torch.nn.Module, 'cuda', lambda m: m):
            runpy.run_path(str(ROOT / 'tools/custom.py'), run_name='__main__')
        print('Eval/custom real config and strict weights: wrong geometry, absent custom path, aux-ignore and legacy CLI')


def check_runner():
    with tempfile.TemporaryDirectory(prefix='runner-fixtures-', dir=str(EXP / 'logs')) as base:
        root = Path(base) / 'test'
        root.mkdir()
        run = root / '20261112-runner-fixture'
        name = run.name
        run.mkdir()
        marker = Path(base) / 'pretrain.state'
        marker.touch()
        with patch.object(runner, 'TEST_ROOT', root), redirect_stdout(io.StringIO()) as capture:
            show = ['show', 'cityscapes', 'combo-fixed', name]
            assert runner.main(show) == 0 and not list(run.iterdir())
            assert 'CONTEXT: pmpm' in capture.getvalue()
            for unsafe in ('../20261112-old', '/tmp/20261112-old', '20261112-old/..'):
                raises(ValueError, 'YYYYMMDD-slug', lambda: runner.main([*show[:3], unsafe]))
            alias = root / '20261112-alias'
            alias.symlink_to(run, target_is_directory=True)
            raises(ValueError, 'separate real directory', lambda: runner.main([*show[:3], alias.name]))
            train = ['train', *show[1:], '--pretrained', str(marker)]
            raises(ValueError, 'plan.md', lambda: runner.main(train))
            plan = run / 'plan.md'
            plan.write_text('Synthetic runner behavior fixture; no training or evaluation data.')
            (run / 'output').mkdir()
            raises(ValueError, 'already has files', lambda: runner.main(train))
            (run / 'output').rmdir()
            raises(ValueError, 'batch size', lambda: runner.main([*show, '--batch-size', '1']))
            with patch.object(runner, 'require_training_inputs'), patch.object(runner, 'launch', return_value=0) as child:
                assert runner.main(train) == 0
                _, phase, command = child.call_args[0]
                assert phase == 'train' and command[-2:] == ['--seed', '304']
            raises(ValueError, 'already has files', lambda: runner.main(train))
            saved, cfg = runner.saved_config(run)
            assert cfg.ABLATION.CONTEXT == 'pmpm' and command[-3] == str(saved)
            changed = cfg.clone()
            changed.defrost()
            for field, value in (('TRAIN_SET', 'list/cityscapes/trainval.lst'),
                                 ('TEST_SET', 'list/cityscapes/test.lst')):
                current = getattr(changed.DATASET, field)
                setattr(changed.DATASET, field, value)
                raises(ValueError, 'train and val', lambda: runner.validate_protocol(changed, run))
                setattr(changed.DATASET, field, current)
            changed.MODEL.NAME = 'pidnet_s'
            raises(ValueError, 'PIDNet-L', lambda: runner.validate_protocol(changed, run))
            data = Path(base) / 'data'
            (data / 'list/cityscapes').mkdir(parents=True)
            (data / cfg.DATASET.TEST_SET).write_text((ROOT / 'data/list/cityscapes/test.lst').read_text())
            changed.MODEL.NAME = 'pidnet_large'
            changed.DATASET.ROOT = str(data) + '/'
            raises(ValueError, 'canonical 500 rows', lambda: runner.require_data(changed, ('TEST_SET',)))
            raises(ValueError, 'inside this RUN', lambda: runner.main(['eval', name, '--checkpoint', str(marker)]))
            weight = run / 'output' / 'model.state'
            weight.touch()
            image_dir = Path(base) / 'images'
            image_dir.mkdir()
            (image_dir / 'synthetic.png').touch()
            with patch.object(runner, 'require_data'), patch.object(runner, 'launch', return_value=0) as child:
                assert runner.main(['eval', name, '--checkpoint', str(weight)]) == 0
                _, phase, command = child.call_args[0]
                assert phase == 'eval' and command[-4:] == ['--cfg', str(saved), 'TEST.MODEL_FILE', str(weight)]
                assert runner.main(['custom', name, '--checkpoint', str(weight), '--images', str(image_dir)]) == 0
                _, phase, command = child.call_args[0]
                assert phase == 'custom' and command[command.index('--cfg') + 1] == str(saved)
                assert Path(command[-1]).parent == run / 'output'
            assert runner.launch(run, 'fixture', [sys.executable, '-B', '-c',
                       'import sys;print("fixture stdout");print("fixture stderr", file=sys.stderr);sys.exit(3)']) == 3
            log, = (run / 'logs').glob('cli-fixture.*.log')
            assert 'fixture stdout\n' in log.read_text() and 'fixture stderr\n' in log.read_text()
            before = saved.read_text()
            saved.write_text(before + '\n')
            raises(ValueError, 'configuration changed', lambda: runner.saved_config(run))
            saved.write_text(before)
            (run / 'output' / 'escape').symlink_to(marker)
            raises(ValueError, 'Symlink in RUN', lambda: runner.saved_config(run))
    print('Runner: read-only preview, plan and exclusive claim, canonical lists, path/phase isolation, config hash, real tee')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('legacy', 'protocol', 'cpu', 'cuda', 'load', 'runner'))
    parser.add_argument('--camvid-checkpoint', type=Path, help='optional read-only 19-class L transfer smoke')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    suites = {'legacy': check_legacy, 'protocol': check_protocol, 'cpu': lambda: check_network('cpu'),
              'cuda': lambda: check_network('cuda'), 'load': lambda: check_loads(args.camvid_checkpoint),
              'runner': check_runner}
    suites[args.suite]()
