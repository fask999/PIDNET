"""CPU-only, synthetic probes of CLI routing, label conversion and boundary width."""

import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib
import io
import logging
from pathlib import Path
import os
import runpy
import shlex
import subprocess
import sys
from unittest import mock
import warnings
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OLD = ROOT / 'test/20261111-pidnet-ablation-switches'
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
os.chdir(str(ROOT))

import configs
from datasets import base_dataset
from datasets.camvid import CamVid
from datasets.cityscapes import Cityscapes
from models import pidnet
import train as train_cli
from utils import checkpoint
from utils.ablation_config import (get_boundary_kwargs, get_boundary_loss_kwargs,
                                   get_model_kwargs, register_ablation_config)
from utils.criterion import BondaryLoss, OhemCrossEntropy
from utils.utils import get_confusion_matrix


def no_cuda(self, *args, **kwargs):
    return self


def probe_custom():
    with mock.patch('builtins.open', mock.mock_open(read_data='')):
        camvid = CamVid('/synthetic/', 'list/camvid/val.lst')
    expected_color = tuple(camvid.color_list[0])
    original_interpolate = torch.nn.functional.interpolate

    class Predictor:
        cuda = eval = no_cuda

        def __call__(self, tensor):
            output = torch.zeros(1, 11, 2, 2)
            output[:, 0] = 2
            return output

    for explicit_p in (None, 'explicit-camvid.pt'):
        local = configs.config.clone()
        argv = ['tools/custom.py', '--cfg', str(OLD / 'configs/pidnet_small_camvid.yaml')]
        if explicit_p:
            argv += ['--p', explicit_p]
        argv += ['--r', '/synthetic', '--output-dir', str(EXP / 'tests/unused-custom-output'),
                 'TEST.MODEL_FILE', 'configured-camvid.pt', 'MODEL.ALIGN_CORNERS', 'False']
        observed_color = []
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, 'argv', argv))
            stack.enter_context(mock.patch.object(configs, 'config', local))
            build = stack.enter_context(mock.patch.object(pidnet, 'get_pred_model', return_value=Predictor()))
            load = stack.enter_context(mock.patch.object(checkpoint, 'load_model_checkpoint', return_value=11))
            interp = stack.enter_context(mock.patch.object(torch.nn.functional, 'interpolate',
                                                            wraps=original_interpolate))
            stack.enter_context(mock.patch('glob.glob', return_value=['/synthetic/frame.png']))
            stack.enter_context(mock.patch.object(cv2, 'imread', return_value=np.zeros((5, 5, 3), np.uint8)))
            stack.enter_context(mock.patch.object(torch.Tensor, 'cuda', no_cuda))
            stack.enter_context(mock.patch.object(os, 'makedirs'))
            stack.enter_context(mock.patch.object(Image.Image, 'save',
                lambda image, path: observed_color.append(tuple(np.asarray(image)[0, 0]))))
            runpy.run_path(str(ROOT / 'tools/custom.py'), run_name='__main__')
        used_path = load.call_args[0][1]
        actual_align = interp.call_args[1]['align_corners']
        n_classes = build.call_args[0][1]
        print('custom: classes={} config.path={} load.path={} config.align={} resize.align={} rgb={} camvid.rgb={}'.format(
            n_classes, local.TEST.MODEL_FILE, used_path, local.MODEL.ALIGN_CORNERS, actual_align,
            observed_color[0], expected_color))
        assert n_classes == 11 and used_path == (explicit_p or '../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt')
        assert load.call_args[1] == {'strict': True, 'prediction': True}
        assert local.MODEL.ALIGN_CORNERS is False and actual_align is True
        assert observed_color == [(128, 64, 128)] and expected_color == (0, 128, 192)


def probe_wrapper():
    out = subprocess.check_output(
        ['bash', str(OLD / 'scripts/run_ablation.sh'), 'show', 'cityscapes', 'baseline',
         'OUTPUT_DIR', str(EXP / 'tests/requested-out'), 'LOG_DIR', str(EXP / 'tests/requested-log')],
        text=True, env=dict(os.environ, PIDNET_PYTHON=sys.executable, PIDNET_RUN_TAG='audit')).strip()
    fields = shlex.split(out)
    cfg_file = fields[fields.index('--cfg') + 1]
    options = fields[fields.index('--seed') + 2:]
    local = register_ablation_config(configs.config.clone())
    configs.update_config(local, type('Args', (), {'cfg': cfg_file, 'opts': options}))
    print('wrapper show:', out)
    print('wrapper effective: OUTPUT_DIR={} LOG_DIR={}'.format(local.OUTPUT_DIR, local.LOG_DIR))
    assert local.OUTPUT_DIR == 'test/20261111-pidnet-ablation-switches/output/cityscapes/baseline-audit'
    assert local.LOG_DIR == 'test/20261111-pidnet-ablation-switches/logs/cityscapes/baseline-audit'


class FakeDataset:
    class_weights = None

    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 2


class FakeDataParallel:
    def __init__(self, model, **kwargs):
        self.module = model

    cuda = no_cuda

    def named_parameters(self):
        return self.module.named_parameters()


def probe_resume(has_resume, pretrained, new_miou=0.45, simulate_not_resume=False, resume_blob=None):
    local = register_ablation_config(configs.config.clone())
    local.defrost()
    local.GPUS = (0,)
    local.MODEL.PRETRAINED = str(EXP / 'tests/not-present-pretrained.pth') if pretrained else ''
    if resume_blob is not None:
        local.MODEL.NAME = 'pidnet_l'
    local.TRAIN.RESUME = True
    local.TRAIN.END_EPOCH = 1
    local.TRAIN.BATCH_SIZE_PER_GPU = 2
    local.WORKERS = 0
    local.freeze()
    final_output = EXP / 'tests/unused-resume-output'
    path = final_output / 'checkpoint.pth.tar'
    original_isfile = os.path.isfile
    original_build = pidnet.get_seg_model
    events = []

    def model(cfg, *args, **kwargs):
        if simulate_not_resume:
            kwargs['load_pretrained'] = not cfg.TRAIN.RESUME
        events.append(('build', kwargs.get('load_pretrained', True)))
        return original_build(cfg, *args, **kwargs)

    def load_file(file, **kwargs):
        events.append(('read', Path(file).name))
        if Path(file) == path and has_resume:
            if resume_blob is not None:
                return resume_blob
            return {'epoch': 0, 'best_mIoU': 0.35, 'optimizer': {}, 'state_dict': {}}
        raise FileNotFoundError(str(file))

    def save_file(value, file):
        events.append(('save', Path(file).name, value.get('best_mIoU')))

    error = None
    with ExitStack() as stack:
        def patch(obj, key, replacement):
            stack.enter_context(mock.patch.object(obj, key, replacement))

        patch(train_cli, 'config', local)
        patch(train_cli, 'parse_args', lambda: type('Args', (), {'seed': 304, 'cfg': 'synthetic.yaml'}))
        patch(train_cli, 'create_logger', lambda *a: (mock.Mock(), str(final_output), str(final_output)))
        patch(train_cli, 'SummaryWriter', lambda *a: mock.Mock())
        patch(train_cli.datasets, 'cityscapes', FakeDataset)
        patch(train_cli, 'train', lambda *args: events.append(('train', args[1])))
        patch(train_cli, 'validate', lambda *a: (3.0, new_miou, []))
        patch(train_cli, 'load_model_checkpoint', lambda *a, **kw: events.append(('restore', kw['strict'])))
        patch(os.path, 'isfile', lambda file: has_resume if Path(file) == path else original_isfile(file))
        patch(pidnet, 'get_seg_model', model)
        patch(torch.cuda, 'device_count', lambda: 1)
        patch(torch.nn, 'DataParallel', FakeDataParallel)
        patch(torch.optim, 'SGD', lambda *a, **kw: mock.Mock(state_dict=lambda: {}))
        patch(torch, 'load', load_file)
        patch(torch, 'save', save_file)
        try:
            train_cli.main()
        except FileNotFoundError as exc:
            error = type(exc).__name__
        except KeyError as exc:
            if resume_blob is None:
                raise
            error = '{} {}'.format(type(exc).__name__, exc)
    print('train resume: resume_file={} pretrained_set={} simulate_not_resume={} new_mIoU={} error={} events={}'.format(
        has_resume, pretrained, simulate_not_resume, new_miou, error, events))
    return error, events


def probe_label():
    raw = np.array([[255, 7, 8, 0], [7, 8, 11, 255]], dtype=np.uint8)
    ok, png = cv2.imencode('.png', raw)
    assert ok
    decoded = cv2.imdecode(png, cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(decoded, raw)

    class SimulatedBroadenedTransform(Cityscapes):
        def label_transform(self, label):
            converted = super().label_transform(label)
            if self.ignore_label < 0:
                return np.where(converted == 255, -1, converted)
            return converted

    for ignore, dataset_type in ((255, Cityscapes), (-1, Cityscapes), (-2, Cityscapes),
                                 (-2, SimulatedBroadenedTransform)):
        with mock.patch('builtins.open', mock.mock_open(read_data='image.png label.png\n')), \
             mock.patch.object(torch.Tensor, 'cuda', no_cuda):
            dataset = dataset_type('/synthetic/', 'list/cityscapes/val.lst', ignore_label=ignore,
                                   flip=False, multi_scale=True, base_size=6, crop_size=(4, 7),
                                   bd_target='valid', bd_mask_void=True, bd_dilate_size=3)
        with mock.patch.object(cv2, 'imread', side_effect=[np.zeros((2, 4, 3), np.uint8), decoded.copy()]), \
             mock.patch.object(base_dataset.random, 'randint', side_effect=[5, 0, 0]):
            _, label, edge, _, _ = dataset[0]
        err = None
        try:
            score = torch.zeros(1, 19, *label.shape)
            sem = OhemCrossEntropy(ignore_label=ignore)._ohem_forward(score, torch.from_numpy(label).long()[None])
        except IndexError as exc:
            err = str(exc)
            sem = None
        print('label: ignore={} variant={} dtype={} shape={} raw255={} raw0={} pad={} raw255.edge={} CE={} error={}'.format(
            ignore, dataset_type.__name__, label.dtype, label.shape, label[0, 0], label[0, -2],
            label[-1, -1], edge[0, 0], sem, err))
        broken = ignore == -2 and dataset_type is Cityscapes
        assert label.shape == (4, 7) and label[0, -2] == ignore and label[-1, -1] == ignore
        assert label[0, 0] == (255 if broken else ignore)
        assert (edge[0, 0] != -1) if broken else (edge[0, 0] == -1)
        assert (err is not None) == broken


def probe_width():
    dataset = base_dataset.BaseDataset()
    label = np.zeros((32, 32), dtype=np.uint8)
    label[:, 16:] = 1
    valid = np.ones_like(label, dtype=bool)
    for k in (1, 3, 5, 7):
        support = np.flatnonzero(dataset._valid_edge(label, valid, k)[16])
        print('valid width setting={} columns={} pixels={}'.format(k, support.tolist(), support.size))
        assert support.size == k + 1
    legacy = dataset._legacy_edge(label, edge_pad=True, edge_size=4)
    valid[:, 14] = False
    clipped = np.flatnonzero(dataset._valid_edge(label, valid, 5)[16])
    print('legacy default straight-edge middle width={}; valid width=5 with void col14: columns={}'.format(
        int(legacy[16].sum()), clipped.tolist()))
    assert int(legacy[16].sum()) == 4 and clipped.size == 4
    local = register_ablation_config(configs.config.clone())
    assert get_boundary_kwargs(local)['bd_dilate_size'] == 4
    local.ABLATION.BOUNDARY.TARGET = 'valid'
    assert get_boundary_kwargs(local)['bd_dilate_size'] == 3


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def get_error(run, exc_type):
    try:
        run()
    except exc_type as exc:
        return str(exc)
    raise AssertionError('Expected {} but call returned normally'.format(exc_type))


def probe_mmseg():
    weight = ROOT / 'pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar'
    before = sha256(weight)
    # No general pickle fallback. The file is a zip tensor checkpoint in this environment.
    assert zipfile.is_zipfile(str(weight))
    payload = torch.load(str(weight), map_location='cpu', weights_only=True)
    raw = payload['state_dict']
    assert all(torch.is_tensor(t) for t in raw.values())
    source = {checkpoint._model_key(k): t for k, t in raw.items()}
    missing_bias = {'conv1.0.bias', 'conv1.3.bias'}
    assert len(raw) == len(source) == 543 and set(payload) == {'state_dict', 'epoch'}
    print('mmseg asset: bytes={} sha256={} envelope={} state_entries={} wrapper=model.'.format(
        weight.stat().st_size, before, sorted(payload), len(raw)))
    assert all(k.startswith('model.') for k in raw)

    local = register_ablation_config(configs.config.clone())
    local.defrost()
    local.MODEL.NAME = 'pidnet_l'
    local.MODEL.PRETRAINED = local.TEST.MODEL_FILE = str(weight)
    local.freeze()
    seed = 304
    torch.manual_seed(seed)
    eval_target = pidnet.get_seg_model(local, imgnet_pretrained=True, load_pretrained=False)
    eval_state = eval_target.state_dict()
    assert len(eval_state) == 545 and set(eval_state) - set(source) == missing_bias
    assert all(t.shape == eval_state[k].shape for k, t in source.items())
    initial_bias = {k: eval_state[k].clone() for k in missing_bias}

    def return_verified(path, **kwargs):
        assert Path(path) == weight
        return payload

    # The real builder and checkpoint code run, with the previously verified
    # weights-only read substituted for the CLI's generic disk deserialization.
    capture = io.StringIO()
    logger = logging.getLogger()
    old_level = logger.level
    handler = logging.StreamHandler(capture)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        torch.manual_seed(seed)
        with mock.patch.object(torch, 'load', side_effect=return_verified) as read:
            partial = pidnet.get_seg_model(local, imgnet_pretrained=False)
        assert read.call_count == 1
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    partial_state = partial.state_dict()
    assert all(torch.equal(partial_state[k], t) for k, t in source.items())
    assert all(torch.equal(partial_state[k], initial_bias[k]) for k in missing_bias)
    assert all(torch.count_nonzero(partial_state[k]) == partial_state[k].numel() for k in missing_bias)
    message = capture.getvalue().strip()
    assert message == 'Loaded 543 matching pretrained model tensors'
    print('train PRETRAINED strict=False: 543/545; initial bias retained max_abs={}; log={!r}'.format(
        {k: float(partial_state[k].abs().max()) for k in sorted(missing_bias)}, message))
    del partial_state, partial

    eval_cli = importlib.import_module('eval')
    eval_value = eval_target.conv1[0].weight.detach().clone()
    with mock.patch.object(eval_cli, 'config', local), \
         mock.patch.object(eval_cli, 'parse_args', return_value=argparse.Namespace(cfg='synthetic.yaml')), \
         mock.patch.object(eval_cli, 'create_logger', return_value=(mock.Mock(), str(EXP / 'tests/unused'), '')), \
         mock.patch.object(pidnet, 'get_seg_model', return_value=eval_target) as build, \
         mock.patch.object(torch, 'load', side_effect=return_verified) as read:
        error = get_error(eval_cli.main, RuntimeError)
    assert build.call_args[1]['load_pretrained'] is False and read.call_count == 1
    assert all(k in error for k in missing_bias) and torch.equal(eval_target.conv1[0].weight, eval_value)
    print('eval strict 543/545: {}'.format(error))

    zeros = {'model.' + k: torch.zeros_like(eval_state[k]) for k in missing_bias}
    simulated = {'state_dict': dict(raw, **zeros)}
    assert checkpoint.load_model_checkpoint(eval_target, simulated, strict=True) == 545
    conv = eval_target.conv1[0]
    x = torch.linspace(-1.0, 1.0, 3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.no_grad():
        neutral_diff = (conv._conv_forward(x, conv.weight, None) - conv(x)).abs().max().item()
    assert neutral_diff == 0.0
    print('in-memory zero-bias variant: eval strict 545/545, isolated stem conv without bias maxdiff={:.6g}'.format(neutral_diff))
    del eval_target, eval_state, conv

    pred = pidnet.get_pred_model('pidnet_l', 19)
    pred_state = pred.state_dict()
    assert len(pred_state) == 519 and set(pred_state) - set(source) == missing_bias
    assert sum(k in pred_state and t.shape == pred_state[k].shape for k, t in source.items()) == 517
    assert len(set(source) - set(pred_state)) == 26 and all(
        k.startswith(('seghead_p.', 'seghead_d.')) for k in set(source) - set(pred_state))
    with mock.patch.object(sys, 'argv', ['tools/custom.py', '--a', 'pidnet-l', '--p', str(weight)]), \
         mock.patch('glob.glob', return_value=[]), \
         mock.patch.object(pidnet, 'get_pred_model', return_value=pred), \
         mock.patch.object(torch, 'load', side_effect=return_verified) as read:
        error = get_error(lambda: runpy.run_path(str(ROOT / 'tools/custom.py'), run_name='__main__'), RuntimeError)
    assert read.call_count == 1 and all(k in error for k in missing_bias)
    print('custom strict 517/519 (26 auxiliary tensors excluded): {}'.format(error))

    old_source = subprocess.check_output(['git', 'show', '4c158cf:tools/custom.py'], cwd=str(ROOT))
    namespace = {'__name__': 'audit_head_custom'}
    exec(compile(old_source, '4c158cf:tools/custom.py', 'exec'), namespace)
    pred_initial_bias = {k: pred_state[k].clone() for k in missing_bias}
    capture = io.StringIO()
    with mock.patch.object(torch, 'load', side_effect=return_verified), redirect_stdout(capture):
        namespace['load_pretrained'](pred, str(weight))
    assert 'Loaded 517 parameters!' in capture.getvalue()
    assert all(torch.equal(pred.state_dict()[k], pred_initial_bias[k]) for k in missing_bias)
    print('HEAD custom: count-only 517/519, both initialized biases unchanged')
    assert checkpoint.load_model_checkpoint(pred, simulated, strict=True, prediction=True) == 519
    print('in-memory zero-bias variant: custom strict 519/519; no checkpoint written')
    del pred, pred_state

    err, events = probe_resume(True, False, resume_blob=payload)
    assert err == "KeyError 'best_mIoU'" and events == [('build', True), ('read', 'checkpoint.pth.tar')]
    assert sha256(weight) == before
    print('train RESUME: actual source envelope errors before strict model restore; source file SHA-256 unchanged')


def probe_dense_nested():
    from models.context import DenseASPP
    source, dest = torch.nn.Module(), torch.nn.Module()
    source.spp = DenseASPP(4, 2, 4, rates=(1, 2, 3, 4))
    dest.spp = DenseASPP(4, 2, 4, rates=(2, 4, 6, 8))
    with mock.patch.object(dest.spp, 'load_state_dict', side_effect=RuntimeError('public method guard')) as guard:
        dest.load_state_dict(source.state_dict(), strict=True)
        assert guard.call_count == 0
        assert dest.spp._dense_rates.tolist() == [1, 2, 3, 4]
        assert [m[-1].dilation[0] for m in dest.spp.layers] == [2, 4, 6, 8]
        error = get_error(lambda: dest.spp.load_state_dict(source.spp.state_dict()), RuntimeError)
    print('Dense child public load_state_dict: direct guard={!r}; parent strict guard_calls=0, rates overwritten, dilation unchanged'.format(error))


def probe_direct_width():
    palette = np.zeros((18, 18, 3), dtype=np.uint8)
    palette[:, :9] = [0, 128, 192]
    palette[:, 9:] = [128, 0, 0]
    raw = np.full((18, 18), 7, np.uint8)
    raw[:, 9:] = 8
    cfg = register_ablation_config(configs.config.clone())
    cfg.ABLATION.BOUNDARY.TARGET = 'valid'
    assert cfg.ABLATION.BOUNDARY.WIDTH == 0
    width = get_boundary_kwargs(cfg)['bd_dilate_size']
    assert width == 3
    for dataset_type, name in ((Cityscapes, 'cityscapes'), (CamVid, 'camvid')):
        for specified in (None, 4, width):
            kwargs = {} if specified is None else {'bd_dilate_size': specified}
            with mock.patch('builtins.open', mock.mock_open(read_data='image.png label.png\n')), \
                 mock.patch.object(torch.Tensor, 'cuda', no_cuda):
                dataset = dataset_type('/synthetic/', 'list/{}/val.lst'.format(name),
                                       multi_scale=False, flip=False, bd_target='valid', **kwargs)
            with mock.patch.object(cv2, 'imread', side_effect=[np.zeros_like(palette), raw.copy()]), \
                 mock.patch.object(Image, 'open', side_effect=[Image.fromarray(np.zeros_like(palette)),
                                                               Image.fromarray(palette)]):
                if specified == width:
                    band = int(dataset[0][2][9].sum())
                    assert band == 4
                    outcome = 'getitem band=' + str(band)
                else:
                    outcome = get_error(lambda: dataset[0], ValueError)
                    assert outcome == 'valid boundary width must be a positive odd integer'
            print('{} valid direct: arg={} constructor ok (effective={}); {}'.format(
                dataset_type.__name__, specified, dataset.bd_dilate_size, outcome))
    print('config WIDTH=0: valid auto=3; legacy auto=4 (covered in first probe)')


def probe_dice_contract():
    cfg = register_ablation_config(configs.config.clone())
    cfg.ABLATION.BOUNDARY.DICE_WEIGHT = 0.5
    assert get_boundary_loss_kwargs(cfg) == {'dice_weight': 0.5, 'mask_void': False}
    x = torch.zeros(1, 1, 2, 2, requires_grad=True)
    invalid = torch.full((1, 2, 2), -1.0)
    unchecked = BondaryLoss(**get_boundary_loss_kwargs(cfg))(x, invalid)
    masked = BondaryLoss(dice_weight=0.5, mask_void=True)(x, invalid)
    assert unchecked.item() == -1 and masked.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(masked, x)[0]) == 0
    clamped = BondaryLoss(dice_weight=0.5)(x, invalid.clamp(min=0))
    assert clamped.item() > 0 and torch.count_nonzero(torch.autograd.grad(clamped, x)[0]) == 4
    print('Dice mask=False cfg independent: four -1 targets -> loss={}; matching mask -> {}; clamped to background -> {} with nonzero grad'.format(
        unchecked.item(), masked.item(), clamped.item()))
    for mask in (False, True):
        multichannel = get_error(lambda: BondaryLoss(dice_weight=0.5, mask_void=mask)(
            torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2)), IndexError)
        print('boundary C=2 target=NHW mask={} -> {}'.format(mask, multichannel))


def probe_custom_edge_cases():
    def run(argv, files, image):
        local = configs.config.clone()
        output = io.StringIO()
        fake_model = mock.Mock()
        fake_model.cuda.return_value = fake_model
        with mock.patch.object(configs, 'config', local), \
             mock.patch.object(sys, 'argv', ['tools/custom.py'] + argv), \
             mock.patch.object(pidnet, 'get_pred_model', return_value=fake_model) as build, \
             mock.patch.object(checkpoint, 'load_model_checkpoint', return_value=1) as load, \
             mock.patch('glob.glob', return_value=files), \
             mock.patch.object(cv2, 'imread', return_value=image) as read, \
             mock.patch.object(Image.Image, 'save') as save, redirect_stdout(output):
            error = None
            try:
                runpy.run_path(str(ROOT / 'tools/custom.py'), run_name='__main__')
            except AttributeError as exc:
                error = str(exc)
        assert load.call_count == 1 and save.call_count == 0
        return build.call_args[0], read.call_count, error, output.getvalue().strip()

    args, reads, error, out = run(['--a', 'pidnet-s'], ['/synthetic/broken.png'], None)
    assert reads == 1 and error is not None and 'NoneType' in error and 'broken.png' not in error
    print('custom bad image: imread=None -> AttributeError {!r}, filename absent'.format(error))
    args, reads, error, out = run(['--a', 'pidnet-s', '--c', 'False'], [], None)
    assert reads == 0 and error is None and args == ('pidnet-s', 19)
    assert out == 'Loaded 1 model tensors'
    print('custom empty image list: normal return, 0 reads/saves, only load message; --c False built 19 classes')
    cam = OLD / 'configs/pidnet_small_camvid.yaml'
    args, reads, error, out = run(['--cfg', str(cam), '--a', 'pidnet-l'], [], None)
    assert reads == 0 and error is None and args == ('pidnet_s', 11)
    print('custom --cfg CamVid + --a pidnet-l: builder {} without conflict notice'.format(args))


def probe_name_and_numpy():
    for name in ('pidnet_s', 'pidnet_m', 'pidnet_l', 'pidnet_l_cityscapes', 'unknown'):
        local = register_ablation_config(configs.config.clone())
        local.MODEL.NAME = name
        with mock.patch.object(pidnet, 'PIDNet') as build:
            pidnet.get_seg_model(local, False, load_pretrained=False)
            train = build.call_args[1]['planes'], build.call_args[1]['n']
            pidnet.get_pred_model(name, 19)
            pred = build.call_args[1]['planes'], build.call_args[1]['n']
        local.ABLATION.I_ATROUS.STAGE = 3
        local.ABLATION.I_ATROUS.BLOCK = 3
        try:
            get_model_kwargs(local)
            accepts = True
        except ValueError:
            accepts = False
        assert train == pred and accepts == (train[1] == 4)
        print('model name {}: train/pred planes,n={}; cfg stage3 block3 accepted={}'.format(name, train, accepts))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        confusion = get_confusion_matrix(torch.zeros(1, 2, 2, dtype=torch.long),
                                         torch.zeros(1, 2, 2, 2), (1, 2, 2), 2)
        minutes = np.int(1.7)
    warnings_for_alias = [w for w in caught if 'np.int' in str(w.message)]
    assert confusion[0, 0] == 4 and minutes == 1 and warnings_for_alias
    print('NumPy {}: current confusion_matrix={} np.int(1.7)={} warnings={}'.format(
        np.__version__, confusion.tolist(), minutes, [w.category.__name__ for w in warnings_for_alias]))
    other = subprocess.check_output(['/home/robot/anaconda3/bin/python', '-B', '-c',
        'import numpy as n; print(n.__version__); print(getattr(n, "int", "AttributeError: removed"))'],
        cwd=str(ROOT), text=True).strip().splitlines()
    print('existing base Python NumPy check: version={}, np.int={}; no application run'.format(*other))
    assert 'AttributeError' in other[-1]


def probe_fadc_cpu_half():
    from models import fadc
    module = fadc.FrequencyAdaptiveConv2d(torch.nn.Conv2d(3, 3, 3, padding=1), 2).half()
    half_x = torch.zeros(1, 3, 8, 8, dtype=torch.float16)
    with mock.patch.object(fadc, 'deform_conv2d', wraps=fadc.deform_conv2d) as deform:
        error = get_error(lambda: module(half_x), RuntimeError)
    print('FADC CPU Half native: deform_calls={} error={!r}'.format(deform.call_count, error))
    assert deform.call_count == 0 and 'not implemented for' in error and 'Half' in error
    assert 'needs torchvision' not in error
    with mock.patch.object(fadc, 'deform_conv2d',
                           side_effect=RuntimeError("slow_conv2d_cpu not implemented for 'Half'")) as fake:
        translated = get_error(lambda: module.float()(half_x.float()), RuntimeError)
    assert fake.call_count == 1 and 'needs torchvision' in translated
    print('FADC synthetic error injected inside deform (FP32 inputs): translated={!r}'.format(translated))


def probe_supplement():
    print('SUPPLEMENT CPU ONLY: torch {} numpy {} seed for weights 304'.format(torch.__version__, np.__version__))
    probe_mmseg()
    probe_dense_nested()
    probe_direct_width()
    probe_dice_contract()
    probe_custom_edge_cases()
    probe_name_and_numpy()
    probe_fadc_cpu_half()
    print('SUPPLEMENT COMPLETE; original probes not repeated; no checkpoint written.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement', action='store_true', help='run only the added CPU audit')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.supplement:
        probe_supplement()
        sys.exit(0)
    probe_custom()
    probe_wrapper()
    err, events = probe_resume(has_resume=True, pretrained=True)
    assert err == 'FileNotFoundError' and events == [('build', True), ('read', 'not-present-pretrained.pth')]
    err, events = probe_resume(has_resume=False, pretrained=True, simulate_not_resume=True)
    assert err is None and ('train', 0) in events and not any(e[0] == 'restore' for e in events)
    err, events = probe_resume(has_resume=True, pretrained=False, new_miou=0.40)
    assert err is None and ('restore', True) in events
    assert ('save', 'checkpoint.pth.tar', 0.35) in events and ('save', 'best.pt', None) in events
    probe_label()
    probe_width()
    print('tools probes passed; custom/train I/O and training were stubbed; actual parse, CLI flow, conversion and width exercised.')
