#!/usr/bin/env python3
"""Preview and launch isolated PIDNet-L ablations from the repository root."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / 'test'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from configs import config
from utils.ablation_config import (get_boundary_loss_kwargs, get_model_kwargs,
                                   get_pretrain_source, register_ablation_config)

PRESETS = ('baseline', 'pappm', 'dappm', 'pmpm', 'denseaspp', 'fixed1', 'fixed2',
           'fadc', 'combo-fixed', 'combo-fadc', 'valid-edge', 'void-mask', 'dice', 'boundary')
PROTOCOL = {'cityscapes': (19, 2975, 500, 'imagenet-l'),
            'camvid': (11, 367, 101, 'cityscapes-l')}
CONFIG_NAME = 'resolved.yaml'


def run_directory(name):
    if not re.fullmatch(r'[0-9]{8}-[a-z0-9]+(?:-[a-z0-9]+)*', name):
        raise ValueError('RUN must be a YYYYMMDD-slug directory name under test/')
    if TEST_ROOT.is_symlink():
        raise ValueError('The repository test/ directory cannot be a symlink')
    run = TEST_ROOT / name
    if run.is_symlink() or run == EXPERIMENT:
        raise ValueError('RUN must be a separate real directory under test/')
    return run


def require_plan(run):
    plan = run / 'plan.md'
    if plan.is_symlink() or not plan.is_file() or plan.stat().st_size == 0:
        raise ValueError('Create a nonempty plan.md in {} before running'.format(run))


def require_fresh(run):
    require_plan(run)
    if any(child.name != 'plan.md' for child in run.iterdir()):
        raise ValueError('RUN already has files or outputs; use a new planned run directory')


def fresh_config(path):
    cfg = config.clone()
    register_ablation_config(cfg)
    cfg.defrost()
    cfg.merge_from_file(str(path))
    return cfg


def validate_protocol(cfg, run):
    dataset = cfg.DATASET.DATASET
    if dataset not in PROTOCOL:
        raise ValueError('Expected Cityscapes or CamVid')
    classes, _, _, source = PROTOCOL[dataset]
    if (cfg.MODEL.NAME != 'pidnet_large' or cfg.DATASET.NUM_CLASSES != classes
            or get_pretrain_source(cfg) != source):
        raise ValueError('RUN must use the matching PIDNet-L experiment config and classes')
    if (cfg.DATASET.TRAIN_SET, cfg.DATASET.TEST_SET, cfg.DATASET.EXTRA_TRAIN_SET) != (
            'list/{}/train.lst'.format(dataset), 'list/{}/val.lst'.format(dataset), ''):
        raise ValueError('RUN must use the train and val lists with no extra training set')
    if cfg.TRAIN.RESUME or cfg.AUTO_RESUME or cfg.TRAIN.BATCH_SIZE_PER_GPU < 2:
        raise ValueError('RUN requires fresh training and a per-GPU batch size of at least 2')
    if dataset == 'camvid' and cfg.TRAIN.END_EPOCH != 121:
        raise ValueError('CamVid RUN must use the effective 121-epoch schedule')
    if (cfg.OUTPUT_DIR, cfg.LOG_DIR) != (str(run / 'output'), str(run / 'logs')):
        raise ValueError('RUN paths must point to its own output/ and logs/')
    get_model_kwargs(cfg)
    get_boundary_loss_kwargs(cfg)


def training_config(args, run):
    cfg = fresh_config(EXPERIMENT / 'configs' / ('pidnet_large_' + args.dataset + '.yaml'))
    opts = cfg.ABLATION
    preset = args.preset
    if preset in ('pappm', 'dappm', 'pmpm', 'denseaspp'):
        opts.CONTEXT = preset
    if preset in ('fixed1', 'fixed2', 'combo-fixed'):
        opts.I_ATROUS.MODE = 'fixed'
        opts.I_ATROUS.RATE = 1 if preset == 'fixed1' else 2
    if preset in ('fadc', 'combo-fadc'):
        opts.I_ATROUS.MODE = 'fadc'
    if preset in ('combo-fixed', 'combo-fadc'):
        opts.CONTEXT = 'pmpm' if preset == 'combo-fixed' else 'denseaspp'
    if preset in ('valid-edge', 'boundary'):
        opts.BOUNDARY.TARGET = 'valid'
    if preset in ('void-mask', 'boundary'):
        opts.BOUNDARY.MASK_VOID = True
    if preset in ('dice', 'boundary'):
        opts.BOUNDARY.DICE_WEIGHT = 1.0
    if args.stage is not None:
        opts.I_ATROUS.STAGE = args.stage
    if args.block is not None:
        opts.I_ATROUS.BLOCK = args.block
    if args.gpus is not None:
        cfg.GPUS = args.gpus
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE_PER_GPU = args.batch_size
    if args.workers is not None:
        cfg.WORKERS = args.workers
    if args.crop_size is not None:
        if any(size < 8 or size % 8 for size in args.crop_size):
            raise ValueError('Crop dimensions must be positive multiples of eight')
        cfg.TRAIN.IMAGE_SIZE = args.crop_size
    data_root = Path(args.data_root) if args.data_root is not None else ROOT / cfg.DATASET.ROOT
    cfg.DATASET.ROOT = str(data_root.resolve()) + os.sep
    if args.pretrained is not None:
        if not args.pretrained.strip():
            raise ValueError('--pretrained must name a checkpoint')
        cfg.MODEL.PRETRAINED = str(Path(args.pretrained).resolve())
    elif cfg.MODEL.PRETRAINED:
        cfg.MODEL.PRETRAINED = str((ROOT / cfg.MODEL.PRETRAINED).resolve())
    cfg.OUTPUT_DIR, cfg.LOG_DIR = str(run / 'output'), str(run / 'logs')
    cfg.freeze()
    validate_protocol(cfg, run)
    return cfg


def split_rows(path):
    with path.open() as stream:
        return [tuple(line.split()) for line in stream]


def require_data(cfg, splits):
    dataset = cfg.DATASET.DATASET
    _, train_count, val_count, _ = PROTOCOL[dataset]
    root = Path(cfg.DATASET.ROOT)
    counts = {'TRAIN_SET': train_count, 'TEST_SET': val_count}
    for field in splits:
        count = counts[field]
        rel = getattr(cfg.DATASET, field)
        expected = split_rows(ROOT / 'data' / rel)
        actual = split_rows(root / rel)
        if (len(actual) != count or any(len(row) != 2 for row in actual)
                or sorted(actual) != sorted(expected)):
            raise ValueError('{} must contain the canonical {} rows'.format(root / rel, count))
        for row in actual:
            for path in row:
                candidate = root / dataset / path
                if not candidate.is_file():
                    raise FileNotFoundError('Missing {} input: {}'.format(field, candidate))


def require_training_inputs(cfg):
    if not cfg.MODEL.PRETRAINED:
        raise ValueError('CamVid training requires --pretrained with a 19-class PIDNet-L Cityscapes checkpoint')
    if not Path(cfg.MODEL.PRETRAINED).is_file():
        raise FileNotFoundError('Training checkpoint not found: {}'.format(cfg.MODEL.PRETRAINED))
    import torch
    if cfg.GPUS != tuple(range(torch.cuda.device_count())) or not cfg.GPUS:
        raise ValueError('GPUS must enumerate all visible CUDA devices from zero; use CUDA_VISIBLE_DEVICES to select them')
    require_data(cfg, ('TRAIN_SET', 'TEST_SET'))


def write_config(run, cfg):
    directory = run / 'configs'
    # Exclusive creation claims this run before any child process starts.
    directory.mkdir()
    (run / 'output').mkdir()
    (run / 'logs').mkdir()
    path = directory / CONFIG_NAME
    payload = cfg.dump().encode('utf-8')
    path.write_bytes(payload)
    path.with_suffix('.sha256').write_text(hashlib.sha256(payload).hexdigest() + '\n')
    return path


def checked_tree(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Expected a real directory: {}'.format(path))
    for directory, dirs, files in os.walk(str(path), followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs + files):
            raise ValueError('Symlink in RUN directory tree: {}'.format(path))


def saved_config(run):
    require_plan(run)
    for part in ('configs', 'output', 'logs'):
        checked_tree(run / part)
    path = run / 'configs' / CONFIG_NAME
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Saved training configuration changed: {}'.format(path))
    cfg = fresh_config(path)
    cfg.freeze()
    validate_protocol(cfg, run)
    return path, cfg


def command_line(command):
    return ' '.join(shlex.quote(str(arg)) for arg in command)


def launch(run, phase, command):
    fd, filename = tempfile.mkstemp(prefix='cli-' + phase + '.', suffix='.log',
                                    dir=str(run / 'logs'), text=True)
    with os.fdopen(fd, 'w') as log:
        line = command_line(command)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()
        with subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True) as child:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return child.wait()


def parse_gpus(value):
    if not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', value):
        raise argparse.ArgumentTypeError('expected a comma-separated list, e.g. 0,1')
    gpus = tuple(map(int, value.split(',')))
    if gpus != tuple(range(len(gpus))):
        raise argparse.ArgumentTypeError('enumerate visible devices consecutively from zero')
    return gpus


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='phase', required=True)
    for phase in ('show', 'train'):
        sub = phases.add_parser(phase, help='preview without writing' if phase == 'show' else 'start a fresh run')
        sub.add_argument('dataset', choices=tuple(PROTOCOL))
        sub.add_argument('preset', choices=PRESETS)
        sub.add_argument('run', metavar='RUN')
        sub.add_argument('--data-root')
        sub.add_argument('--pretrained')
        sub.add_argument('--gpus', type=parse_gpus)
        sub.add_argument('--batch-size', type=int)
        sub.add_argument('--workers', type=int, choices=range(0, 65), metavar='0..64')
        sub.add_argument('--crop-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
        sub.add_argument('--stage', type=int, choices=(3, 4, 5))
        sub.add_argument('--block', type=int)
    for phase in ('eval', 'custom'):
        sub = phases.add_parser(phase, help='reuse the saved training configuration')
        sub.add_argument('run', metavar='RUN')
        sub.add_argument('--checkpoint', required=True, help='checkpoint file under RUN/output/')
        if phase == 'custom':
            sub.add_argument('--images', required=True)
            sub.add_argument('--extension', default='.png')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run = run_directory(args.run)
    command = [sys.executable, '-B', '-u', 'tools/' + (
        'train' if args.phase == 'show' else args.phase) + '.py']
    if args.phase in ('show', 'train'):
        cfg = training_config(args, run)
        path = run / 'configs' / CONFIG_NAME
        command += ['--cfg', str(path), '--seed', '304']
        if args.phase == 'show':
            print(command_line(command))
            print(cfg.dump(), end='')
            return 0
        require_fresh(run)
        require_training_inputs(cfg)
        write_config(run, cfg)
    else:
        path, cfg = saved_config(run)
        checkpoint = Path(args.checkpoint).resolve()
        try:
            checkpoint.relative_to(run / 'output')
        except ValueError:
            raise ValueError('--checkpoint must be inside this RUN output/') from None
        if not checkpoint.is_file():
            raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint))
        command += ['--cfg', str(path)]
        if args.phase == 'eval':
            require_data(cfg, ('TEST_SET',))
            command += ['TEST.MODEL_FILE', str(checkpoint)]
        else:
            images = Path(args.images).resolve()
            if not images.is_dir():
                raise FileNotFoundError('Image directory not found: {}'.format(images))
            if not re.fullmatch(r'\.[a-zA-Z0-9]+', args.extension):
                raise ValueError('Extension must look like .png or .jpg')
            if not any(images.glob('*' + args.extension)):
                raise ValueError('No {} images found in {}'.format(args.extension, images))
            output = tempfile.mkdtemp(prefix='custom-', dir=str(run / 'output'))
            command += ['--p', str(checkpoint), '--r', str(images), '--t', args.extension,
                        '--output-dir', output]
    return launch(run, args.phase, command)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        sys.exit('Error: {}'.format(error))
