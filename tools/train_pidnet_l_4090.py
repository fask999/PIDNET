#!/usr/bin/env python3
"""Launch fresh PIDNet-L training runs on one selected desktop RTX 4090."""
import argparse
import datetime
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / 'test'
TEMPLATES = TEST_ROOT / '20261112-pidnet-l-4090-launcher' / 'templates'
OLD_RUNNER = TEST_ROOT / '20261112-pidnet-l-ablation' / 'scripts' / 'run_ablation.py'
PRESETS = ('baseline', 'pappm', 'dappm', 'pmpm', 'denseaspp', 'fixed1', 'fixed2',
           'fadc', 'combo-fixed', 'combo-fadc', 'valid-edge', 'void-mask', 'dice', 'boundary')
I_SUITE = ('baseline', 'pappm', 'pmpm', 'denseaspp', 'fixed1', 'fixed2', 'fadc',
           'combo-fixed', 'combo-fadc')


def nonnegative(value):
    try:
        result = int(value)
        if result < 0:
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cityscapes', 'camvid'), default='cityscapes')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preset', choices=PRESETS, help='single preset (default: baseline)')
    group.add_argument('--i-suite', action='store_true', help='run nine I-suite presets serially; stop on failure')
    parser.add_argument('--run', default=datetime.date.today().strftime('%Y%m%d') + '-pidnet-l-' + str(os.getpid()),
                        help='fresh YYYYMMDD-slug under test/ (default: date-pidnet-l-PID)')
    parser.add_argument('--gpu', type=nonnegative, default=0, help='physical GPU index in PCI bus order')
    parser.add_argument('--batch-size', type=nonnegative, default=2)
    parser.add_argument('--workers', type=nonnegative)
    parser.add_argument('--data-root', help='data root containing canonical lists and referenced images/labels')
    parser.add_argument('--pretrained', help='CamVid requires explicit trusted 19-class Cityscapes-L checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='read-only preview; no CUDA/data/weight checks')
    parser.add_argument('--stage', type=int, choices=(3, 4, 5))
    parser.add_argument('--block', type=nonnegative)
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error('--batch-size must be at least 2')
    if args.workers is not None and args.workers > 64:
        parser.error('--workers must be at most 64')
    return args


def targets(args):
    if not re.fullmatch(r'[0-9]{8}-[a-z0-9]+(?:-[a-z0-9]+)*', args.run):
        raise ValueError('--run must be YYYYMMDD-slug')
    if TEST_ROOT.is_symlink() or not TEST_ROOT.is_dir():
        raise ValueError('test/ must be a real directory')
    presets = I_SUITE if args.i_suite else (args.preset or 'baseline',)
    result = [(p, TEST_ROOT / (args.run + '-' + p if args.i_suite else args.run)) for p in presets]
    for _, run in result:
        if os.path.lexists(str(run)):
            raise ValueError('run path already exists: {}'.format(run))
    return result


def load_runner(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('pidnet_l_ablation_runner', str(OLD_RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_args(args, preset):
    return argparse.Namespace(dataset=args.dataset, preset=preset, stage=args.stage, block=args.block,
                              gpus=(0,), batch_size=args.batch_size, workers=args.workers,
                              crop_size=None, data_root=args.data_root, pretrained=args.pretrained)


def command(run):
    return [sys.executable, '-B', '-u', 'tools/train.py', '--cfg',
            str(run / 'configs' / 'resolved.yaml'), '--seed', '304']


def display_command(cmd, gpu):
    return 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={} {}'.format(
        shlex.quote(str(gpu)), ' '.join(shlex.quote(str(arg)) for arg in cmd))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def desktop_4090(name):
    return bool(re.fullmatch(r'(?:NVIDIA\s+)?GeForce RTX 4090', name.strip(), re.IGNORECASE))


def preflight(cfg, runner, facts, preset, interrupted=lambda: False):
    def record(label, value):
        facts.append('- {}: {}'.format(label, value))
        if interrupted():
            raise InterruptedError('signal received during preflight')

    for label, cmd in (('git HEAD', ['git', 'rev-parse', 'HEAD']),
                       ('git status --short', ['git', 'status', '--short']),
                       ('git diff --stat', ['git', 'diff', '--stat'])):
        value = subprocess.check_output(cmd, cwd=str(ROOT), universal_newlines=True).strip()
        record(label, ('\n```\n' + value + '\n```') if '\n' in value else (value or '(clean)'))
    import torch
    record('Python', sys.version.replace('\n', ' '))
    record('torch / CUDA build', '{} / {}'.format(torch.__version__, torch.version.cuda))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('CUDA must expose exactly one available GPU')
    props = torch.cuda.get_device_properties(0)
    record('GPU', '{}; physical selection {}; logical 0; total {} bytes'.format(
        props.name, os.environ['CUDA_VISIBLE_DEVICES'], props.total_memory))
    try:
        free, total = torch.cuda.mem_get_info(0)
        record('GPU free / total at preflight', '{} / {} bytes'.format(free, total))
    except (AttributeError, RuntimeError) as exc:
        record('GPU free memory', 'unavailable: {}'.format(exc))
    if not desktop_4090(props.name):
        raise RuntimeError('GPU must be a desktop GeForce RTX 4090 (not 4090D or laptop)')
    for module in ('cv2', 'PIL', 'tensorboardX', 'tqdm'):
        importlib.import_module(module)
        record('dependency ' + module, 'available')
    import numpy as np
    if getattr(np, 'int', None) is None:
        raise RuntimeError('NumPy np.int is unavailable; legacy training requires a compatible NumPy')
    record('NumPy / np.int', '{} / available'.format(np.__version__))
    x = torch.ones(1, device='cuda')
    if (x + 1).item() != 2:
        raise RuntimeError('CUDA basic kernel check failed')
    del x
    record('CUDA basic kernel', 'passed')
    if preset in ('fadc', 'combo-fadc'):
        from torchvision.ops import deform_conv2d
        inp = torch.ones((1, 1, 3, 3), device='cuda')
        offset = torch.zeros((1, 18, 3, 3), device='cuda')
        weight = torch.ones((1, 1, 3, 3), device='cuda')
        out = deform_conv2d(inp, offset, weight, padding=(1, 1))
        torch.cuda.synchronize()
        if out.shape != inp.shape or not bool(torch.isfinite(out).all().item()):
            raise RuntimeError('CUDA deform_conv2d result invalid')
        del inp, offset, weight, out
        record('CUDA deform_conv2d', 'passed')
    for field in ('TRAIN_SET', 'TEST_SET'):
        path = Path(cfg.DATASET.ROOT) / getattr(cfg.DATASET, field)
        record(field + ' list', '{}; SHA-256 {}'.format(path, sha256(path)) if path.is_file() else '{}; missing'.format(path))
    weight_path = Path(cfg.MODEL.PRETRAINED) if cfg.MODEL.PRETRAINED else None
    profile = 'imagenet-l' if cfg.DATASET.DATASET == 'cityscapes' else 'cityscapes-l'
    record('weight profile', profile + '; file compatibility does not establish provenance or training split')
    record('weight', '{}; SHA-256 {}'.format(weight_path, sha256(weight_path)) if weight_path and weight_path.is_file() else '{}; missing'.format(weight_path or '(not supplied)'))
    runner.require_training_inputs(cfg)
    record('data and checkpoint existence', 'passed legacy canonical lists and referenced inputs')
    from models.pidnet import get_seg_model
    model = get_seg_model(cfg, imgnet_pretrained=('imagenet' in cfg.MODEL.PRETRAINED), load_pretrained=True)
    del model
    record('CPU checkpoint compatibility', 'passed legacy required shared-tensor/profile checks; provenance unverified')


def run_child(cmd, log_path, on_started=lambda: None, received=None):
    if received is None:
        received = []
    previous = {}
    def on_signal(signum, frame):
        received.append(signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, on_signal)
    start = time.monotonic()
    child = None
    sent = 0
    signal_time = None
    try:
        with log_path.open('w') as log:
            if received:
                return None, 128 + received[0], time.monotonic() - start, received
            child = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            on_started()
            while True:
                while sent < len(received):
                    try:
                        os.killpg(child.pid, received[sent])
                    except ProcessLookupError:
                        pass
                    sent += 1
                    if signal_time is None:
                        signal_time = time.monotonic()
                if signal_time is not None and time.monotonic() - signal_time > 10:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    raw = child.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            code = 128 + received[0] if received else (128 - raw if raw < 0 else raw)
            return raw, code, time.monotonic() - start, received
    finally:
        try:
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                # The leader may have exited while descendants remain alive.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if child.poll() is None:
                    child.wait(timeout=10)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def render(kind, run, facts, cmd, observations='', artifacts=''):
    template = (TEMPLATES / (kind + '.md')).read_text(encoding='utf-8')
    return template.format(date=datetime.date.today().isoformat(), experiment=run.name,
                           metadata='\n'.join(facts), command=cmd,
                           observations=observations, artifacts=artifacts)


def run_one(args, preset, run):
    run.mkdir()  # exclusive; never adopt an existing run
    facts = ['- Dataset / preset: {} / {}'.format(args.dataset, preset),
             '- Python: {}; executable: {}'.format(sys.version.replace('\n', ' '), sys.executable),
             '- GPU physical index (PCI bus order): {}; logical GPU 0'.format(args.gpu),
             '- Batch / workers: {} / {}'.format(args.batch_size, args.workers if args.workers is not None else 'configuration default'),
             '- Seed: 304; FP32 (AMP disabled); no resume or automatic batch/LR changes',
             '- Validation: val.lst, single scale, no flip; test split unused',
             '- Configuration, Git, device, data and checkpoint: pending preflight; see notes.md and report.md']
    # This plan is present before importing the old runner or performing preflight.
    planned = display_command(command(run), args.gpu)
    (run / 'plan.md').write_text(render('plan', run, facts, planned), encoding='utf-8')
    actual = '# Not started; planned command: ' + planned
    obs = 'Preflight not completed.'
    started = time.monotonic()
    code = 1
    parent_signals = []
    old_handlers = {}
    def on_parent_signal(signum, frame):
        parent_signals.append(signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, on_parent_signal)
    launched = []
    cmd = None
    def write_notes(status, shown_command):
        (run / 'notes.md').write_text(
            '# PIDNet-L run notes\n\n' + '\n'.join(facts) + '\n\n' + status +
            '\n\n```bash\n' + shown_command + '\n```\n', encoding='utf-8')
    raw = None
    try:
        runner = load_runner(args.gpu)
        cfg = runner.training_config(old_args(args, preset), run)
        cmd = command(run)
        planned = display_command(cmd, args.gpu)
        facts.append('- Schedule: {} epochs; LR {}; crop {}; classes {}'.format(
            cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR, list(cfg.TRAIN.IMAGE_SIZE), cfg.DATASET.NUM_CLASSES))
        preflight(cfg, runner, facts, preset, lambda: bool(parent_signals))
        if parent_signals:
            raise InterruptedError('signal received during preflight')
        path = runner.write_config(run, cfg)
        facts.append('- Resolved config: {}; SHA-256 {}'.format(path, sha256(path)))
        log_path = run / 'logs' / 'train.log'
        print('Training log: {}'.format(log_path), flush=True)
        write_notes('Preflight passed; command planned, process not yet started.', '# Planned: ' + planned)
        if parent_signals:
            raise InterruptedError('signal received before process creation')
        def on_started():
            launched.append(True)
            write_notes('Training process started; final status pending.', planned)
        raw, code, elapsed, received = run_child(cmd, log_path, on_started, parent_signals)
        if launched:
            actual = display_command(cmd, args.gpu)
        if raw is None:
            actual = '# Not started; signal received before process creation; planned command: ' + planned
        obs = 'Raw child return code: {}; shell return code: {}; elapsed: {:.3f} s; signals received: {}'.format(raw, code, elapsed, received)
    except Exception as exc:
        obs = 'Preflight or launch failed: {}: {}; raw child return code: unavailable; shell return code: {}; elapsed: {:.3f} s'.format(type(exc).__name__, exc, 128 + parent_signals[0] if parent_signals else 1, time.monotonic() - started)
        print(obs, file=sys.stderr, flush=True)
    finally:
        if parent_signals:
            code = 128 + parent_signals[0]
            obs += '; final shell return code: {}; parent signals: {}'.format(code, parent_signals)
        if launched and cmd is not None:
            actual = display_command(cmd, args.gpu)
        try:
            def record_error(path, exc):
                print('Failed to write {}: {}: {}; raw child return code: {}; shell return code: {}'.format(
                    path, type(exc).__name__, exc, raw, 128 + parent_signals[0] if parent_signals else code),
                    file=sys.stderr, flush=True)

            record_failed = False
            try:
                write_notes(obs, actual)
            except Exception as exc:
                record_failed = True
                record_error(run / 'notes.md', exc)
            try:
                paths = ('plan.md', 'notes.md', 'configs/resolved.yaml', 'configs/resolved.sha256', 'logs/train.log', 'output',
                         'output/{}/resolved/checkpoint.pth.tar'.format(args.dataset),
                         'output/{}/resolved/best.pt'.format(args.dataset))
                artifacts = '\n'.join('- {}: {}'.format(run / p, 'exists' if (run / p).exists() else 'not generated') for p in paths)
                (run / 'report.md').write_text(render('report', run, facts, actual, obs, artifacts), encoding='utf-8')
            except Exception as exc:
                record_failed = True
                record_error(run / 'report.md', exc)
            if record_failed and not code:
                code = 1
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        if parent_signals:
            code = 128 + parent_signals[0]
    return code


def main(argv=None):
    args = parse_args(argv)
    selected = targets(args)
    if args.dry_run:
        runner = load_runner(args.gpu)
        print('CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={}; Python {}'.format(args.gpu, sys.executable))
        if args.dataset == 'camvid' and not args.pretrained:
            print('CamVid actual training requires explicit trusted 19-class Cityscapes-L --pretrained')
        for preset, run in selected:
            cfg = runner.training_config(old_args(args, preset), run)
            print('\nPreset: {}\n{}\n{}'.format(preset, display_command(command(run), args.gpu), cfg.dump()))
        return 0
    for preset, run in selected:
        code = run_one(args, preset, run)
        if code:
            return code
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        sys.exit('Error: {}'.format(exc))
