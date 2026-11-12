#!/usr/bin/env python3
"""Controlled launcher checks; no training, evaluations, or external temp paths."""
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / 'test/20261112-pidnet-l-4090-launcher'
spec = importlib.util.spec_from_file_location('launcher', str(ROOT / 'tools/train_pidnet_l_4090.py'))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
runner = launch.load_runner(0)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(DEV / 'scripts'), prefix='check-')
        self.test_root = Path(self.tmp.name)
        self.patch = mock.patch.object(launch, 'TEST_ROOT', self.test_root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def args(self, *args):
        return launch.parse_args(['--run', '20261112-check'] + list(args))

    def cfg(self, dataset='cityscapes', preset='baseline'):
        args = self.args('--dataset', dataset)
        return runner.training_config(launch.old_args(args, preset), self.test_root / '20261112-check')

    def test_help_without_third_party(self):
        cmd = [sys.executable, '-S', '-B', str(ROOT / 'tools/train_pidnet_l_4090.py'), '--help']
        result = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'--i-suite', result.stdout)

    def test_dry_run_no_writes_and_cache(self):
        def snapshot():
            paths = list(self.test_root.rglob('*')) + list(ROOT.rglob('__pycache__'))
            return sorted((str(p), p.stat().st_mtime_ns) for p in paths)
        before = snapshot()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(['--run', '20261112-preview', '--dry-run', '--dataset', 'camvid']), 0)
        self.assertIn('requires explicit trusted', output.getvalue())
        self.assertEqual(before, snapshot())

    def test_config_presets_and_protocol(self):
        for ds, classes, crop, epochs, lr in [('cityscapes', 19, [1024, 1024], 484, .01),
                                               ('camvid', 11, [960, 720], 121, .005)]:
            for preset in launch.PRESETS:
                cfg = self.cfg(ds, preset)
                self.assertEqual((cfg.MODEL.NAME, cfg.DATASET.NUM_CLASSES), ('pidnet_large', classes))
                self.assertEqual((list(cfg.TRAIN.IMAGE_SIZE), cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR), (crop, epochs, lr))
                self.assertEqual((cfg.GPUS, cfg.TRAIN.BATCH_SIZE_PER_GPU), ((0,), 2))
                self.assertEqual((cfg.DATASET.TRAIN_SET, cfg.DATASET.TEST_SET),
                                 ('list/{}/train.lst'.format(ds), 'list/{}/val.lst'.format(ds)))
                self.assertFalse(cfg.TEST.FLIP_TEST or cfg.TEST.MULTI_SCALE or cfg.TRAIN.RESUME or cfg.AUTO_RESUME)
                if preset in ('pappm', 'pmpm', 'denseaspp', 'dappm'):
                    self.assertEqual(cfg.ABLATION.CONTEXT, preset)
                if preset in ('fadc', 'combo-fadc'):
                    self.assertEqual(cfg.ABLATION.I_ATROUS.MODE, 'fadc')
                if preset in ('fixed1', 'fixed2', 'combo-fixed'):
                    self.assertEqual(cfg.ABLATION.I_ATROUS.MODE, 'fixed')
                    self.assertEqual(cfg.ABLATION.I_ATROUS.RATE, 1 if preset == 'fixed1' else 2)

    def test_mask_before_old_import(self):
        original = importlib.util.spec_from_file_location
        def intercept(*args, **kwargs):
            self.assertEqual(os.environ['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')
            return original(*args, **kwargs)
        with mock.patch.object(importlib.util, 'spec_from_file_location', side_effect=intercept):
            launch.load_runner(7)

    def test_gpu_model_and_no_cuda(self):
        self.assertTrue(launch.desktop_4090('NVIDIA GeForce RTX 4090'))
        for name in ('NVIDIA GeForce RTX 4090 D', 'NVIDIA GeForce RTX 4090D', 'NVIDIA GeForce RTX 4090 Laptop GPU', 'NVIDIA GeForce GTX 1650'):
            self.assertFalse(launch.desktop_4090(name))
        with mock.patch('subprocess.check_output', return_value='git'), mock.patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                launch.preflight(self.cfg(), runner, [], 'baseline')

    def test_mocked_gpu_preflight_and_reject_1650(self):
        import torch
        import models.pidnet
        props = mock.Mock()
        props.name, props.total_memory = 'NVIDIA GeForce RTX 4090', 24 * 1024 ** 3
        weight = self.test_root / 'weight.pt'
        weight.write_bytes(b'fake weight')
        for split in ('train', 'val'):
            path = self.test_root / 'list/cityscapes' / (split + '.lst')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fake row')
        args = self.args('--pretrained', str(weight), '--data-root', str(self.test_root))
        cfg = runner.training_config(launch.old_args(args, 'baseline'), self.test_root / args.run)
        tensor = mock.MagicMock()
        (tensor + 1).item.return_value = 2
        facts = []
        with mock.patch('subprocess.check_output', return_value='git'), \
             mock.patch('torch.cuda.is_available', return_value=True), \
             mock.patch('torch.cuda.device_count', return_value=1), \
             mock.patch('torch.cuda.get_device_properties', return_value=props), \
             mock.patch('torch.cuda.mem_get_info', return_value=(1, 2)), \
             mock.patch('torch.ones', return_value=tensor), \
             mock.patch.object(runner, 'require_training_inputs') as inputs, \
             mock.patch.object(models.pidnet, 'get_seg_model') as model:
            launch.preflight(cfg, runner, facts, 'baseline')
            inputs.assert_called_once_with(cfg)
            model.assert_called_once()
            self.assertIn('CPU checkpoint compatibility', '\n'.join(facts))
            self.assertIn('SHA-256', '\n'.join(facts))
            props.name = 'NVIDIA GeForce GTX 1650'
            with self.assertRaisesRegex(RuntimeError, 'desktop GeForce RTX 4090'):
                launch.preflight(cfg, runner, [], 'baseline')

    def test_numpy_compatibility(self):
        import numpy as np
        import torch
        props = mock.Mock(name='props')
        props.name, props.total_memory = 'NVIDIA GeForce RTX 4090', 24000
        with mock.patch('subprocess.check_output', return_value='git'), \
             mock.patch('torch.cuda.is_available', return_value=True), \
             mock.patch('torch.cuda.device_count', return_value=1), \
             mock.patch('torch.cuda.get_device_properties', return_value=props), \
             mock.patch('torch.cuda.mem_get_info', return_value=(1, 2)), \
             mock.patch.object(launch.importlib, 'import_module'), \
             mock.patch.object(np, 'int', None):
            with self.assertRaisesRegex(RuntimeError, 'np.int'):
                launch.preflight(self.cfg(), runner, [], 'baseline')
        self.assertIsNotNone(getattr(np, 'int', None))

    def test_missing_data_weight_and_mismatch(self):
        import torch
        cfg = self.cfg('camvid')
        with self.assertRaisesRegex(ValueError, '--pretrained'):
            runner.require_training_inputs(cfg)
        cfg = self.cfg()
        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint'):
            runner.require_training_inputs(cfg)
        weight = self.test_root / 'fake.pt'
        torch.save({'state_dict': {}}, str(weight))
        args = self.args('--pretrained', str(weight), '--data-root', str(self.test_root))
        cfg = runner.training_config(launch.old_args(args, 'baseline'), self.test_root / args.run)
        with mock.patch('torch.cuda.device_count', return_value=1):
            with self.assertRaises(FileNotFoundError):
                runner.require_training_inputs(cfg)
        from models.pidnet import get_seg_model
        with self.assertRaisesRegex(RuntimeError, 'required PIDNet-L tensors'):
            get_seg_model(cfg, imgnet_pretrained=True, load_pretrained=True)

    def test_existing_and_symlinks(self):
        args = self.args()
        path = self.test_root / args.run
        path.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            launch.targets(args)
        path.rmdir()
        path.symlink_to(self.test_root / 'missing')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            launch.targets(args)
        path.unlink()
        linked = self.test_root / 'testlink'
        linked.symlink_to(self.test_root, target_is_directory=True)
        with mock.patch.object(launch, 'TEST_ROOT', linked):
            with self.assertRaisesRegex(ValueError, 'real directory'):
                launch.targets(args)

    def test_saved_config_compatibility(self):
        run = self.test_root / '20261112-check'
        run.mkdir()
        (run / 'plan.md').write_text('signed plan')
        cfg = self.cfg()
        runner.write_config(run, cfg)
        path, saved = runner.saved_config(run)
        self.assertEqual(saved.dump(), cfg.dump())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.with_suffix('.sha256').read_text().strip())

    def test_suite_order_precheck_and_stop(self):
        args = self.args('--i-suite')
        self.assertEqual([p for p, _ in launch.targets(args)], list(launch.I_SUITE))
        occupied = self.test_root / (args.run + '-combo-fadc')
        occupied.mkdir()
        with mock.patch.object(launch, 'run_one') as one:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                launch.main(['--run', args.run, '--i-suite'])
            one.assert_not_called()
        occupied.rmdir()
        seen = []
        def execute(args, preset, run):
            seen.append(preset)
            return 3 if preset == 'fixed1' else 0
        with mock.patch.object(launch, 'run_one', side_effect=execute):
            self.assertEqual(launch.main(['--run', args.run, '--i-suite']), 3)
        self.assertEqual(seen, list(launch.I_SUITE[:5]))
        seen.clear()
        with mock.patch.object(launch, 'run_one', side_effect=lambda args, preset, run: seen.append(preset) or 0):
            self.assertEqual(launch.main(['--run', args.run, '--i-suite']), 0)
        self.assertEqual(seen, list(launch.I_SUITE))

    def test_child_codes_and_signal(self):
        path = self.test_root / 'child.log'
        raw, code, elapsed, received = launch.run_child([sys.executable, '-c', 'import sys; print("hello"); sys.exit(7)'], path)
        self.assertEqual((raw, code, received), (7, 7, []))
        self.assertIn('hello', path.read_text())
        raw, code, elapsed, received = launch.run_child([sys.executable, '-c',
            'import os,signal,time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(10)'], path)
        self.assertEqual((raw, code, received), (-signal.SIGTERM, 128 + signal.SIGTERM, [signal.SIGTERM]))
        self.assertLess(elapsed, 5)

    def test_child_descendants_stop_on_nonzero_and_signal(self):
        log = self.test_root / 'child.log'
        markers = []
        for interrupted in (False, True):
            marker = self.test_root / ('signal' if interrupted else 'failure')
            markers.append(marker)
            descendant = 'import time,pathlib; time.sleep(1); pathlib.Path({!r}).write_text("alive")'.format(str(marker))
            tail = ('os.kill(os.getppid(), signal.SIGTERM); time.sleep(5)' if interrupted else 'sys.exit(7)')
            leader = ('import subprocess,sys,os,signal,time; '
                      'subprocess.Popen([sys.executable,"-c",{!r}]); '.format(descendant) + tail)
            raw, shell, elapsed, signals = launch.run_child([sys.executable, '-c', leader], log)
            self.assertEqual(shell, 143 if interrupted else 7)
        time.sleep(1.3)
        for marker in markers:
            self.assertFalse(marker.exists(), 'descendant survived: {}'.format(marker))

    def test_graceful_child_signal_and_preflight_signal(self):
        path = self.test_root / 'child.log'
        code = ('import os,signal,time,sys; '
                'signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0)); '
                'os.kill(os.getppid(),signal.SIGTERM); time.sleep(10)')
        raw, shell, elapsed, received = launch.run_child([sys.executable, '-c', code], path)
        self.assertEqual((raw, shell, received), (0, 143, [signal.SIGTERM]))
        def preflight(*args):
            os.kill(os.getpid(), signal.SIGINT)
        with mock.patch.object(launch, 'preflight', side_effect=preflight):
            self.assertEqual(launch.run_one(self.args(), 'baseline', self.test_root / '20261112-check'), 130)
        report = (self.test_root / '20261112-check' / 'report.md').read_text()
        self.assertIn('Not started', report)
        self.assertIn('shell return code: 130', report)
        self.assertIn('parent signals: [2]', report)

    def test_notes_exist_before_launch(self):
        run = self.test_root / '20261112-check'
        def fake_preflight(cfg, old, facts, preset, interrupted):
            facts.append('- simulated device / lists / checkpoint: checked')
        def fake_child(cmd, log_path, on_started, received):
            notes = (run / 'notes.md').read_text()
            self.assertIn('Preflight passed', notes)
            self.assertIn('simulated device / lists / checkpoint: checked', notes)
            self.assertIn('Resolved config:', notes)
            self.assertIn('process not yet started', notes)
            on_started()
            log_path.write_text('fake child, no training')
            return 7, 7, 0.1, []
        with mock.patch.object(launch, 'preflight', side_effect=fake_preflight), \
             mock.patch.object(launch, 'run_child', side_effect=fake_child):
            self.assertEqual(launch.run_one(self.args(), 'baseline', run), 7)
        self.assertIn('shell return code: 7', (run / 'notes.md').read_text())
        self.assertNotIn('Not started', (run / 'report.md').read_text())
        self.assertIn('pending preflight', (run / 'plan.md').read_text())

    def test_preflight_failure_writes_plan_report_no_command(self):
        with mock.patch.object(launch, 'preflight', side_effect=RuntimeError('test failure')):
            self.assertEqual(launch.run_one(self.args(), 'baseline', self.test_root / '20261112-check'), 1)
        run = self.test_root / '20261112-check'
        self.assertTrue((run / 'plan.md').stat().st_size)
        report = (run / 'report.md').read_text()
        self.assertIn('Not started', report)
        self.assertIn('test failure', report)
        self.assertIn('not generated', report)
        notes = (run / 'notes.md').read_text()
        self.assertIn('test failure', notes)
        self.assertIn('Not started', notes)

    def test_final_record_failures_preserve_child_status_and_try_both(self):
        for failed, exit_code, expected in (('report.md', 7, 7), ('notes.md', 7, 7),
                                             ('report.md', 0, 1), ('report.md', -15, 143)):
            run = self.test_root / ('20261112-{}-{}'.format(failed.split('.')[0], exit_code))
            path = run / failed
            # The notes file exists before launch; replace it only in the child.
            termination = ('os.kill(os.getpid(), signal.SIGTERM)' if exit_code < 0
                           else 'sys.exit({})'.format(exit_code))
            code = ('import os,pathlib,signal,sys; p=pathlib.Path({!r}); '
                    'p.unlink() if p.exists() else None; p.mkdir(); '.format(str(path)) + termination)
            stderr = io.StringIO()
            with mock.patch.object(launch, 'preflight'), \
                 mock.patch.object(launch, 'command', return_value=[sys.executable, '-c', code]), \
                 contextlib.redirect_stderr(stderr):
                self.assertEqual(launch.run_one(self.args(), 'baseline', run), expected)
            self.assertIn(str(path), stderr.getvalue())
            self.assertIn('raw child return code: {}'.format(exit_code), stderr.getvalue())
            shell_code = 128 - exit_code if exit_code < 0 else exit_code
            self.assertIn('shell return code: {}'.format(shell_code), stderr.getvalue())
            other = run / ('notes.md' if failed == 'report.md' else 'report.md')
            self.assertTrue(other.is_file())
            self.assertIn('shell return code: {}'.format(shell_code), other.read_text())

    def test_suite_stops_when_successful_child_cannot_write_report(self):
        def child_command(run):
            code = 'import pathlib; pathlib.Path({!r}).mkdir()'.format(str(run / 'report.md'))
            return [sys.executable, '-c', code]
        with mock.patch.object(launch, 'preflight'), \
             mock.patch.object(launch, 'command', side_effect=child_command), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.main(['--run', '20261112-suite', '--i-suite']), 1)
        self.assertTrue((self.test_root / '20261112-suite-baseline').exists())
        self.assertFalse((self.test_root / '20261112-suite-pappm').exists())

    def test_signal_during_final_record_write(self):
        original = Path.write_text
        def write(path, *args, **kwargs):
            if path.name == 'report.md':
                os.kill(os.getpid(), signal.SIGTERM)
            return original(path, *args, **kwargs)
        run = self.test_root / '20261112-signal-write'
        with mock.patch.object(launch, 'preflight'), \
             mock.patch.object(launch, 'command', return_value=[sys.executable, '-c', 'pass']), \
             mock.patch.object(Path, 'write_text', write):
            self.assertEqual(launch.run_one(self.args(), 'baseline', run), 143)
        self.assertTrue((run / 'report.md').is_file())

    def test_started_callback_error_cleans_descendants(self):
        ready, marker = self.test_root / 'ready', self.test_root / 'survived'
        descendant = ('import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                      'pathlib.Path({!r}).touch(); time.sleep(1); pathlib.Path({!r}).touch()'.format(str(ready), str(marker)))
        leader = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{!r}]); time.sleep(20)'.format(descendant)
        def fail_after_ready():
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            raise OSError('simulated notes error')
        started = time.monotonic()
        with self.assertRaisesRegex(OSError, 'simulated notes error'):
            launch.run_child([sys.executable, '-c', leader], self.test_root / 'child.log', fail_after_ready)
        self.assertLess(time.monotonic() - started, 5)
        time.sleep(1.2)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
