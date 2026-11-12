"""CPU-only model/config/checkpoint checks; does not read experimental data."""

import gc
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections import OrderedDict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, ROOT)

from configs import config
from models.context import DenseASPP, MixedPool, PMPM
from models.model_utils import DAPPM, PAPPM
from models.pidnet import PIDNet, get_pred_model, get_seg_model
from utils.ablation_config import (register_ablation_config, get_model_kwargs,
                                   get_boundary_kwargs, get_boundary_loss_kwargs)
from utils.checkpoint import load_model_checkpoint


def legacy_class():
    source = subprocess.check_output(['git', 'show', 'HEAD:models/pidnet.py'], cwd=ROOT)
    old = types.ModuleType('models._original_pidnet')
    old.__package__ = 'models'
    exec(compile(source, 'HEAD:models/pidnet.py', 'exec'), old.__dict__)
    return old.PIDNet


def variant(name, classes=19):
    if name == 's':
        return dict(m=2, n=3, planes=32, ppm_planes=96, head_planes=128,
                    num_classes=classes)
    if name == 'm':
        return dict(m=2, n=3, planes=64, ppm_planes=96, head_planes=128,
                    num_classes=classes)
    return dict(m=3, n=4, planes=64, ppm_planes=112, head_planes=256,
                num_classes=classes)


class ContextIntegration(unittest.TestCase):
    def test_original_defaults(self):
        original = legacy_class()
        x = torch.randn(1, 3, 64, 128)
        for name in ('s', 'm', 'l'):
            with self.subTest(model=name):
                torch.manual_seed(304)
                old = original(**variant(name)).eval()
                torch.manual_seed(304)
                new = PIDNet(**variant(name)).eval()
                self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                for key, value in old.state_dict().items():
                    self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
                with torch.no_grad():
                    old_outputs, new_outputs = old(x), new(x)
                for a, b in zip(old_outputs, new_outputs):
                    self.assertTrue(torch.equal(a, b))
                explicit = 'dappm' if name == 'l' else 'pappm'
                torch.manual_seed(304)
                chosen = PIDNet(context=explicit, **variant(name)).eval()
                for key, value in old.state_dict().items():
                    self.assertTrue(torch.equal(value, chosen.state_dict()[key]), key)
                with torch.no_grad():
                    chosen_outputs = chosen(x)
                for a, b in zip(old_outputs, chosen_outputs):
                    self.assertTrue(torch.equal(a, b))
                del old, new, chosen, old_outputs, new_outputs, chosen_outputs
                gc.collect()

    def test_every_context_shape_and_gradient(self):
        constructors = dict(pappm=PAPPM, dappm=DAPPM, pmpm=PMPM, denseaspp=DenseASPP)
        for name in ('s', 'm', 'l'):
            for classes in (11, 19):
                for context in ('auto', 'pappm', 'dappm', 'pmpm', 'denseaspp'):
                    with self.subTest(model=name, classes=classes, context=context):
                        model = PIDNet(context=context, **variant(name, classes)).eval()
                        expected = constructors.get(context, DAPPM if name == 'l' else PAPPM)
                        self.assertIs(type(model.spp), expected)
                        x = torch.randn(1, 3, 64, 128, requires_grad=True)
                        outputs = model(x)
                        self.assertEqual([tuple(y.shape) for y in outputs],
                                         [(1, classes, 8, 16), (1, classes, 8, 16), (1, 1, 8, 16)])
                        sum(y.square().mean() for y in outputs).backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in model.spp.parameters()))
                        del model, x, outputs
                        gc.collect()

    def test_mixed_pool_and_dense_connections(self):
        x = torch.arange(6 * 7 * 11, dtype=torch.float32).reshape(1, 6, 7, 11)
        mixed = MixedPool(5, 2)(x)
        a, b, c = torch.split(x, 2, 1)
        expected = torch.cat([torch.nn.functional.avg_pool2d(a, 5, 2, 2),
                              torch.nn.functional.avg_pool2d(b, (1, 5), 2, (0, 2)),
                              torch.nn.functional.avg_pool2d(c, (5, 1), 2, (2, 0))], 1)
        self.assertTrue(torch.equal(mixed, expected))
        dense = DenseASPP(12, 4, 7, (1, 2, 3, 4)).eval()
        shapes = []
        handles = [layer.register_forward_pre_hook(lambda m, inputs: shapes.append(tuple(inputs[0].shape)))
                   for layer in dense.layers]
        dense(torch.randn(2, 12, 5, 9))
        for handle in handles:
            handle.remove()
        self.assertEqual([shape[1] for shape in shapes], [12, 16, 20, 24])
        self.assertEqual([layer[-1].dilation for layer in dense.layers],
                         [(1, 1), (2, 2), (3, 3), (4, 4)])
        custom = DenseASPP(12, 4, 7, (4, 3, 2, 1))
        self.assertEqual([layer[-1].dilation for layer in custom.layers],
                         [(4, 4), (3, 3), (2, 2), (1, 1)])
        custom.load_state_dict({k: v for k, v in dense.state_dict().items() if k != '_dense_rates'}, strict=False)
        custom.eval()
        rate_input = torch.randn(2, 12, 9, 11)
        with torch.no_grad():
            self.assertFalse(torch.allclose(dense(rate_input), custom(rate_input)))

    def test_fixed_stages_and_lazy_import(self):
        self.assertNotIn('models.fadc', sys.modules)
        for stage in (3, 4, 5):
            with self.subTest(stage=stage):
                torch.manual_seed(304)
                baseline = PIDNet(**variant('s')).eval()
                torch.manual_seed(304)
                fixed = PIDNet(atrous_mode='fixed', atrous_stage=stage,
                               atrous_block=1, atrous_rate=2, **variant('s')).eval()
                signature = '_i_atrous_signature_fixed_s{}_b1_r2'.format(stage)
                self.assertEqual(set(fixed.state_dict()) - set(baseline.state_dict()), {signature})
                for key, tensor in baseline.state_dict().items():
                    self.assertTrue(torch.equal(tensor, fixed.state_dict()[key]), key)
                for other_stage in (3, 4, 5):
                    before = getattr(baseline, 'layer{}'.format(other_stage))[1].conv2
                    after = getattr(fixed, 'layer{}'.format(other_stage))[1].conv2
                    self.assertEqual(before.dilation, (1, 1))
                    self.assertEqual(after.dilation, (2, 2) if stage == other_stage else (1, 1))
                del baseline, fixed
                gc.collect()
        self.assertNotIn('models.fadc', sys.modules)

    def test_fixed_checkpoint_requires_matching_geometry(self):
        common = dict(planes=4, ppm_planes=8, head_planes=8)
        settings = dict(atrous_mode='fixed', atrous_stage=4, atrous_block=1, atrous_rate=4)
        source = PIDNet(**common, **settings).eval()
        raw = source.state_dict()
        wrapped = {'state_dict': {'module.model.' + k: v for k, v in raw.items()}}
        saved_signature = '_i_atrous_signature_fixed_s4_b1_r4'
        self.assertIn(saved_signature, raw)
        rebuilt = PIDNet(**common, **settings).eval()
        self.assertEqual(load_model_checkpoint(rebuilt, wrapped), len(raw))
        x = torch.randn(1, 3, 64, 128)
        with torch.no_grad():
            for a, b in zip(source(x), rebuilt(x)):
                self.assertTrue(torch.equal(a, b))

        for change in (dict(atrous_rate=2), dict(atrous_stage=3),
                       dict(atrous_block=2), dict(atrous_mode='none')):
            with self.subTest(change=change):
                wrong = PIDNet(**common, **{**settings, **change}).eval()
                with self.assertRaisesRegex(RuntimeError, saved_signature):
                    load_model_checkpoint(wrong, wrapped)
                with self.assertRaisesRegex(RuntimeError, saved_signature):
                    wrong.load_state_dict(raw, strict=True)

        partial = PIDNet(**common, **{**settings, 'atrous_rate': 2}).eval()
        self.assertEqual(load_model_checkpoint(partial, wrapped, strict=False), len(raw) - 1)
        self.assertTrue(torch.equal(partial.layer4[1].conv2.weight, raw['layer4.1.conv2.weight']))
        self.assertEqual(partial.layer4[1].conv2.dilation, (2, 2))
        self.assertIn('_i_atrous_signature_fixed_s4_b1_r2', partial.state_dict())
        self.assertNotIn(saved_signature, partial.state_dict())
        with self.assertRaisesRegex(RuntimeError, saved_signature):
            load_model_checkpoint(source, partial.state_dict())
        load_model_checkpoint(partial, partial.state_dict())

    def test_checkpoint_formats_and_wrong_structure(self):
        for context in ('auto', 'pmpm', 'denseaspp'):
            with self.subTest(context=context):
                model = PIDNet(context=context, **variant('s')).eval()
                raw = model.state_dict()
                for prefix in ('', 'model.', 'module.', 'module.model.'):
                    wrapped = {'state_dict': OrderedDict((prefix + k, v) for k, v in raw.items())}
                    wrapped['state_dict']['sem_loss.weight'] = torch.ones(19)
                    self.assertEqual(load_model_checkpoint(model, wrapped), len(raw))
                pred = get_pred_model('pidnet_s', 19, context=context).eval()
                output = os.path.join(ROOT, 'test/20261111-pidnet-ablation-switches/output')
                with tempfile.TemporaryDirectory(prefix='context-test-', dir=output) as folder:
                    path = os.path.join(folder, 'checkpoint.pt')
                    torch.save({'state_dict': {'module.model.' + k: v for k, v in raw.items()}}, path)
                    self.assertEqual(load_model_checkpoint(model, path), len(raw))
                    load_model_checkpoint(pred, path, prediction=True)
                load_model_checkpoint(pred, {'state_dict': {'model.' + k: v for k, v in raw.items()}}, prediction=True)
                x = torch.randn(1, 3, 64, 128)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(x)[1], pred(x)))
                broken = dict(raw)
                broken.pop('final_layer.conv2.weight')
                with self.assertRaisesRegex(RuntimeError, 'missing'):
                    load_model_checkpoint(model, broken)
                broken = dict(raw)
                broken['final_layer.conv2.weight'] = torch.zeros(1)
                with self.assertRaisesRegex(RuntimeError, 'shape_mismatch'):
                    load_model_checkpoint(model, broken)
                del model, pred, raw
                gc.collect()
        pappm = PIDNet(context='pappm', **variant('s'))
        pmpm = PIDNet(context='pmpm', **variant('s'))
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            load_model_checkpoint(pmpm, pappm.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
            load_model_checkpoint(pappm, pmpm.state_dict())
        self.assertGreater(load_model_checkpoint(pmpm, pappm.state_dict(), strict=False), 0)
        dense_a = PIDNet(context='denseaspp', **variant('s'))
        dense_b = PIDNet(context='denseaspp', dense_rates=(4, 3, 2, 1), **variant('s'))
        with self.assertRaisesRegex(RuntimeError, 'configured rates differ'):
            load_model_checkpoint(dense_b, dense_a.state_dict())
        self.assertGreater(load_model_checkpoint(dense_b, dense_a.state_dict(), strict=False), 0)
        self.assertEqual(dense_b.spp._dense_rates.tolist(), [4, 3, 2, 1])

    def test_config_defaults_and_errors(self):
        cfg = config.clone()
        if hasattr(cfg, 'ABLATION'):
            cfg.defrost()
            del cfg.ABLATION
        self.assertEqual(get_model_kwargs(cfg)['context'], 'auto')
        self.assertEqual(get_boundary_kwargs(cfg)['bd_dilate_size'], 4)
        register_ablation_config(cfg)
        cfg.ABLATION.BOUNDARY.TARGET = 'valid'
        self.assertEqual(get_boundary_kwargs(cfg)['bd_dilate_size'], 3)
        cfg.ABLATION.BOUNDARY.WIDTH = 2
        with self.assertRaises(ValueError): get_boundary_kwargs(cfg)
        cfg.ABLATION.BOUNDARY.WIDTH = 0
        cfg.ABLATION.BOUNDARY.DICE_WEIGHT = float('nan')
        with self.assertRaises(ValueError): get_boundary_loss_kwargs(cfg)
        for kwargs in (dict(context='other'), dict(dense_rates=(1, 2, 3)),
                       dict(dense_rates=(1, 2, 0, 4)), dict(atrous_stage=2),
                       dict(atrous_block=0), dict(atrous_stage=5, atrous_block=2),
                       dict(atrous_rate=0), dict(atrous_mode='bad')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PIDNet(**kwargs)

    def test_cli_parses_and_constructs(self):
        # Separate processes preserve each entry point's real global yacs config.
        code = '''import sys
sys.path.insert(0, 'tools')
import {entry} as cli
from models.pidnet import get_seg_model, get_pred_model
from utils.ablation_config import get_model_kwargs, get_boundary_kwargs
sys.argv = ['{entry}', '--cfg', 'configs/camvid/pidnet_small_camvid.yaml'] + {extra} + ['ABLATION.CONTEXT', 'denseaspp', 'ABLATION.DENSE_RATES', '[4,3,2,1]', 'ABLATION.I_ATROUS.MODE', 'fixed', 'ABLATION.I_ATROUS.STAGE', '3', 'ABLATION.I_ATROUS.BLOCK', '1', 'ABLATION.I_ATROUS.RATE', '3', 'ABLATION.BOUNDARY.TARGET', 'valid']
args = cli.parse_args()
cfg = cli.config
model = (get_pred_model(cfg.MODEL.NAME, cfg.DATASET.NUM_CLASSES, **get_model_kwargs(cfg)) if '{entry}' == 'custom' else get_seg_model(cfg, True, load_pretrained=False))
assert model.layer3[1].conv2.dilation == (3, 3)
assert [layer[-1].dilation[0] for layer in model.spp.layers] == [4, 3, 2, 1]
assert model.final_layer.conv2.out_channels == 11
assert get_boundary_kwargs(cfg)['bd_dilate_size'] == 3
assert cfg.TEST.OUTPUT_INDEX == 1
'''
        for entry in ('train', 'eval', 'custom'):
            with self.subTest(entry=entry):
                extra = "['--output-dir', 'test/20261111-pidnet-ablation-switches/output/custom']" if entry == 'custom' else '[]'
                subprocess.check_call([sys.executable, '-B', '-c', code.format(entry=entry, extra=extra)], cwd=ROOT)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
