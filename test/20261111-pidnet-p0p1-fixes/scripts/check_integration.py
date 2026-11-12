"""Reuse the original read-only ablation checks; keep all new CLI fixtures here."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


EXP = Path(__file__).resolve().parents[1]
OLD = EXP.with_name('20261111-pidnet-ablation-switches')
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(OLD / 'tests'))


def verify_ohem_legacy_eligible():
    from configs import config
    from test_boundary import baseline_object
    from utils.criterion import OhemCrossEntropy

    # At least K pixels meet the old threshold; keep every eligible pixel.
    old = baseline_object('utils/criterion.py', 'OhemCrossEntropy')(thres=.8, min_kept=2)
    new = OhemCrossEntropy(thres=.8, min_kept=2)
    x = torch.tensor([[[[-2., -1., 0., 1., 3.]], [[0., 0., 0., 0., 0.]]]], requires_grad=True)
    y = torch.zeros(1, 1, 5, dtype=torch.long)
    assert len(config.LOSS.BALANCE_WEIGHTS) != 1
    old_loss, new_loss = old(x, y), new(x, y)
    assert torch.equal(old_loss, new_loss)
    old_grad = torch.autograd.grad(old_loss, x)[0]
    new_grad = torch.autograd.grad(new_loss, x)[0]
    assert torch.equal(old_grad, new_grad)
    assert int(new_grad[:, 0].count_nonzero()) == 4
    print('OHEM (4 threshold candidates, K=2): bitwise equal to HEAD in loss and gradient', flush=True)


def cpu_checks():
    from verify_ablation import verify_presets

    loader = unittest.defaultTestLoader
    context = 'test_context_integration.ContextIntegration.'
    boundary = 'test_boundary.BoundaryLosses.'
    names = [context + 'test_' + name for name in (
        'original_defaults', 'every_context_shape_and_gradient', 'mixed_pool_and_dense_connections',
        'fixed_stages_and_lazy_import', 'fixed_checkpoint_requires_matching_geometry',
        'config_defaults_and_errors', 'cli_parses_and_constructs',
    )] + ['test_boundary.BoundaryTargets'] + [boundary + 'test_' + name for name in (
        'default_bce_exact_value_and_gradient', 'dice_without_bce_with_and_without_mask',
        'full_model_ce_and_ohem_mask_bas_and_backpropagate',
    )] + ['test_atrous.FixedAtrousTests']
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromNames(names))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)
    verify_ohem_legacy_eligible()
    verify_presets()


def checkpoint_format_checks():
    import test_context_integration as legacy

    # The original test hard-codes its old experiment output for a temporary
    # file. Supply the same tempfile operation with this experiment's directory.
    def new_fixture(*, prefix, dir):
        assert Path(dir).resolve() == OLD / 'output'
        return TemporaryDirectory(prefix=prefix, dir=str(EXP / 'output'))

    name = 'test_context_integration.ContextIntegration.test_checkpoint_formats_and_wrong_structure'
    with patch.object(legacy, 'tempfile', SimpleNamespace(TemporaryDirectory=new_fixture)):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(name))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


def cuda_checks():
    import verify_ablation as legacy

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_atrous.FADCTests'))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)

    # Resolve the existing wrapper's settings without running it. The original
    # CLI smoke uses EXP for fixtures; substitute this experiment only there.
    entries = {pair: legacy.cli_args(*pair) for pair in (
        ('cityscapes', 'combo-fadc'), ('camvid', 'combo-fixed'))}
    legacy.cli_args = lambda dataset, preset: (entries[(dataset, preset)][0], dict(entries[(dataset, preset)][1]))
    legacy.EXP = EXP
    legacy.np.random.seed(legacy.SEED)
    torch.manual_seed(legacy.SEED)
    legacy.verify_cli()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('cpu', 'cuda', 'checkpoint-formats'))
    args = parser.parse_args()
    torch.manual_seed(304)
    torch.set_num_threads(2)
    {'cpu': cpu_checks, 'cuda': cuda_checks, 'checkpoint-formats': checkpoint_format_checks}[args.suite]()
