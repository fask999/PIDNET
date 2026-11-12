import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from configs import config
from datasets.base_dataset import BaseDataset
from datasets.camvid import CamVid
from datasets.cityscapes import Cityscapes
from utils.criterion import BondaryLoss, CrossEntropy, OhemCrossEntropy
from utils.utils import FullModel


def baseline_object(path, name):
    base = '4c158cf24ce432f0a8cb43364fae38d93cee0dc3'
    source = subprocess.check_output(['git', 'show', base + ':' + path], cwd=ROOT)
    namespace = {}
    exec(compile(source, path + '@' + base, 'exec'), namespace)
    return namespace[name]


def sample(dataset, label, *args, **kwargs):
    image = np.zeros(label.shape + (3,), dtype=np.uint8)
    return dataset.gen_sample(image, label, *args, **kwargs)


class BoundaryTargets(unittest.TestCase):
    def test_legacy_matches_baseline(self):
        original = baseline_object('datasets/base_dataset.py', 'BaseDataset')
        labels = np.random.RandomState(304).choice([0, 1, 2, 255], (16, 21)).astype(np.uint8)
        for edge_pad, city in [(True, True), (False, False)]:
            for base_size, crop_size, x in [(19, (17, 19), 0), (27, (16, 23), 2)]:
                with self.subTest(edge_pad=edge_pad, base_size=base_size):
                    samples = []
                    for cls in (original, BaseDataset):
                        dataset = cls(base_size=base_size, crop_size=crop_size)
                        with mock.patch('random.randint', side_effect=[5, x, 1 if x else 0]), \
                                mock.patch('numpy.random.choice', return_value=0):
                            samples.append(sample(dataset, labels, edge_pad=edge_pad, city=city))
                    for expected, actual in zip(*samples):
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.dtype, expected.dtype)
        # Legacy edge padding used to make the wrong shape below 12 pixels.
        _, label, edge = sample(BaseDataset(), labels[:5], multi_scale=False, is_flip=False)
        self.assertEqual(edge.shape, label.shape)
        self.assertFalse(edge.any())

    def test_valid_targets_after_scale_crop_pad_flip(self):
        labels = np.tile([0, 1, 1, 255, 2], (3, 1)).astype(np.uint8)
        for ignore in (255, -1):
            for mask in (False, True):
                with self.subTest(ignore=ignore, mask=mask):
                    dataset = BaseDataset(ignore_label=ignore, base_size=10, crop_size=(7, 6),
                                          bd_target='valid', bd_mask_void=mask)
                    with mock.patch('random.randint', side_effect=[5, 1, 0]), \
                            mock.patch('numpy.random.choice', return_value=0):
                        image, label, edge = sample(dataset, labels, edge_size=1)
                    expected_label = np.tile([ignore, 1, 1, 1, 1, 0], (7, 1))
                    expected_label[-1] = ignore
                    expected_edge = np.tile([-1 if mask else 0, 0, 0, 0, 1, 1], (7, 1))
                    expected_edge[-1] = -1 if mask else 0
                    self.assertEqual(image.shape, (3, 7, 6))
                    np.testing.assert_array_equal(label, expected_label)
                    np.testing.assert_array_equal(edge, expected_edge)

    def test_neighbor_rule_width_and_thin_void_barrier(self):
        dataset = BaseDataset(bd_target='valid')
        labels = np.tile([0, 0, 1, 1, 255, 2, 2, 2], (3, 1)).astype(np.uint8)
        for width, expected in (
                (1, [0, 1, 1, 0, 0, 0, 0, 0]),
                (3, [1, 1, 1, 1, 0, 0, 0, 0]),
                (7, [1, 1, 1, 1, 0, 0, 0, 0])):
            _, _, edge = sample(dataset, labels, False, False, edge_size=width)
            np.testing.assert_array_equal(edge, np.tile(expected, (3, 1)))
        _, _, transposed = sample(dataset, labels.T, False, False, edge_size=1)
        np.testing.assert_array_equal(transposed.T, np.tile([0, 1, 1, 0, 0, 0, 0, 0], (3, 1)))
        _, _, diagonal = sample(dataset, np.array([[0, 255], [255, 1]]), False, False,
                                edge_size=1)
        self.assertFalse(diagonal.any())
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            sample(dataset, labels, False, False, edge_size=4)

    def test_void_mask_is_independent_of_target_mode(self):
        labels = np.zeros((15, 21), dtype=np.uint8)
        labels[6:9, 8:11] = 255
        for target, width in [('legacy', 4), ('valid', 3)]:
            for radius in (0, 1):
                with self.subTest(target=target, radius=radius):
                    ds = BaseDataset(bd_target=target, bd_mask_void=True, bd_void_radius=radius)
                    _, _, edges = sample(ds, labels, False, False, edge_size=width)
                    ignored = np.zeros_like(labels, dtype=bool)
                    ignored[6-radius:9+radius, 8-radius:11+radius] = True
                    np.testing.assert_array_equal(edges == -1, ignored)
                    if target == 'valid':
                        self.assertFalse(edges[~ignored].any())
                    elif radius == 0:
                        self.assertTrue((edges[~ignored] == 1).any())

    def test_both_dataset_adapters_keep_five_outputs_and_signed_void(self):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        indexed = np.tile([7, 8, 8, 0, 255], (3, 1)).astype(np.uint8)
        rgb_label = np.tile(np.array([[0, 128, 192], [128, 0, 0], [128, 0, 0],
                                     [0, 0, 0], [0, 0, 0]], dtype=np.uint8), (3, 1, 1))
        for dataset_type in (Cityscapes, CamVid):
            for ignore in (255, -1):
                with self.subTest(dataset=dataset_type.__name__, ignore=ignore), \
                        mock.patch('builtins.open', mock.mock_open(read_data='image.png label.png\n')), \
                        mock.patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
                    dataset = dataset_type('synthetic/', 'val.lst', multi_scale=False, flip=False,
                                           ignore_label=ignore, bd_dilate_size=1,
                                           bd_target='valid', bd_mask_void=True)
                    if dataset_type is Cityscapes:
                        reader = mock.patch('cv2.imread', side_effect=[image, indexed.copy()])
                    else:
                        reader = mock.patch('PIL.Image.open', side_effect=[
                            Image.fromarray(image), Image.fromarray(rgb_label)])
                    with reader:
                        transformed, labels, edges, size, name = dataset[0]
                    self.assertEqual(transformed.shape, (3, 3, 5))
                    np.testing.assert_array_equal(labels, np.tile([0, 1, 1, ignore, ignore], (3, 1)))
                    np.testing.assert_array_equal(edges, np.tile([1, 1, 0, -1, -1], (3, 1)))
                    np.testing.assert_array_equal(size, image.shape)
                    self.assertEqual(name, 'label')


class BoundaryLosses(unittest.TestCase):
    def test_default_bce_exact_value_and_gradient(self):
        original = baseline_object('utils/criterion.py', 'BondaryLoss')()
        current = BondaryLoss()
        for labels in ([0, 1, 0, 0, 1, 0], [0] * 6, [1] * 6):
            logits = torch.linspace(-1.0, 1.0, 6).reshape(1, 1, 2, 3).requires_grad_()
            target = torch.tensor(labels, dtype=torch.float32).reshape(1, 2, 3)
            a, b = original(logits, target), current(logits, target)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(torch.autograd.grad(a, logits)[0],
                                        torch.autograd.grad(b, logits)[0]))

    def test_masked_reduction_and_single_class_gradients(self):
        logits = torch.tensor([0.1, 20, -0.3, 0.7, -20]).reshape(1, 1, 1, 5).requires_grad_()
        target = torch.tensor([[[1., -1., 0., 0., -1.]]])
        weights = torch.tensor([2/3, 1/3, 1/3])
        loss_fn = BondaryLoss(coeff_bce=1, mask_void=True)
        expected = F.binary_cross_entropy_with_logits(
            logits.flatten()[[0, 2, 3]], target.flatten()[[0, 2, 3]], weights)
        loss = loss_fn(logits, target)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad.flatten()[[1, 4]], torch.zeros(2)))
        for label in (0.0, 1.0, -1.0):
            logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
            target = torch.full((1, 2, 3), label)
            loss = loss_fn(logits, target)
            torch.testing.assert_close(loss, torch.tensor(0 if label < 0 else np.log(2),
                                                          dtype=torch.float32))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(bool(logits.grad.any()), label >= 0)

    def test_dice_without_bce_with_and_without_mask(self):
        for mask in (False, True):
            logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
            target = torch.tensor([[[1., 0., -1. if mask else 0.]]])
            loss = BondaryLoss(coeff_bce=0, dice_weight=0.75, mask_void=mask)(logits, target)
            torch.testing.assert_close(loss, torch.tensor(0.75 * (1 - 2 / (3 if mask else 3.5))))
            loss.backward()
            self.assertTrue(logits.grad.flatten()[:2].ne(0).all())
            self.assertEqual(logits.grad.flatten()[2].item() == 0, mask)
        logits = torch.zeros(1, 1, 256, 512, requires_grad=True)
        target = torch.ones(1, 256, 512, dtype=torch.float16)
        count = target.numel()
        loss = BondaryLoss(coeff_bce=0, dice_weight=1)(logits, target)
        torch.testing.assert_close(loss, torch.tensor(1 - (count + 1) / (1.5 * count + 1)))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_semantic_losses_keep_nonempty_formula_and_guard_empty(self):
        score = torch.tensor([0.5, -2.0, 0.4, -0.5, 2.0, -0.4]).reshape(1, 2, 1, 3)
        score.requires_grad_()
        target = torch.tensor([[[0, 0, 255]]])
        for cls in (CrossEntropy, OhemCrossEntropy):
            original = baseline_object('utils/criterion.py', cls.__name__)(ignore_label=255)
            current = cls(ignore_label=255)
            torch.testing.assert_close(current([score, score], target),
                                       original([score, score], target), atol=0, rtol=0)
            if cls is CrossEntropy:
                torch.testing.assert_close(current(score, target),
                                           config.LOSS.SB_WEIGHTS * F.cross_entropy(
                                               score, target, ignore_index=255))
            empty = current(score, torch.full_like(target, 255))
            self.assertEqual(empty.item(), 0)
            self.assertTrue(torch.autograd.grad(empty, score)[0].eq(0).all())
        easy = torch.tensor([[[[8., 8.]], [[0., 0.]]]], requires_grad=True)
        empty_mining = OhemCrossEntropy(thres=0.9)(easy, torch.zeros(1, 1, 2, dtype=torch.long))
        self.assertEqual(empty_mining.item(), 0)
        self.assertTrue(torch.autograd.grad(empty_mining, easy)[0].eq(0).all())

    def test_full_model_ce_and_ohem_mask_bas_and_backpropagate(self):
        class Toy(torch.nn.Module):
            def __init__(self, gate_open):
                super().__init__()
                self.aux = torch.nn.Parameter(torch.zeros(1, 2, 2, 4))
                self.main = torch.nn.Parameter(torch.linspace(-0.4, 0.5, 16).reshape(1, 2, 2, 4))
                self.boundary = torch.nn.Parameter(torch.full((1, 1, 2, 4), 2.0 if gate_open else -2.0))

            def forward(self, inputs):
                return [self.aux, self.main, self.boundary]

        with mock.patch.dict(config.TRAIN, {'IGNORE_LABEL': 255}):
            for cls in (CrossEntropy, OhemCrossEntropy):
                for case in ('masked', 'closed', 'all_ignore'):
                    with self.subTest(criterion=cls.__name__, case=case):
                        labels = torch.tensor([[[0, 1, 1, 255], [1, 0, 0, 1]]])
                        bd_gt = torch.tensor([[[1., -1., 0., -1.], [-1., 1., 0., 0.]]])
                        if case == 'all_ignore':
                            labels.fill_(255)
                            bd_gt.fill_(-1)
                        net = Toy(case != 'closed')
                        semantic = cls(ignore_label=255)
                        boundary = BondaryLoss(dice_weight=0.25, mask_void=True)
                        loss, outputs, acc, (sem, bd) = FullModel(net, semantic, boundary)(
                            torch.zeros(1), labels, bd_gt)
                        self.assertEqual(len(outputs), 2)
                        self.assertTrue(torch.isfinite(acc))
                        if case == 'masked':
                            bas_labels = torch.tensor([[[0, 255, 1, 255], [255, 0, 0, 1]]])
                            expected = sem + bd + semantic(net.main, bas_labels)
                            self.assertFalse(torch.allclose(expected, sem + bd + semantic(net.main, labels)))
                        else:
                            expected = sem + bd
                        torch.testing.assert_close(loss.squeeze(0), expected)
                        self.assertTrue(torch.isfinite(loss).all())
                        loss.mean().backward()
                        for param in net.parameters():
                            self.assertTrue(torch.isfinite(param.grad).all())
                            self.assertEqual(bool(param.grad.any()), case != 'all_ignore')
                        self.assertTrue(net.boundary.grad[:, 0][bd_gt < 0].eq(0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
