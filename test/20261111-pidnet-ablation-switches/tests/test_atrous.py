import copy
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import torch
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[3]
FADC_TEST_DEVICE = os.environ.get("FADC_TEST_DEVICE", "cuda")
sys.path.insert(0, str(REPO))

from models.i_atrous import configure_i_atrous
from models.pidnet import PIDNet
from utils.checkpoint import load_model_checkpoint


def small_pidnet(**kwargs):
    return PIDNet(planes=4, ppm_planes=8, head_planes=8, **kwargs)


class FixedAtrousTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(304)

    def test_noop_fixed_rate_one_and_selection(self):
        model = small_pidnet()
        convs = {name: module for name, module in model.named_modules() if name.endswith(".conv2")}
        original_geometry = {name: (conv.dilation, conv.padding) for name, conv in convs.items()}
        original_state = copy.deepcopy(model.state_dict())
        chosen = model.layer4[1].conv2
        inputs = torch.randn(2, chosen.in_channels, 13, 15)
        rng = torch.random.get_rng_state()
        original = chosen(inputs)
        configure_i_atrous(model.layer4, mode="none", block_index=1, rate=2)
        configure_i_atrous(model.layer4, mode="fixed", block_index=1, rate=1)
        self.assertTrue(torch.equal(chosen(inputs), original))
        configure_i_atrous(model.layer4, mode="fixed", block_index=1, rate=2)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(model.state_dict().keys(), original_state.keys())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[key]), key)
        for name, conv in convs.items():
            self.assertIs(dict(model.named_modules())[name], conv)
            if name == "layer4.1.conv2":
                self.assertEqual(conv.dilation, (2, 2))
                self.assertEqual(conv.padding, (2, 2))
            else:
                self.assertEqual((conv.dilation, conv.padding), original_geometry[name], name)
        self.assertEqual(chosen(inputs).shape, original.shape)
        self.assertFalse(torch.equal(chosen(inputs), original))

    def test_fixed_rate_two_moves_an_impulse_by_two(self):
        model = small_pidnet()
        with torch.no_grad():
            conv = model.layer5[1].conv2
            conv.weight.zero_()
            conv.weight[0, 0, 0, 2] = 1.0
            image = torch.zeros(1, conv.in_channels, 7, 9)
            image[0, 0, 3, 4] = 1.0
            expected = torch.zeros(1, conv.out_channels, 7, 9)
            expected[0, 0, 4, 3] = 1.0
            self.assertTrue(torch.equal(conv(image), expected))

            configure_i_atrous(model.layer5, mode="fixed", block_index=1, rate=2)
            expected.zero_()
            expected[0, 0, 5, 2] = 1.0
            self.assertTrue(torch.equal(conv(image), expected))

    def test_rejects_the_downsampling_block(self):
        model = small_pidnet()
        for stage in (model.layer3, model.layer4, model.layer5):
            with self.assertRaisesRegex(ValueError, "downsampling"):
                configure_i_atrous(stage, mode="fixed", block_index=0, rate=2)

    def test_optional_torchvision_import_and_error(self):
        # Fresh Python so another test's successful import cannot hide this dependency.
        result = subprocess.run(
            [sys.executable, "-B", "-c", textwrap.dedent("""
                import sys
                from importlib.abc import MetaPathFinder

                attempts = []

                class MissingTorchvision(MetaPathFinder):
                    def find_spec(self, fullname, path, target=None):
                        if fullname == 'torchvision' or fullname.startswith('torchvision.'):
                            attempts.append(fullname)
                            raise ModuleNotFoundError(fullname)

                sys.meta_path.insert(0, MissingTorchvision())
                from models.i_atrous import configure_i_atrous
                from models.pidnet import PIDNet

                model = PIDNet(planes=4, ppm_planes=8, head_planes=8)
                configure_i_atrous(model.layer4, mode='none', block_index=1, rate=2)
                configure_i_atrous(model.layer3, mode='fixed', block_index=1, rate=2)
                assert not attempts
                original = model.layer4[1].conv2
                try:
                    configure_i_atrous(model.layer4, mode='fadc', block_index=1, rate=2)
                except RuntimeError as error:
                    assert 'compiled deform_conv2d' in str(error) and 'PyTorch' in str(error), error
                    assert isinstance(error.__cause__, ModuleNotFoundError), error.__cause__
                else:
                    raise AssertionError('FADC silently continued without torchvision')
                assert attempts and model.layer4[1].conv2 is original
            """)], cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


@unittest.skipUnless(FADC_TEST_DEVICE == "cpu" or torch.cuda.is_available(),
                     "requires CUDA torchvision ops (or explicit FADC_TEST_DEVICE=cpu)")
class FADCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(304)
        self.device = torch.device(FADC_TEST_DEVICE)

    def test_frequency_gates_select_distinct_bands(self):
        from models.fadc import _FrequencySelection

        selector = _FrequencySelection(1).to(self.device)
        y = torch.arange(16, device=self.device)[:, None]
        x = torch.arange(18, device=self.device)[None, :]
        high = 0.25 * (1 - 2 * ((y + x) % 2))
        middle = 0.125 * torch.cos(2 * math.pi * 3 * y / 16)
        features = (2 + high + middle)[None, None]
        with torch.no_grad():
            self.assertTrue(torch.equal(selector(features), features))
            selector.gates.bias[0] = -math.log(3)
            selector.gates.bias[-1] = math.log(3)
            expected = (3 + 0.5 * high + middle)[None, None]
            torch.testing.assert_close(selector(features), expected, atol=1e-6, rtol=0)

    def test_initial_sampling_matches_fixed_at_rate_one_and_two(self):
        for rate, stage_name in ((1, "layer3"), (2, "layer5")):
            with self.subTest(rate=rate, stage=stage_name):
                model = small_pidnet().eval()
                stage = getattr(model, stage_name)
                original = stage[1].conv2
                other_modules = dict(model.named_modules())
                original_state = copy.deepcopy(model.state_dict())
                rng = torch.random.get_rng_state()
                configure_i_atrous(stage, mode="fadc", block_index=1, rate=rate)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
                conv = stage[1].conv2
                self.assertFalse(conv.training)
                self.assertTrue(torch.equal(conv.weight, original.weight))
                for name, value in original_state.items():
                    self.assertTrue(torch.equal(model.state_dict()[name], value), name)
                selected_name = stage_name + ".1.conv2"
                for name, module in other_modules.items():
                    if name != selected_name:
                        self.assertIs(dict(model.named_modules())[name], module)

                conv = conv.to(self.device)
                inputs = torch.randn(2, conv.in_channels, 13, 15, device=self.device)
                with torch.no_grad():
                    actual = conv(inputs)
                    expected = F.conv2d(inputs, original.weight.to(self.device), padding=rate, dilation=rate)
                    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=1e-5)

    def test_fadc_learns_and_reloads_with_real_ops(self):
        model = small_pidnet(atrous_mode="fadc", atrous_rate=2)
        conv = model.layer5[1].conv2.to(self.device)
        inputs = torch.randn(2, conv.in_channels, 19, 23, device=self.device, requires_grad=True)
        output = conv(inputs)
        self.assertEqual(output.shape, (2, conv.out_channels, 19, 23))
        self.assertTrue(torch.isfinite(output).all())
        loss = F.mse_loss(output, torch.randn_like(output))
        loss.backward()
        for name, parameter in conv.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name, grad in (
                ("input", inputs.grad),
                ("base weight", conv.weight.grad),
                ("AdaDR", conv.conv_offset.weight.grad),
                ("mask", conv.conv_mask.weight.grad),
                ("AdaKern low", conv.low_gates.gates.weight.grad),
                ("AdaKern high", conv.high_gates.gates.weight.grad),
                ("FreqSelect", conv.freq_select.gates.weight.grad),
        ):
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0, name)
            print("{} {} gradient L1: {:.6g}".format(self.device, name, grad.abs().sum().item()))
        self.assertTrue((conv.freq_select.gates.weight.grad.flatten(1).abs().sum(dim=1) > 0).all())

        torch.optim.SGD(conv.parameters(), lr=0.05).step()
        with torch.no_grad():
            selected = conv.freq_select(inputs)
            offsets = conv._offsets(selected)
            # Top-middle (dy, dx)=(-delta,0); middle stays put; bottom-middle is +delta.
            self.assertTrue(torch.equal(offsets[:, 2], -offsets[:, 14]))
            self.assertEqual(torch.count_nonzero(offsets[:, (3, 8, 9, 15)]).item(), 0)
            self.assertGreater(offsets[:, 2].std().item(), 0)
            kernels = conv._adaptive_weight(selected)
            self.assertFalse(torch.equal(kernels[0], kernels[1]))
            trained = conv(inputs)
            self.assertFalse(torch.equal(trained, output))
            torch.testing.assert_close(conv(inputs[:1]), trained[:1], atol=4e-6, rtol=1e-5)

        # Round-trip learned parameters through an actual state_dict serialization.
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        raw = torch.load(saved, map_location="cpu")
        saved_signature = "_i_atrous_signature_fadc_s5_b1_r2"
        self.assertIn(saved_signature, raw)
        rebuilt = small_pidnet(atrous_mode="fadc", atrous_rate=2)
        loaded = rebuilt.load_state_dict(raw, strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        with torch.no_grad():
            reloaded = rebuilt.layer5[1].conv2.to(self.device)(inputs)
            torch.testing.assert_close(reloaded, trained, atol=0, rtol=0)

        wrong = small_pidnet(atrous_mode="fadc", atrous_rate=4)
        with self.assertRaisesRegex(RuntimeError, saved_signature):
            load_model_checkpoint(wrong, raw)
        self.assertEqual(load_model_checkpoint(wrong, raw, strict=False), len(raw) - 1)
        self.assertEqual(wrong.layer5[1].conv2.nominal_rate, 4)
        self.assertIn("_i_atrous_signature_fadc_s5_b1_r4", wrong.state_dict())
        self.assertNotIn(saved_signature, wrong.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
