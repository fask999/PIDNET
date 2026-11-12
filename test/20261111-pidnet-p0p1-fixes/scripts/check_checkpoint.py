"""CPU regressions for raw/partial loading and the explicit L-weight migration."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import logging
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest import mock

import torch
from torch import nn


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import eval as eval_cli
from configs import config
from models.context import DenseASPP
from models.pidnet import PIDNet, get_seg_model
from utils.checkpoint import load_model_checkpoint


SEED = 304
BASELINE = '4c158cf24ce432f0a8cb43364fae38d93cee0dc3'
ORIGINAL = ROOT / 'pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar'
CONVERTED = EXP / 'output/pidnet-l-cityscapes-zero-stem-bias.pth.tar'
BIASES = ('conv1.0.bias', 'conv1.3.bias')
RATES = (1, 2, 3, 4)
OTHER_RATES = (2, 4, 6, 8)


def fails(run, *messages):
    try:
        run()
    except RuntimeError as exc:
        assert all(message in str(exc) for message in messages), str(exc)
        return str(exc)
    raise AssertionError('Expected RuntimeError containing {!r}'.format(messages))


def assert_state(model, expected):
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k], v) for k, v in expected.items())


def snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def small_pidnet(rates):
    return PIDNet(planes=8, ppm_planes=8, head_planes=16, context='denseaspp', dense_rates=rates)


def nested(child):
    wrapper = nn.Module()
    wrapper.net = child
    return wrapper


def check_dilation(module, rates):
    assert tuple(module._dense_rates.tolist()) == rates
    assert [layer[-1].dilation for layer in module.layers] == [(r, r) for r in rates]
    assert [layer[-1].padding for layer in module.layers] == [(r, r) for r in rates]


def check_raw():
    def dense(rates):
        return DenseASPP(4, 2, 4, rates=rates)

    for build, key, get_dense in (
            (dense, '_dense_rates', lambda m: m),
            (lambda rates: nested(dense(rates)), 'net._dense_rates', lambda m: m.net),
            (small_pidnet, 'spp._dense_rates', lambda m: m.spp),
            (lambda rates: nested(small_pidnet(rates)), 'net.spp._dense_rates', lambda m: m.net.spp)):
        source = dict(build(RATES).state_dict())  # existing key format, no extra signatures
        for strict in (True, False):
            same = build(RATES)
            incompat = same.load_state_dict(source, strict=strict)
            assert not incompat.missing_keys and not incompat.unexpected_keys
            assert_state(same, source)
            check_dilation(get_dense(same), RATES)

            other = build(OTHER_RATES)
            before = snapshot(other)
            assert any(not torch.equal(value, before[name]) for name, value in source.items() if name != key)
            fails(lambda: other.load_state_dict(source, strict=strict), key, 'configured rates')
            assert_state(other, before)
            check_dilation(get_dense(other), OTHER_RATES)

            malformed = dict(source)
            malformed[key] = source[key].to(torch.float32)
            fails(lambda: same.load_state_dict(malformed, strict=strict), key, 'configured rates')
            assert_state(same, source)

        missing = dict(source)
        del missing[key]
        fails(lambda: same.load_state_dict(missing), key, 'Missing key(s)')
        assert same.load_state_dict(missing, strict=False).missing_keys == [key]
        check_dilation(get_dense(same), RATES)
        print('raw {}: strict/partial same-rate and nested load, atomic rate rejection, missing marker and dilation passed'.format(key))


def check_partial():
    source = small_pidnet(RATES).state_dict()
    target = small_pidnet(OTHER_RATES)
    before = snapshot(target)
    wrapped = {'module.model.' + k: v for k, v in source.items()}
    fails(lambda: load_model_checkpoint(target, wrapped), "rate_mismatch=['spp._dense_rates']", 'configured rates differ')
    assert_state(target, before)

    missing, shape, rate = 'conv1.0.bias', 'conv1.3.weight', 'spp._dense_rates'
    del wrapped['module.model.' + missing]
    wrapped['module.model.' + shape] = torch.zeros(1)
    wrapped.update({'module.model.nonexistent': torch.zeros(1), 'module.sem_loss.weight': torch.ones(19),
                    'bd_loss.pos_weight': torch.ones(1)})
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger('utils.checkpoint')
    logger.addHandler(handler)
    try:
        count = load_model_checkpoint(target, {'state_dict': wrapped}, strict=False)
    finally:
        logger.removeHandler(handler)
    assert type(count) is int and count == len(source) - 3
    assert_state(target, {k: before[k] if k in (missing, shape, rate) else v for k, v in source.items()})
    assert not torch.equal(target.conv1[0].weight, before['conv1.0.weight'])
    check_dilation(target.spp, OTHER_RATES)
    log = stream.getvalue().strip()
    assert all(item in log for item in (
        "missing=['conv1.0.bias']", "shape_mismatch=['module.model.conv1.3.weight']",
        "rate_mismatch=['spp._dense_rates']", "unexpected=['module.model.nonexistent']",
        "ignored=[('bd_loss.pos_weight', 'loss buffer'), ('module.sem_loss.weight', 'loss buffer')]")), log
    print(log)


def model_config(size):
    local = config.clone()
    local.defrost()
    local.merge_from_file(str(ROOT / 'configs/cityscapes/pidnet_{}_cityscapes.yaml'.format(size)))
    return local


@torch.no_grad()
def check_defaults(sample):
    old = ModuleType('models._checkpoint_baseline')
    code = subprocess.check_output(['git', 'show', BASELINE + ':models/pidnet.py'], cwd=str(ROOT))
    exec(compile(code, BASELINE + ':models/pidnet.py', 'exec'), old.__dict__)
    for size, m, n, planes, ppm, head in (('small', 2, 3, 32, 96, 128),
                                          ('medium', 2, 3, 64, 96, 128),
                                          ('large', 3, 4, 64, 112, 256)):
        torch.manual_seed(SEED)
        legacy = old.PIDNet(m=m, n=n, planes=planes, ppm_planes=ppm, head_planes=head).eval()
        torch.manual_seed(SEED)
        current = get_seg_model(model_config(size), False, load_pretrained=False).eval()
        legacy_state = legacy.state_dict()
        assert_state(current, legacy_state)
        assert load_model_checkpoint(current, legacy_state) == len(legacy_state)
        assert all(torch.equal(a, b) for a, b in zip(legacy(sample), current(sample)))
        print('default {}: {} legacy state entries and all three outputs match HEAD exactly'.format(size, len(legacy_state)))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_conversion_guards(source, converted, original_keys):
    def cli(input_path, output, error, flag=True):
        args = [sys.executable, '-B', 'tools/convert_pidnet_l_stem_bias.py',
                '--source', str(input_path), '--output', str(output)]
        if flag:
            args.append('--assume-biasless-stem')
        proc = subprocess.run(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.returncode == 2 and error in proc.stdout, proc.stdout

    converted_sha = sha256(converted)
    manifest = converted.with_name(converted.name[:-len('.pth.tar')] + '.manifest.json')
    manifest_sha = sha256(manifest)
    with TemporaryDirectory(prefix='check-converter-', dir=str(EXP / 'output')) as scratch:
        scratch = Path(scratch)
        out = scratch / 'new.pth.tar'
        out_manifest = scratch / 'new.manifest.json'
        cli(source, out, '--assume-biasless-stem', flag=False)
        cli(source, converted, 'existing or aliased output')
        cli(source, ROOT / 'forbidden.pth.tar', 'Output must be inside')
        cli(source, source, 'Output must be inside')
        out.symlink_to(source)
        cli(source, out, 'existing or aliased output')
        assert out.is_symlink()
        out.unlink()
        out_manifest.write_text('existing manifest', encoding='utf-8')
        cli(source, out, 'existing or aliased output')
        assert out_manifest.read_text(encoding='utf-8') == 'existing manifest' and not out.exists()
        out_manifest.unlink()

        wrong_keys = scratch / 'wrong-keys.pth.tar'
        wrong_shape = scratch / 'wrong-shape.pth.tar'
        tiny = torch.zeros(1)
        keys = {key: tiny for key in original_keys}
        torch.save({'epoch': -1, 'state_dict': keys}, wrong_shape)
        del keys['model.conv1.0.weight']
        keys['model.some_other_tensor'] = tiny
        torch.save({'epoch': -1, 'state_dict': keys}, wrong_keys)
        cli(wrong_keys, out, 'except for the two stem biases')
        cli(wrong_shape, out, 'does not match target')
        cli(wrong_shape, wrong_shape, 'existing or aliased output')
        assert not out.exists() and not out_manifest.exists()
    assert sha256(converted) == converted_sha and sha256(manifest) == manifest_sha
    print('converter guards: flag, output/source aliases, output boundary, existing manifest, keys and shape passed')


class LoadedBeforeCuda(Exception):
    pass


def check_real_entrypoints(source, converted, original, migrated):
    # Read both packages from disk with weights_only above. Exercise the real
    # CLI calls with those objects and stop at CUDA, before dataset/prediction.
    payloads = {source: original, converted: migrated}

    def safe_read(path, map_location):
        assert map_location == 'cpu'
        return payloads[Path(path).resolve()]

    def cuda_stop(model):
        loaded.append(model)
        raise LoadedBeforeCuda

    def invoke(call, path):
        if path == source:
            message = fails(call, *BIASES)
            assert not loaded and "missing=['conv1.0.bias', 'conv1.3.bias']" in message
        else:
            try:
                call()
            except LoadedBeforeCuda:
                pass
            else:
                raise AssertionError('Expected the successful load to reach CUDA')
            assert len(loaded) == 1
            return loaded.pop().eval()

    loaded = []
    with mock.patch.object(torch, 'load', side_effect=safe_read), mock.patch.object(PIDNet, 'cuda', new=cuda_stop):
        for path in payloads:
            local = model_config('large')
            local.TEST.MODEL_FILE = str(path)
            local.freeze()
            eval_log = mock.Mock()
            with mock.patch.object(eval_cli, 'config', local), \
                 mock.patch.object(eval_cli, 'parse_args', return_value=argparse.Namespace(cfg='unused.yaml')), \
                 mock.patch.object(eval_cli, 'create_logger', return_value=(eval_log, str(EXP / 'output'), None)):
                eval_model = invoke(eval_cli.main, path)
            capture = io.StringIO()
            with mock.patch.object(sys, 'argv', ['tools/custom.py', '--a', 'pidnet-l', '--p', str(path)]), \
                 mock.patch('glob.glob', return_value=[]), redirect_stdout(capture):
                custom_model = invoke(lambda: runpy.run_path(str(ROOT / 'tools/custom.py'), run_name='__main__'), path)
            if path == converted:
                eval_log.info.assert_any_call('=> loaded %d model tensors', 545)
                assert 'Loaded 519 model tensors' in capture.getvalue()
    print('real strict entrypoints: original rejected for two missing biases; converted eval 545/545, custom 519/519')
    return eval_model, custom_model


@torch.no_grad()
def check_migration(source, converted, sample):
    before = sha256(source)
    original = torch.load(str(source), map_location='cpu', weights_only=True)
    migrated = torch.load(str(converted), map_location='cpu', weights_only=True)
    manifest = json.loads(converted.with_name(converted.name[:-len('.pth.tar')] + '.manifest.json').read_text(encoding='utf-8'))
    assert set(original) == {'state_dict', 'epoch'} and set(migrated) == {'state_dict'}
    raw, transformed = original['state_dict'], migrated['state_dict']
    assert len(raw) == 543 and len(transformed) == 545 and all(k.startswith('model.') for k in raw)
    expected_map = {k: k[len('model.'):] for k in raw}
    assert manifest['source_to_target'] == expected_map
    assert set(transformed) - set(expected_map.values()) == set(manifest['zero_filled_keys']) == set(BIASES)
    assert all(torch.equal(value, transformed[expected_map[key]]) for key, value in raw.items())
    assert all(torch.count_nonzero(transformed[key]).item() == 0 for key in BIASES)
    assert manifest['source'] == {'path': str(source), 'bytes': source.stat().st_size, 'epoch': original['epoch'],
                                  'sha256_before': before, 'sha256_after': before, 'tensor_count': 543}
    assert manifest['target'] == {'path': str(converted), 'sha256': sha256(converted), 'architecture': 'PIDNet-L',
                                  'dataset': 'Cityscapes', 'classes': 19, 'tensor_count': 545}
    assert 'no bias' in manifest['assumption'] and manifest['source_provenance_verified'] is False
    assert manifest['training_resume_compatible'] is False

    eval_model, custom_model = check_real_entrypoints(source, converted, original, migrated)
    reference = get_seg_model(model_config('large'), False, load_pretrained=False).eval()
    for idx in (0, 3):
        reference.conv1[idx].bias = None
    reference.load_state_dict({expected_map[k]: t for k, t in raw.items()}, strict=True)
    assert all(reference.conv1[idx].bias is None for idx in (0, 3))
    reference_outputs = reference(sample)
    differences = [(a - b).abs().max().item() for a, b in zip(eval_model(sample), reference_outputs)]
    custom_difference = (custom_model(sample) - reference_outputs[1]).abs().max().item()
    assert differences == [0.0, 0.0, 0.0] and custom_difference == 0.0, (differences, custom_difference)
    print('whole local model, no-bias vs zero-bias, max abs difference: aux_p={}, main={}, aux_d={}, custom={}'.format(
        *differences, custom_difference))
    check_conversion_guards(source, converted, raw)
    assert sha256(source) == before
    print('source SHA-256 unchanged: {}'.format(before))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ORIGINAL)
    parser.add_argument('--converted', type=Path, default=CONVERTED)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    sample = torch.randn(1, 3, 64, 96)
    print('CPU, torch {}, seed {}, input {}'.format(torch.__version__, SEED, tuple(sample.shape)), flush=True)
    check_raw()
    check_partial()
    check_defaults(sample)
    check_migration(args.source.resolve(strict=True), args.converted.resolve(strict=True), sample)
    print('checkpoint checks passed')


if __name__ == '__main__':
    main()
