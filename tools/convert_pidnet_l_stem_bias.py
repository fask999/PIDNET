"""Explicitly migrate a biasless-stem PIDNet-L Cityscapes weight package."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from types import SimpleNamespace
from zipfile import BadZipFile, ZipFile

import torch

import _init_paths
from models.pidnet import get_seg_model


OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'test/20261111-pidnet-p0p1-fixes/output'
STEM_BIASES = ('conv1.0.bias', 'conv1.3.bias')
ASSUMPTION = 'The source stem convolutions corresponding to conv1.0 and conv1.3 had no bias.'


def _sha256(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _read_weight_archive(stream):
    # Older torch versions can dispatch TorchScript or legacy archives through
    # other paths. Require the ordinary torch.save tensor ZIP before loading.
    if stream.read(4) != b'PK\x03\x04':
        raise ValueError('Source must be a torch.save tensor ZIP archive')
    stream.seek(0)
    with ZipFile(stream) as archive:
        members = [name.partition('/') for name in archive.namelist()]
    names = {name for _, _, name in members}
    fixed = {'data.pkl', 'version', 'byteorder', '.data/serialization_id',
             '.format_version', '.storage_alignment'}
    if (len({root for root, _, _ in members}) != 1 or len(names) != len(members)
            or not {'data.pkl', 'version'} <= names
            or any(not name or (name not in fixed and not re.fullmatch(r'data/[0-9]+', name))
                   for name in names)):
        raise ValueError('Source must be a torch.save tensor ZIP archive')
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=True)


def _map_tensors(payload):
    if (not isinstance(payload, dict) or set(payload) != {'epoch', 'state_dict'}
            or type(payload['epoch']) is not int or not isinstance(payload['state_dict'], dict)):
        raise ValueError('Expected an epoch integer and a state_dict in the source package')
    raw = payload['state_dict']
    if len(raw) != 543 or any(not isinstance(key, str) or not key.startswith('model.') for key in raw):
        raise ValueError('Expected exactly 543 source keys with the model. prefix')

    cfg = SimpleNamespace(MODEL=SimpleNamespace(NAME='pidnet_large'),
                          DATASET=SimpleNamespace(NUM_CLASSES=19))
    target = get_seg_model(cfg, imgnet_pretrained=False, load_pretrained=False).state_dict()
    mapping = {key: key[len('model.'):] for key in sorted(raw)}
    expected = set(target) - set(STEM_BIASES)
    actual = set(mapping.values())
    if (len(target) != 545 or not set(STEM_BIASES) <= set(target) or actual != expected):
        raise ValueError('Source must map to default PIDNet-L Cityscapes except for the two stem biases: '
                         'missing={}, unexpected={}'.format(sorted(expected - actual), sorted(actual - expected)))

    converted = {}
    for key, name in mapping.items():
        value = raw[key]
        reference = target[name]
        if (not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype
                or value.layout != reference.layout or value.device.type != 'cpu'):
            raise ValueError('Source tensor {} does not match target {} (shape {}, dtype {}, layout {})'.format(
                key, name, tuple(reference.shape), reference.dtype, reference.layout))
        converted[name] = value
    for name in STEM_BIASES:
        converted[name] = torch.zeros_like(target[name])
    return converted, mapping


def _output_paths(source, output):
    if not output.name.endswith('.pth.tar'):
        raise ValueError('Output name must end in .pth.tar')
    directory = output.parent.resolve(strict=True)
    if OUTPUT_DIR.resolve(strict=True) != OUTPUT_DIR or not (
            directory == OUTPUT_DIR or OUTPUT_DIR in directory.parents):
        raise ValueError('Output must be inside {}'.format(OUTPUT_DIR))
    output = directory / output.name
    manifest = directory / (output.name[:-len('.pth.tar')] + '.manifest.json')
    for path in (output, manifest):
        if path == source or os.path.lexists(path):
            raise FileExistsError('Refusing existing or aliased output: {}'.format(path))
    return output, manifest


def convert(source, output, *, assume_biasless_stem):
    if not assume_biasless_stem:
        raise ValueError('Explicitly pass --assume-biasless-stem to confirm the conversion assumption')
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError('Source must be a file: {}'.format(source))
    output, manifest = _output_paths(source, Path(output))
    with source.open('rb') as stream:
        before = _sha256(stream)
        source_size = os.fstat(stream.fileno()).st_size
        payload = _read_weight_archive(stream)
    converted, mapping = _map_tensors(payload)

    created = []
    try:
        with output.open('x+b') as weights:
            created.append(output)
            with manifest.open('x', encoding='utf-8') as description:
                created.append(manifest)
                # This is a model-only package, with no epoch or optimizer state.
                torch.save({'state_dict': converted}, weights)
                weights.flush()
                converted_sha = _sha256(weights)
                with source.open('rb') as stream:
                    after = _sha256(stream)
                if after != before:
                    raise RuntimeError('Source SHA-256 changed during conversion')
                json.dump({
                    'manifest_version': 1,
                    'source': {'path': str(source), 'bytes': source_size, 'epoch': payload['epoch'],
                               'sha256_before': before, 'sha256_after': after, 'tensor_count': len(mapping)},
                    'target': {'path': str(output), 'sha256': converted_sha, 'architecture': 'PIDNet-L',
                               'dataset': 'Cityscapes', 'classes': 19, 'tensor_count': len(converted)},
                    'source_to_target': mapping,
                    'zero_filled_keys': list(STEM_BIASES),
                    'assumption': ASSUMPTION,
                    'source_provenance_verified': False,
                    'training_resume_compatible': False,
                }, description, indent=2, sort_keys=True)
                description.write('\n')
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='original PIDNet-L model package (read only)')
    parser.add_argument('--output', required=True, type=Path, help='new .pth.tar path inside {}'.format(OUTPUT_DIR))
    parser.add_argument('--assume-biasless-stem', action='store_true', required=True,
                        help='explicitly acknowledge that both missing stem biases are assumed absent in the source')
    args = parser.parse_args()
    try:
        output, manifest = convert(args.source, args.output, assume_biasless_stem=args.assume_biasless_stem)
    except (OSError, ValueError, TypeError, RuntimeError, BadZipFile, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    print('Wrote {} and {}'.format(output, manifest))


if __name__ == '__main__':
    main()
