"""Load legacy and ablation model checkpoints with explicit completeness checks."""

import logging
import os

import torch

from models.context import DenseASPP


logger = logging.getLogger(__name__)


def _model_key(key):
    while key.startswith(('module.', 'model.')):
        key = key.split('.', 1)[1]
    return key


def _read_state_dict(checkpoint):
    if isinstance(checkpoint, (str, bytes, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a state dictionary or contain state_dict')
    source = checkpoint.get('state_dict', checkpoint)
    if not isinstance(source, dict):
        raise ValueError('Checkpoint state_dict must be a dictionary')
    return source


def _compatible(value, tensor):
    return (torch.is_tensor(value) and value.shape == tensor.shape
            and value.layout == tensor.layout and value.device.type != 'meta')


def load_l_pretrained(model, checkpoint, source_profile):
    """Load only the shared tensors required by the selected L training source."""
    if source_profile not in ('imagenet-l', 'cityscapes-l'):
        raise ValueError('Unknown PIDNet-L pretraining source: {}'.format(source_profile))
    if isinstance(checkpoint, (str, bytes, os.PathLike)):
        if not os.fsdecode(checkpoint).strip():
            raise ValueError('{} requires explicit MODEL.PRETRAINED'.format(source_profile))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('{} pretrained checkpoint does not exist: {}'.format(
                source_profile, checkpoint))

    source = {}
    for key, value in _read_state_dict(checkpoint).items():
        if isinstance(key, str):
            name = _model_key(key)
            if name in source:
                raise ValueError('Checkpoint contains duplicate model tensor {}'.format(name))
            source[name] = value

    # Validate the classification tensors before dropping them from transfer.
    projection = 'final_layer.conv2.'
    weight, bias = (source.get(projection + name) for name in ('weight', 'bias'))
    if source_profile == 'cityscapes-l':
        if not (torch.is_tensor(weight) and weight.shape == (19, 256, 1, 1)
                and torch.is_tensor(bias) and bias.shape == (19,)):
            raise RuntimeError('cityscapes-l requires a 19-class PIDNet-L final classification head')
    elif torch.is_tensor(weight) and weight.ndim == 4 and weight.shape[0] in (11, 19):
        raise RuntimeError('imagenet-l cannot use an 11/19-class segmentation checkpoint')

    backbone = ('conv1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.')
    skip = ('spp.', 'seghead_p.', 'seghead_d.', projection, '_i_atrous_signature_')
    fadc_children = ('freq_select', 'conv_offset', 'conv_mask', 'low_gates', 'high_gates')
    matched, missing, mismatched = {}, [], []
    for name, tensor in model.state_dict().items():
        if source_profile == 'imagenet-l' and not name.startswith(backbone):
            continue
        if name.startswith(skip) or ('.conv2.' in name and
                                    name.split('.conv2.', 1)[1].split('.', 1)[0] in fadc_children):
            continue
        if name not in source:
            # PyTorch's older BatchNorm state dictionaries omit this counter.
            if not name.endswith('.num_batches_tracked'):
                missing.append(name)
        elif not _compatible(source[name], tensor):
            mismatched.append(name)
        else:
            matched[name] = source[name]
    if missing or mismatched or not matched:
        raise RuntimeError('{} does not match the required PIDNet-L tensors: missing={}, shape_mismatch={}'.format(
            source_profile, missing, mismatched))
    model.load_state_dict(matched, strict=False)
    return len(matched)


def load_model_checkpoint(model, checkpoint, *, strict=True, prediction=False):
    """Load raw/state_dict checkpoints, accepting model/module wrapper prefixes.

    Legacy training pretraining can use ``strict=False`` to retain only matching
    tensors, including same-shape weights from a differently configured
    DenseASPP. The incompatible rate marker is skipped and reported separately.
    Inference always checks every model tensor; only loss buffers and, for
    prediction, auxiliary heads are ignored. Returns the number of matched keys.
    """
    source = _read_state_dict(checkpoint)
    target = model.state_dict()
    dense_modules = {(name + '.' if name else '') + '_dense_rates': module
                     for name, module in model.named_modules(remove_duplicate=False) if isinstance(module, DenseASPP)}
    matched, unexpected, shape_mismatch, rate_mismatch, ignored = {}, [], {}, [], {}
    for key, value in source.items():
        if not isinstance(key, str):
            unexpected.append(repr(key))
            continue
        name = _model_key(key)
        if name.startswith(('sem_loss.', 'bd_loss.')):
            ignored[key] = 'loss buffer'
            continue
        if prediction and name.startswith(('seghead_p.', 'seghead_d.')):
            ignored[key] = 'prediction excludes auxiliary head'
            continue
        if name not in target:
            unexpected.append(key)
        elif name in dense_modules and not dense_modules[name]._checkpoint_rates_match(value):
            rate_mismatch.append(name)
        elif not _compatible(value, target[name]):
            shape_mismatch[key] = name
        elif name in matched:
            raise ValueError('Checkpoint contains duplicate model tensor {}'.format(name))
        else:
            matched[name] = value

    missing = sorted(set(target) - set(matched) - set(shape_mismatch.values()) - set(rate_mismatch))
    incomplete = missing or unexpected or shape_mismatch or rate_mismatch
    details = 'missing={}, unexpected={}, shape_mismatch={}, rate_mismatch={}{}; ignored={}'.format(
        missing, sorted(unexpected), sorted(shape_mismatch), sorted(rate_mismatch),
        ' (configured rates differ)' if rate_mismatch else '', sorted(ignored.items()))
    if strict and incomplete:
        raise RuntimeError('Checkpoint does not match model: ' + details)
    if not matched:
        raise RuntimeError('Checkpoint has no matching model tensors: ' + details)
    model.load_state_dict(matched, strict=strict)
    if incomplete:
        logger.warning('Partially loaded %d/%d model tensors; skipped incompatible entries: %s',
                       len(matched), len(target), details)
    elif ignored:
        logger.info('Loaded %d model tensors with intentionally ignored entries: %s', len(matched), details)
    return len(matched)
