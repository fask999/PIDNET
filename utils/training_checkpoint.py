"""Validate the training state written by PIDNet's SGD training entry point."""

import math
from numbers import Integral, Real

import torch


def _is_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite_number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def read_training_checkpoint(path):
    """Read a resume archive on CPU and check its envelope before setup."""
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except FileNotFoundError as exc:
        raise FileNotFoundError('TRAIN.RESUME requires checkpoint {}'.format(path)) from exc
    except Exception as exc:
        raise ValueError('Cannot read training checkpoint {}: {}'.format(path, exc)) from exc

    prefix = 'Invalid training checkpoint {}: '.format(path)
    if not isinstance(checkpoint, dict):
        raise ValueError(prefix + 'expected a dictionary containing training state')
    epoch = checkpoint.get('epoch')
    if not _is_integer(epoch) or epoch < 0:
        raise ValueError(prefix + 'epoch must be a nonnegative integer')
    best = checkpoint.get('best_mIoU')
    if not _is_finite_number(best) or not 0 <= best <= 1:
        raise ValueError(prefix + 'best_mIoU must be a finite number in [0, 1]')
    if not isinstance(checkpoint.get('state_dict'), dict) or not checkpoint['state_dict']:
        raise ValueError(prefix + 'state_dict must be a nonempty model state dictionary')
    optimizer = checkpoint.get('optimizer')
    if not isinstance(optimizer, dict):
        raise ValueError(prefix + 'optimizer must be an SGD state dictionary')
    if not isinstance(optimizer.get('state'), dict):
        raise ValueError(prefix + 'optimizer.state must be a dictionary')
    if not isinstance(optimizer.get('param_groups'), list):
        raise ValueError(prefix + 'optimizer.param_groups must be a list')
    return checkpoint


def restore_sgd_state(optimizer, saved, path):
    """Check positional SGD state before PyTorch attaches it to live parameters."""
    prefix = 'Invalid training checkpoint {}: optimizer.'.format(path)
    groups = saved['param_groups']
    if len(groups) != len(optimizer.param_groups):
        raise ValueError(prefix + 'param_groups count does not match the current SGD optimizer')

    parameters, validated_groups = [], []
    for i, (group, current) in enumerate(zip(groups, optimizer.param_groups)):
        where = prefix + 'param_groups[{}]'.format(i)
        if not isinstance(group, dict):
            raise ValueError(where + ' must be a dictionary')
        ids = group.get('params')
        end = len(parameters) + len(current['params'])
        # Native SGD archives number parameters in group order. These positional
        # IDs cannot detect a model reordering that leaves the shapes unchanged.
        if (not isinstance(ids, list) or any(not _is_integer(index) for index in ids)
                or ids != list(range(len(parameters), end))):
            raise ValueError(where + '.params do not match the current SGD parameter positions')
        parameters.extend(current['params'])

        group = group.copy()
        scalar_limit = min((torch.finfo(param.dtype).max for param in current['params']
                            if param.is_floating_point()), default=math.inf)
        for name in ('lr', 'momentum', 'dampening', 'weight_decay'):
            value = group.get(name)
            if (not _is_finite_number(value) or abs(value) > scalar_limit
                    or (name != 'dampening' and value < 0)):
                raise ValueError(where + '.' + name + ' is missing or invalid for SGD')
            group[name] = float(value)
        # PyTorch fills in options missing from older SGD archives.
        for name in ('nesterov', 'maximize', 'differentiable'):
            if name in group and not isinstance(group[name], bool):
                raise ValueError(where + '.' + name + ' must be a boolean')
        for name in ('foreach', 'fused'):
            if group.get(name) is not None and not isinstance(group[name], bool):
                raise ValueError(where + '.' + name + ' must be a boolean or None')
        if group.get('nesterov') and (group['momentum'] <= 0 or group['dampening'] != 0):
            raise ValueError(where + '.nesterov requires positive momentum and zero dampening')
        if group.get('differentiable'):
            raise ValueError(where + '.differentiable is incompatible with PIDNet leaf parameters')
        if group.get('fused') and ('fused' not in optimizer.defaults or group.get('foreach')):
            raise ValueError(where + '.fused is unsupported by this SGD optimizer configuration')
        validated_groups.append(group)

    for index, state in saved['state'].items():
        where = prefix + 'state[{}]'.format(repr(index))
        if not _is_integer(index) or not 0 <= index < len(parameters):
            raise ValueError(where + ' does not refer to an SGD parameter')
        if not isinstance(state, dict) or state.keys() - {'momentum_buffer'}:
            raise ValueError(where + ' must contain only SGD momentum state')
        buffer = state.get('momentum_buffer')
        if buffer is not None:
            if not torch.is_tensor(buffer) or buffer.layout != torch.strided or not buffer.is_floating_point():
                raise ValueError(where + '.momentum_buffer must be a dense floating-point tensor or None')
            if buffer.shape != parameters[index].shape:
                raise ValueError(where + '.momentum_buffer shape {} does not match parameter shape {}'.format(
                    tuple(buffer.shape), tuple(parameters[index].shape)))

    try:
        optimizer.load_state_dict({'state': saved['state'], 'param_groups': validated_groups})
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(prefix + 'could not load SGD state: {}'.format(exc)) from exc
