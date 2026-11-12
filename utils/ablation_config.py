"""Registration and validation for optional PIDNet ablations."""

import math
from yacs.config import CfgNode as CN

from models.context import validate_dense_rates


def _defaults():
    node = CN()
    node.PRETRAIN_SOURCE = 'legacy'
    node.CONTEXT = 'auto'
    node.DENSE_RATES = [1, 2, 3, 4]
    node.I_ATROUS = CN()
    node.I_ATROUS.MODE = 'none'
    node.I_ATROUS.STAGE = 5
    node.I_ATROUS.BLOCK = 1
    node.I_ATROUS.RATE = 2
    node.BOUNDARY = CN()
    node.BOUNDARY.TARGET = 'legacy'
    node.BOUNDARY.WIDTH = 0
    node.BOUNDARY.MASK_VOID = False
    node.BOUNDARY.VOID_RADIUS = 0
    node.BOUNDARY.DICE_WEIGHT = 0.0
    node.BOUNDARY.SINGLE_CLASS_BCE_WEIGHT = 0.0
    return node


def register_ablation_config(cfg):
    """Install missing defaults on the actual cfg before yacs merges input."""
    was_frozen = cfg.is_frozen()
    cfg.defrost()
    defaults = _defaults()
    if hasattr(cfg, 'ABLATION'):
        defaults.merge_from_other_cfg(cfg.ABLATION)
    cfg.ABLATION = defaults
    if was_frozen:
        cfg.freeze()
    return cfg


def _ablation(cfg):
    if not hasattr(cfg, 'ABLATION'):
        return _defaults()
    # Support old/partial configuration objects without mutating them.
    defaults = _defaults()
    if isinstance(cfg.ABLATION, CN):
        defaults.merge_from_other_cfg(cfg.ABLATION)
        return defaults
    return cfg.ABLATION


def get_pretrain_source(cfg):
    source = getattr(_ablation(cfg), 'PRETRAIN_SOURCE', 'legacy')
    if source == 'legacy':
        return source
    expected = {'imagenet-l': ('cityscapes', 19), 'cityscapes-l': ('camvid', 11)}
    if source not in expected:
        raise ValueError('ABLATION.PRETRAIN_SOURCE must be legacy, imagenet-l, or cityscapes-l')
    if cfg.MODEL.NAME not in ('pidnet_large', 'pidnet_l', 'pidnet-l'):
        raise ValueError('{} requires a PIDNet-L model name'.format(source))
    if (cfg.DATASET.DATASET, cfg.DATASET.NUM_CLASSES) != expected[source]:
        dataset, classes = expected[source]
        raise ValueError('{} requires {} with {} classes'.format(source, dataset, classes))
    return source


def get_model_kwargs(cfg):
    options = _ablation(cfg)
    get_pretrain_source(cfg)
    context = getattr(options, 'CONTEXT', 'auto')
    if context not in ('auto', 'pappm', 'dappm', 'pmpm', 'denseaspp'):
        raise ValueError('ABLATION.CONTEXT must be auto, pappm, dappm, pmpm, or denseaspp')
    rates = validate_dense_rates(getattr(options, 'DENSE_RATES', [1, 2, 3, 4]))
    atrous = getattr(options, 'I_ATROUS', _defaults().I_ATROUS)
    mode = getattr(atrous, 'MODE', 'none')
    stage = getattr(atrous, 'STAGE', 5)
    block = getattr(atrous, 'BLOCK', 1)
    rate = getattr(atrous, 'RATE', 2)
    if mode not in ('none', 'fixed', 'fadc'):
        raise ValueError('ABLATION.I_ATROUS.MODE must be none, fixed, or fadc')
    if type(stage) is not int or stage not in (3, 4, 5):
        raise ValueError('ABLATION.I_ATROUS.STAGE must be 3, 4, or 5')
    name = cfg.MODEL.NAME
    n = 3 if 's' in name or 'm' in name else 4
    block_count = n if stage in (3, 4) else 2
    if type(block) is not int or not 1 <= block < block_count:
        raise ValueError('ABLATION.I_ATROUS.BLOCK must select a non-first block in the chosen stage')
    if type(rate) is not int or rate < 1:
        raise ValueError('ABLATION.I_ATROUS.RATE must be a positive integer')
    return dict(context=context, dense_rates=rates, atrous_mode=mode,
                atrous_stage=stage, atrous_block=block, atrous_rate=rate)


def get_boundary_kwargs(cfg):
    boundary = getattr(_ablation(cfg), 'BOUNDARY', _defaults().BOUNDARY)
    target = getattr(boundary, 'TARGET', 'legacy')
    width = getattr(boundary, 'WIDTH', 0)
    mask = getattr(boundary, 'MASK_VOID', False)
    radius = getattr(boundary, 'VOID_RADIUS', 0)
    dice = getattr(boundary, 'DICE_WEIGHT', 0.0)
    single = getattr(boundary, 'SINGLE_CLASS_BCE_WEIGHT', 0.0)
    if target not in ('legacy', 'valid'):
        raise ValueError('ABLATION.BOUNDARY.TARGET must be legacy or valid')
    if type(width) is not int or width < 0 or (target == 'valid' and width != 0 and width % 2 != 1):
        raise ValueError('BOUNDARY.WIDTH must be nonnegative; valid uses zero or a positive odd width')
    if type(mask) is not bool:
        raise ValueError('BOUNDARY.MASK_VOID must be boolean')
    if type(radius) is not int or radius < 0:
        raise ValueError('BOUNDARY.VOID_RADIUS must be a nonnegative integer')
    if isinstance(dice, bool) or not isinstance(dice, (int, float)) or not math.isfinite(dice) or dice < 0:
        raise ValueError('BOUNDARY.DICE_WEIGHT must be finite and nonnegative')
    if isinstance(single, bool) or not isinstance(single, (int, float)) or not math.isfinite(single) or single < 0:
        raise ValueError('BOUNDARY.SINGLE_CLASS_BCE_WEIGHT must be finite and nonnegative')
    return dict(bd_target=target, bd_dilate_size=width or (4 if target == 'legacy' else 3),
                bd_mask_void=mask, bd_void_radius=radius)


def get_boundary_loss_kwargs(cfg):
    kwargs = get_boundary_kwargs(cfg)
    boundary = getattr(_ablation(cfg), 'BOUNDARY', _defaults().BOUNDARY)
    return dict(dice_weight=float(getattr(boundary, 'DICE_WEIGHT', 0.0)),
                mask_void=kwargs['bd_mask_void'],
                single_class_bce_weight=float(getattr(boundary, 'SINGLE_CLASS_BCE_WEIGHT', 0.0)))
