"""Optional I-branch contexts. The pooling and rates are local design choices."""

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import PAPPM


def validate_dense_rates(rates):
    if not isinstance(rates, (tuple, list)) or len(rates) != 4:
        raise ValueError('DENSE_RATES must contain exactly four positive integers')
    if any(type(rate) is not int or rate < 1 for rate in rates):
        raise ValueError('DENSE_RATES must contain exactly four positive integers')
    return tuple(rates)


class MixedPool(nn.Module):
    """Split channels across square, horizontal and vertical average pools."""

    def __init__(self, kernel_size, stride):
        super(MixedPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if x.shape[1] < 3:
            raise ValueError('MixedPool needs at least three channels')
        channels = x.shape[1]
        a, b, c = torch.split(x, [channels // 3, channels // 3,
                                  channels - 2 * (channels // 3)], dim=1)
        k, s, p = self.kernel_size, self.stride, self.kernel_size // 2
        square = F.avg_pool2d(a, k, s, p)
        horizontal = F.avg_pool2d(b, (1, k), (s, s), (0, p))
        vertical = F.avg_pool2d(c, (k, 1), (s, s), (p, 0))
        return torch.cat((square, horizontal, vertical), dim=1)


class PMPM(PAPPM):
    """PAPPM parallel fusion with three mixed-pooling scales.

    Existing scale projections mix channels after the concatenated pools. This
    is an implementation choice, not a claim of the paper's exact settings.
    """

    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PMPM, self).__init__(inplanes, branch_planes, outplanes, BatchNorm)
        for scale, kernel, stride in ((self.scale1, 5, 2), (self.scale2, 9, 4),
                                      (self.scale3, 17, 8)):
            scale[0] = MixedPool(kernel, stride)
        # PAPPM's learned tensors have the same shapes; a persistent marker
        # lets strict loading distinguish the two context architectures.
        self.register_buffer('_context_signature', torch.tensor(1, dtype=torch.uint8))


class DenseASPP(nn.Module):
    """Four densely connected projected atrous layers with residual fusion."""

    def __init__(self, inplanes, branch_planes, outplanes, rates=(1, 2, 3, 4)):
        super(DenseASPP, self).__init__()
        rates = validate_dense_rates(rates)
        self._configured_rates = rates
        # Unlike I_ATROUS.RATE, these four rates belong to this new context's
        # own structure, so checkpoints can validate them on reload.
        self.register_buffer('_dense_rates', torch.tensor(rates, dtype=torch.int64))
        self.layers = nn.ModuleList()
        for i, rate in enumerate(rates):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(inplanes + i * branch_planes, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes + i * branch_planes, branch_planes, 1, bias=False),
                nn.BatchNorm2d(branch_planes, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, 3, padding=rate,
                          dilation=rate, bias=False)))
        self.compression = nn.Sequential(
            nn.BatchNorm2d(inplanes + 4 * branch_planes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes + 4 * branch_planes, outplanes, 1, bias=False))
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=0.1), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, 1, bias=False))

    def _checkpoint_rates_match(self, value):
        return (torch.is_tensor(value) and value.dtype == torch.int64
                and value.layout == torch.strided and value.device.type != 'meta'
                and value.shape == (len(self._configured_rates),)
                and tuple(value.tolist()) == self._configured_rates)

    def _check_checkpoint_rates(self, state_dict, prefix):
        key = prefix + '_dense_rates'
        if key in state_dict and not self._checkpoint_rates_match(state_dict[key]):
            raise RuntimeError('DenseASPP checkpoint {} does not match configured rates {}'.format(
                key, self._configured_rates))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Raise before loading this subtree. Appending to error_msgs would let
        # PyTorch continue copying its descendants, even on a geometry error.
        self._check_checkpoint_rates(state_dict, prefix)
        super(DenseASPP, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.compression(torch.cat(features, dim=1)) + self.shortcut(x)
