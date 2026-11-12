# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Adapted from Linwei Chen et al.'s FADC implementation (Apache-2.0).
# See LICENSE-FADC for the license and fadc.py's docstring for changes.

"""PIDNet adaptation of Frequency-Adaptive Dilated Convolution (CVPR 2024).

Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html
Source: https://github.com/Linwei-Chen/FADC/blob/923102e6344d9b90ec427cb15201efd514bf316e/FADC_only/conv_custom.py

This adaptation keeps one stride-1 3x3 conv2, the author's (2, 4, 8) FFT
scales and both AdaKern components. It uses symmetric frequency masks, gates
the lowest band too, uses GroupNorm on pooled kernel descriptors (batch size
one is valid), and replaces the source's padding and initial gains with
PIDNet's zero padding and neutral gains. Softplus gives an exactly nominal,
trainable initial dilation including at rate 1. The reference's MMCV op is
replaced with torchvision's modulated DCNv2 op; the
adaptation needs neither an MMCV registry nor the optional DCT path.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


_DCNV2_HINT = (
    "ABLATION.I_ATROUS.MODE='fadc' needs torchvision's compiled deform_conv2d op. "
    "Install torchvision with CPU/CUDA ops matching the installed PyTorch "
    "and target device, or set ABLATION.I_ATROUS.MODE to 'fixed' or 'none'."
)

try:
    from torchvision.ops import deform_conv2d
    if not hasattr(torch.ops.torchvision, "deform_conv2d"):
        raise RuntimeError("torchvision did not register deform_conv2d")
except (ImportError, OSError, RuntimeError, AttributeError) as exc:
    raise RuntimeError(_DCNV2_HINT) from exc


def _zero_conv(in_channels, out_channels, kernel_size):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=kernel_size // 2, device="cpu")
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class _FrequencySelection(nn.Module):
    """Learn a spatial gain for each rectangular Fourier band."""

    scales = (2, 4, 8)

    def __init__(self, channels):
        super().__init__()
        self.gates = _zero_conv(channels, len(self.scales) + 1, 3)

    def forward(self, x):
        height, width = x.shape[-2:]
        # Half precision FFTs on CUDA require power-of-two dimensions.
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.rfft2(fft_input, norm="ortho")
        fy = torch.fft.fftfreq(height, device=x.device).abs()[:, None]
        fx = torch.fft.rfftfreq(width, device=x.device)[None, :]
        radius = torch.maximum(fy, fx)
        gains = 2 * self.gates(x).sigmoid() - 1
        previous = selected = x
        for index, scale in enumerate(self.scales):
            low = torch.fft.irfft2(spectrum * (radius < 0.5 / scale),
                                   s=(height, width), norm="ortho").to(x.dtype)
            selected = selected + (previous - low) * gains[:, index:index + 1]
            previous = low
        # Zero residual gains give the original x exactly.
        return selected + previous * gains[:, -1:]


class _ChannelGates(nn.Module):
    """Form per-sample input and output gains for one kernel component."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        hidden = max(in_channels // 16, 16)
        self.project = nn.Conv2d(in_channels, hidden, 1, bias=False, device="cpu")
        self.norm = nn.GroupNorm(1, hidden, device="cpu")
        self.gates = _zero_conv(hidden, in_channels + out_channels, 1)

    def forward(self, pooled):
        gains = 2 * self.gates(F.relu(self.norm(self.project(pooled)))).sigmoid()
        input_gain = gains[:, :self.in_channels]
        output_gain = gains[:, self.in_channels:]
        return output_gain.unsqueeze(2) * input_gain.unsqueeze(1)


class FrequencyAdaptiveConv2d(nn.Module):
    """Adapt the original PIDNet conv2, retaining its weight and bias keys."""

    def __init__(self, original, rate):
        super().__init__()
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.nominal_rate = rate
        self.weight = nn.Parameter(original.weight.detach().clone(),
                                   requires_grad=original.weight.requires_grad)
        self.register_parameter("bias", None)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(),
                                     requires_grad=original.bias.requires_grad)

        # The optional predictors must not advance the caller's random state.
        with torch.random.fork_rng(devices=[]):
            self.freq_select = _FrequencySelection(self.in_channels)
            self.conv_offset = _zero_conv(self.in_channels, 1, 3)
            self.conv_mask = _zero_conv(self.in_channels, 9, 3)
            self.low_gates = _ChannelGates(self.in_channels, self.out_channels)
            self.high_gates = _ChannelGates(self.in_channels, self.out_channels)

        # DCNv2 offsets interleave (dy, dx) in row-major 3x3 kernel order.
        grid = torch.tensor([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)], device="cpu")
        self.register_buffer("_relative_grid", grid.reshape(1, 18, 1, 1), persistent=False)
        self.to(device=original.weight.device, dtype=original.weight.dtype)
        self.train(original.training)

    def _offsets(self, features):
        # DCNv2 already samples the regular dilation-1 grid; add only D - 1.
        dilation = F.softplus(self.conv_offset(features)) / math.log(2) * self.nominal_rate
        return (dilation - 1) * self._relative_grid

    def _adaptive_weight(self, features):
        pooled = F.adaptive_avg_pool2d(features, 1)
        dc = self.weight.mean(dim=(-2, -1), keepdim=True)
        residual = self.weight - dc
        return (self.weight + dc * (self.low_gates(pooled) - 1)
                + residual * (self.high_gates(pooled) - 1))

    def forward(self, x):
        x = self.freq_select(x)
        batch, _, height, width = x.shape
        offset = self._offsets(x)
        mask = 2 * self.conv_mask(x).sigmoid()
        weight = self._adaptive_weight(x)
        bias = None if self.bias is None else self.bias.repeat(batch)

        # torchvision infers one conv/deform group per sample from these shapes.
        try:
            output = deform_conv2d(
                x.reshape(1, -1, height, width).contiguous(),
                offset.reshape(1, -1, height, width).contiguous(),
                weight.reshape(-1, self.in_channels, 3, 3).contiguous(),
                bias=bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                mask=mask.reshape(1, -1, height, width).contiguous(),
            )
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if any(detail in message for detail in (
                "could not run 'torchvision::deform_conv2d'",
                "couldn't load custom c++ ops", "not compiled with",
                "not implemented for", "no kernel image", "invalid device function",
            )):
                raise RuntimeError(_DCNV2_HINT) from exc
            raise
        return output.reshape(batch, self.out_channels, height, width)
