"""Apply the I-branch dilation ablation after PIDNet initializes its weights."""

from torch import nn

from .model_utils import BasicBlock, Bottleneck


def configure_i_atrous(stage, *, mode, block_index, rate):
    """Change only the selected I-stage block's 3x3 conv2 in place."""
    if mode == "none":
        return
    if mode not in ("fixed", "fadc"):
        raise ValueError("I_ATROUS mode must be 'none', 'fixed', or 'fadc'")
    if type(rate) is not int or rate < 1:
        raise ValueError("I_ATROUS rate must be a positive integer")
    if type(block_index) is not int or not 0 < block_index < len(stage):
        raise ValueError("I_ATROUS must select a block after the I-stage downsampling block")

    block = stage[block_index]
    if (not isinstance(block, (BasicBlock, Bottleneck)) or
            block.stride != 1 or block.downsample is not None):
        raise ValueError("I_ATROUS requires a stride-1 BasicBlock or Bottleneck without downsampling")
    conv = block.conv2
    if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3) or
            conv.stride != (1, 1) or conv.padding != (1, 1) or
            conv.dilation != (1, 1) or conv.groups != 1 or conv.padding_mode != "zeros"):
        raise ValueError("I_ATROUS requires the original stride-1, zero-padded PIDNet conv2")

    if mode == "fixed":
        conv.dilation = conv.padding = (rate, rate)
    else:
        # The extension is optional for every configuration except FADC.
        from .fadc import FrequencyAdaptiveConv2d

        block.conv2 = FrequencyAdaptiveConv2d(conv, rate)
