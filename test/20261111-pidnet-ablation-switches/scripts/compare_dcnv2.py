"""Optional backend evidence: identical DCNv2 inputs on MMCV CPU and torchvision CUDA."""

import torch
import torchvision
from mmcv import __version__ as mmcv_version
from mmcv.ops import get_compiling_cuda_version, modulated_deform_conv2d
from torchvision.ops import deform_conv2d


def main():
    torch.manual_seed(304)
    batch, cin, cout, height, width = 2, 4, 5, 7, 9
    x = torch.randn(1, batch * cin, height, width)
    offset = 0.35 * torch.randn(1, batch * 18, height, width)
    mask = torch.rand(1, batch * 9, height, width)
    weight = torch.randn(batch * cout, cin, 3, 3)
    bias = torch.randn(batch * cout)
    vjp = torch.randn(1, batch * cout, height, width)
    originals = (x, offset, mask, weight, bias)
    cpu = [item.clone().requires_grad_() for item in originals]
    gpu = [item.cuda().requires_grad_() for item in originals]
    shape_args = ((1, 1), (1, 1), (1, 1))
    actual_cpu = modulated_deform_conv2d(*cpu, *shape_args, batch, batch)
    actual_gpu = deform_conv2d(gpu[0], gpu[1], gpu[3], gpu[4], *shape_args, mask=gpu[2])
    grad_cpu = torch.autograd.grad(actual_cpu, cpu, vjp)
    grad_gpu = torch.autograd.grad(actual_gpu, gpu, vjp.cuda())
    print("seed 304; batch {}, groups {}, deform groups {}; feature {}x{}".format(
        batch, batch, batch, height, width))
    print("torch {}; MMCV {} compiled CUDA {}; torchvision {} compiled CUDA {}; {}".format(
        torch.__version__, mmcv_version, get_compiling_cuda_version(), torchvision.__version__,
        torch.ops.torchvision._cuda_version(), torch.cuda.get_device_name(0)))
    for name, a, b in zip(("output", "input grad", "offset grad", "mask grad", "weight grad", "bias grad"),
                          (actual_cpu,) + grad_cpu, (actual_gpu,) + grad_gpu):
        b = b.detach().cpu()
        torch.testing.assert_close(a.detach(), b, atol=2e-5, rtol=2e-5)
        print("{} maximum absolute difference: {:.9g}".format(name, (a.detach() - b).abs().max().item()))


if __name__ == "__main__":
    main()
