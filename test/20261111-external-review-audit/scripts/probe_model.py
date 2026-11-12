"""Run synthetic probes against the current model modules; no repo weights/data."""
import contextlib
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import models.fadc as fadc
from models.context import PMPM
from models.i_atrous import configure_i_atrous
from models.pidnet import get_pred_model
from utils.checkpoint import load_model_checkpoint

SEED = 304


def peak(x):
    return x.detach().abs().max().item()


def dtype(x):
    return str(x.dtype).replace('torch.', '') if x is not None else 'None'


def probe_amp(device, amp=False, upstream=False):
    torch.manual_seed(SEED)
    original = nn.Conv2d(4, 6, 3, padding=1, bias=False)
    layer = fadc.FrequencyAdaptiveConv2d(original, 2).to(device)
    upstream_conv = nn.Conv2d(4, 4, 1).to(device)
    x = torch.randn(2, 4, 7, 11, device=device, requires_grad=True)
    real_deform = fadc.deform_conv2d
    real_softplus = fadc.F.softplus
    captured = {}

    def record_softplus(x, *args, **kwargs):
        result = real_softplus(x, *args, **kwargs)
        print('  offset conv output -> softplus: {} -> {}'.format(dtype(x), dtype(result)), flush=True)
        return result

    def record(x, offset, weight, bias=None, **kwargs):
        captured.update(x=x, offset=offset, weight=weight, mask=kwargs['mask'], bias=bias)
        print('  deform boundary:', {k: dtype(v) for k, v in captured.items()}, flush=True)
        y = real_deform(x, offset, weight, bias=bias, **kwargs)
        print('  deform output:', dtype(y), flush=True)
        captured['kwargs'] = kwargs
        return y

    cm = torch.cuda.amp.autocast(dtype=torch.float16) if amp else contextlib.nullcontext()
    print('AMP device={} autocast={} upstream={}'.format(device, amp, upstream), flush=True)
    with patch.object(fadc, 'deform_conv2d', record), patch.object(fadc.F, 'softplus', record_softplus), cm:
        features = upstream_conv(x) if upstream else x
        print('  FADC input:', dtype(features), flush=True)
        y = layer(features)
        loss = y.float().square().mean()
    loss.backward()
    grads = {'input': x.grad, 'base.weight': layer.weight.grad,
             'offset.weight': layer.conv_offset.weight.grad,
             'mask.weight': layer.conv_mask.weight.grad}
    if upstream:
        grads['upstream.weight'] = upstream_conv.weight.grad
    print('  loss={:.9g} grads(dtype,absmax)={}'.format(
        loss.item(), {k: (dtype(v), peak(v)) for k, v in grads.items()}), flush=True)
    assert torch.isfinite(y).all() and all(torch.isfinite(v).all() and peak(v) > 0
                                          for v in grads.values())
    assert all(dtype(captured[key]) == ('float16' if amp and key == 'mask'
                                        or upstream and key == 'x' else 'float32')
               for key in ('x', 'offset', 'weight', 'mask'))
    assert dtype(y) == ('float16' if upstream else 'float32')
    if amp:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            kw = dict(captured['kwargs'])
            try:
                real_deform(captured['x'], captured['offset'], captured['weight'],
                            bias=captured['bias'], **kw)
            except RuntimeError as exc:
                print('  same mixed inputs, autocast disabled:', str(exc).splitlines()[0], flush=True)
            else:
                print('  same mixed inputs, autocast disabled: succeeded', flush=True)
            kw['mask'] = kw['mask'].float()
            control = real_deform(captured['x'].float(), captured['offset'].float(),
                                  captured['weight'].float(), bias=None, **kw).reshape_as(y).to(y.dtype)
            print('  native all-FP32 and recast control maxdiff={:.9g}'.format(peak(control - y)))
            assert torch.equal(control, y)


def probe_gates():
    torch.manual_seed(SEED)
    original = nn.Conv2d(4, 6, 3, padding=1, bias=False)
    layer = fadc.FrequencyAdaptiveConv2d(original, 2)
    x = torch.randn(2, 4, 7, 11)
    target = torch.randn(2, 6, 7, 11)
    neutral = F.conv2d(x, original.weight, padding=2, dilation=2)
    err = peak(neutral - layer(x))
    print('FADC neutral vs fixed conv, absmax={:.9g}'.format(err))
    torch.testing.assert_close(neutral, layer(x), rtol=1e-5, atol=1e-6)
    lr, wd = 0.01, 5e-4
    optimizer = torch.optim.SGD(layer.parameters(), lr=lr, weight_decay=wd)
    keys = ('project.weight', 'norm.weight', 'norm.bias', 'gates.weight')
    params = {side + '.' + name: p for side in ('low_gates', 'high_gates')
              for name, p in getattr(layer, side).named_parameters() if name in keys}
    for step in (1, 2):
        optimizer.zero_grad()
        loss = F.mse_loss(layer(x), target)
        loss.backward()
        print('FADC gradient backward {} loss={:.9g}'.format(step, loss.item()))
        before = {name: p.detach().clone() for name, p in params.items()}
        if step == 1:
            optimizer.step()
        for name, p in params.items():
            g = peak(p.grad)
            if step == 1:
                delta = peak(p - before[name])
                pure_error = peak(p - before[name].add(before[name], alpha=-lr * wd))
                print('  {} grad={:.9g} wd*p0={:.9g} delta={:.9g} decay_only_error={:.9g}'.format(
                    name, g, peak(wd * before[name]), delta, pure_error))
                if not name.endswith('.gates.weight'):
                    assert g == pure_error == 0
                else:
                    assert g > 0 and pure_error > 0
            else:
                print('  {} grad={:.9g}'.format(name, g))
                assert g > 0


def probe_rng():
    torch.manual_seed(SEED)
    model = get_pred_model('pidnet_s', 19)
    rng = torch.random.get_rng_state()
    first, second = model.layer3[1:3]
    configure_i_atrous(model.layer3, mode='fadc', block_index=1, rate=2)
    rng1 = torch.random.get_rng_state()
    configure_i_atrous(model.layer3, mode='fadc', block_index=2, rate=2)
    rng2 = torch.random.get_rng_state()
    torch.rand(1)
    third = fadc.FrequencyAdaptiveConv2d(nn.Conv2d(128, 128, 3, padding=1), 2)
    print('FADC RNG: same stage3 block shape={} caller RNG unchanged={}/{} base_weight_equal={}'.format(
        tuple(first.conv2.weight.shape), torch.equal(rng, rng1), torch.equal(rng, rng2),
        torch.equal(first.conv2.weight, second.conv2.weight)))
    assert torch.equal(rng, rng1) and torch.equal(rng, rng2)
    assert not torch.equal(first.conv2.weight, second.conv2.weight)
    for branch in ('low_gates', 'high_gates'):
        a, b, c = [getattr(conv, branch).project.weight for conv in
                   (first.conv2, second.conv2, third)]
        print('  {} project shape={} equal_same_rng={} equal_advanced_rng={} maxabs={:.9g}'.format(
            branch, tuple(a.shape), torch.equal(a, b), torch.equal(a, c), peak(a)))
        assert torch.equal(a, b) and not torch.equal(a, c)


def probe_pmpm():
    model = PMPM(3, 2, 3).eval()
    for ix in (1, 2, 3):
        pool = getattr(model, 'scale{}'.format(ix))[0]
        s, k = pool.stride, pool.kernel_size
        for h, w in ((2 * s + 3, 2 * s + 4), (2, 3), (1, 1)):
            batch = torch.zeros(h + w, 3, h, w)
            for row in range(h):
                batch[row, 0:2, row, 0] = 1
            for col in range(w):
                batch[h + col, 0, 0, col] = batch[h + col, 2, 0, col] = 1
            pooled = pool(batch)
            nonzero = pooled.flatten(2).amax(2) > 0
            horizontal = nonzero[:h, 1].nonzero().flatten().tolist()
            vertical = nonzero[h:, 2].nonzero().flatten().tolist()
            square_all = nonzero[:, 0].all().item()
            print('PMPM scale={} k={} s={} hw={} output={} square_all={} H_rows={} V_cols={} edge(square,H,V)={}'.format(
                ix, k, s, (h, w), tuple(pooled.shape[-2:]), square_all, horizontal,
                vertical, (pooled[0, 0, 0, 0].item(), pooled[0, 1, 0, 0].item(),
                           pooled[h, 2, 0, 0].item())))
            assert horizontal == list(range(0, h, s)) and vertical == list(range(0, w, s))
            assert square_all and tuple(pooled.shape[-2:]) == ((h - 1)//s + 1, (w - 1)//s + 1)


def fresh(seed, **kwargs):
    torch.manual_seed(seed)
    return get_pred_model('pidnet_s', 19, **kwargs).eval()


def load_attempt(label, call):
    try:
        print('{} result={}'.format(label, call()))
    except RuntimeError as exc:
        print('{} RuntimeError={}'.format(label, ' '.join(str(exc).split())))
        return False
    return True


@torch.no_grad()
def probe_checkpoints():
    a, b = (1, 2, 3, 4), (2, 4, 6, 8)
    key, weight = 'spp._dense_rates', 'spp.layers.0.5.weight'
    kwargs = dict(context='denseaspp')
    src = fresh(SEED, dense_rates=a, **kwargs)
    raw = src.state_dict()
    wrapped = {'state_dict': {'module.model.' + k: v for k, v in raw.items()}}
    x = torch.randn(1, 512, 17, 21)
    out_a = src.spp(x)
    print('Dense src rates={} tensor_count={}'.format(a, len(raw)))
    outputs = {}
    for case, rates, expected in (('helper_strict_cross', b, False),
                                  ('helper_partial_cross', b, True),
                                  ('native_strict_cross', b, True),
                                  ('helper_strict_same', a, True),
                                  ('native_strict_same', a, True)):
        model = fresh(SEED + 1, dense_rates=rates, **kwargs)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        if case.startswith('native'):
            call = lambda: model.load_state_dict(raw, strict=True)
        else:
            call = lambda: load_model_checkpoint(model, wrapped, strict='partial' not in case,
                                                 prediction=True)
        assert load_attempt(case, call) == expected
        state = model.state_dict()
        metadata = tuple(model.spp._dense_rates.tolist())
        dilation = tuple(layer[-1].dilation[0] for layer in model.spp.layers)
        output = outputs[case] = model.spp(x)
        changed = sum(not torch.equal(v, before[k]) for k, v in state.items())
        all_except_rates = all(torch.equal(v, raw[k]) for k, v in state.items() if k != key)
        print('  metadata={} dilation={} changed={} dense_weight_from_src={} all_other_tensors_from_src={} maxdiff_to_src={:.9g}'.format(
            metadata, dilation, changed, torch.equal(state[weight], raw[weight]), all_except_rates,
            peak(output - out_a)))
        assert dilation == rates
        if not expected:
            assert not changed
        else:
            assert all_except_rates and torch.equal(state[weight], raw[weight])
            assert metadata == (b if 'partial' in case else a)
            assert (peak(output - out_a) == 0) == (rates == a)
    assert torch.equal(outputs['helper_partial_cross'], outputs['native_strict_cross'])
    print('Dense cross native == partial output (same actual dilation/weights): True')

    for mode in ('fixed', 'fadc'):
        kwargs = dict(atrous_mode=mode, atrous_stage=3, atrous_block=1)
        src = fresh(SEED, atrous_rate=2, **kwargs).state_dict()
        key = 'layer3.1.conv2.weight'
        for case in ('native_strict', 'helper_strict', 'helper_partial'):
            target = fresh(SEED + 1, atrous_rate=4, **kwargs)
            before = {k: v.clone() for k, v in target.state_dict().items()}
            if case == 'native_strict':
                call = lambda: target.load_state_dict(src, strict=True)
            else:
                call = lambda: load_model_checkpoint(target, src, strict=case == 'helper_strict')
            assert load_attempt('I-ATROUS {} {}'.format(mode, case), call) == (case == 'helper_partial')
            state = target.state_dict()
            changed = sum(not torch.equal(v, before[k]) for k, v in state.items())
            copied = torch.equal(state[key], src[key])
            conv = target.layer3[1].conv2
            rate = conv.dilation[0] if mode == 'fixed' else conv.nominal_rate
            print('  changed={} atrous_weight_from_src={} target_operator_rate={}'.format(
                changed, copied, rate))
            assert (changed > 0) == copied == (case != 'helper_strict') and rate == 4


@torch.no_grad()
def probe_baseline():
    old = types.ModuleType('models._pidnet_audit_head')
    old.__package__ = 'models'
    code = subprocess.check_output(['git', 'show', '4c158cf:models/pidnet.py'])
    exec(compile(code, '4c158cf:models/pidnet.py', 'exec'), old.__dict__)
    x = torch.randn(1, 3, 64, 96)
    for size in ('s', 'm', 'l'):
        torch.manual_seed(SEED)
        legacy = old.get_pred_model('pidnet_' + size, 19).eval()
        torch.manual_seed(SEED)
        current = get_pred_model('pidnet_' + size, 19).eval()
        a, b = legacy.state_dict(), current.state_dict()
        key_equal = list(a) == list(b)
        tensor_equal = key_equal and all(torch.equal(v, b[k]) for k, v in a.items())
        output_equal = torch.equal(legacy(x), current(x))
        print('Default HEAD PIDNet-{} keys={} state={} output={}'.format(
            size, key_equal, tensor_equal, output_equal))
        assert key_equal and tensor_equal and output_equal


if __name__ == '__main__':
    print('Python={} torch={} torchvision={} torch.cuda={}'.format(
        sys.version.split()[0], torch.__version__, torchvision.__version__, torch.version.cuda))
    op = 'torchvision::deform_conv2d'
    print('Dispatcher registrations for {}:'.format(op))
    for line in torch._C._dispatch_dump(op).splitlines():
        if line.startswith(('CPU:', 'CUDA:', 'Autocast')):
            print('  ' + line)
    if torch.cuda.is_available():
        print('GPU={}'.format(torch.cuda.get_device_name(0)))
        probe_amp('cuda')
        probe_amp('cuda', amp=True)
        probe_amp('cuda', amp=True, upstream=True)
        print('CUDA PROBES COMPLETE', flush=True)
    else:
        print('CUDA unavailable: torch.cuda.is_available() == False', flush=True)
    probe_amp('cpu')
    probe_gates()
    probe_rng()
    probe_pmpm()
    probe_checkpoints()
    probe_baseline()
    print('MODEL PROBES COMPLETE', flush=True)
