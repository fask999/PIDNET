"""Exercise the real Cityscapes eval and config custom programs with a synthetic PNG."""
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


ROOT = Path.cwd()
EXP = ROOT / 'test/20261112-pidnet-l-ablation'
WEIGHT = ROOT / ('test/20261111-pidnet-p0p1-fixes/output/'
                 'pidnet-l-cityscapes-zero-stem-bias.pth.tar')
EXPECTED = '9026f05938c07b8a81ee6da033dbbff439c0e4d9a1f0e58e62a9c5c5fc3699f1'


def run_cli(script, args, *, expect=None):
    argv = [sys.executable, '-B', script] + [str(arg) for arg in args]
    p = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding='utf-8', timeout=50)
    if expect is not None:
        assert p.returncode != 0 and expect in p.stdout, (argv, p.returncode, p.stdout[-4000:])
        return
    assert p.returncode == 0, (argv, p.returncode, p.stdout[-4000:])
    return p.stdout


def main(cfg):
    h = sha256()
    with WEIGHT.open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            h.update(block)
    assert h.hexdigest() == EXPECTED, h.hexdigest()
    print('Existing converted checkpoint SHA-256 verified: ' + EXPECTED)
    with tempfile.TemporaryDirectory(dir=EXP, prefix='cli-synthetic-') as name:
        fixture = Path(name)
        data = fixture / 'data'
        images = data / 'cityscapes' / 'images'
        labels = data / 'cityscapes' / 'labels'
        lists = data / 'list' / 'cityscapes'
        for d in (images, labels, lists):
            d.mkdir(parents=True)
        rgb = np.random.RandomState(304).randint(0, 256, (64, 96, 3), dtype=np.uint8)
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[:, :48], mask[:, 48:] = 7, 8
        Image.fromarray(rgb).save(str(images / 'fixture.png'))
        Image.fromarray(mask).save(str(labels / 'fixture.png'))
        (lists / 'val.lst').write_text('images/fixture.png labels/fixture.png\n')
        eval_args = ['--cfg', cfg, 'DATASET.ROOT', str(data) + '/',
                     'TEST.IMAGE_SIZE', '[96, 64]', 'TEST.BASE_SIZE', '96',
                     'TEST.MODEL_FILE', WEIGHT, 'OUTPUT_DIR', fixture / 'output',
                     'LOG_DIR', fixture / 'logs']
        ev = run_cli('tools/eval.py', eval_args)
        assert 'MeanIU:' in ev and 'Done' in ev, ev
        count = re.search(r'loaded (\d+) model tensors', ev)
        assert count and int(count.group(1)) == 545, ev
        print('tools/eval.py: actual CUDA default L loaded 545 tensors and evaluated one '
              'synthetic validation PNG (fixture metrics omitted).')
        run_cli('tools/eval.py', eval_args + ['ABLATION.I_ATROUS.MODE', 'fixed'],
                expect='_i_atrous_signature_fixed_s5_b1_r2')
        print('tools/eval.py: strict load rejects default weights for fixed rate 2.')
        out = fixture / 'preview'
        custom_args = ['--cfg', cfg, '--r', images, '--output-dir', out]
        pred = run_cli('tools/custom.py', custom_args + ['TEST.MODEL_FILE', WEIGHT])
        count = re.search(r'Loaded (\d+) model tensors', pred)
        assert count and 0 < int(count.group(1)) < 545, pred
        with Image.open(str(out / 'fixture.png')) as im:
            assert im.size == (96, 64) and im.mode == 'RGB', (im.size, im.mode)
        print('tools/custom.py: actual CUDA configured L loaded {} inference tensors '
              'via TEST.MODEL_FILE and wrote 96x64 RGB PNG.'.format(count.group(1)))
        run_cli('tools/custom.py', custom_args, expect='TEST.MODEL_FILE')
        print('tools/custom.py: protected L config rejects omission of both weight paths.')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
