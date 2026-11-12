"""Compare this task's source and protected assets with the opening snapshot."""

import difflib
import hashlib
import os
from pathlib import Path
import subprocess


EXP = Path(__file__).resolve().parents[1]
LOG = EXP / 'logs'
ROOT = EXP.parents[1]
EXPECTED = {
    'M': ('models/context.py', 'models/pidnet.py', 'tools/train.py', 'utils/ablation_config.py',
          'utils/checkpoint.py', 'utils/criterion.py'),
    'A': ('tools/convert_pidnet_l_stem_bias.py', 'utils/training_checkpoint.py'),
}
GROUPS = {
    'protected': ('configs', 'data', 'pretrained_models', 'test/20261111-external-review-audit',
                  'test/20261111-pidnet-ablation-switches'),
    'source': ('models', 'utils', 'datasets', 'tools'),
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(group):
    rows = {}
    for folder in GROUPS[group]:
        for base, subdirs, names in os.walk(folder):
            for name in subdirs + names:
                path = Path(base) / name
                if path.is_symlink():
                    rows[str(path)] = 'link:' + os.readlink(path)
                elif path.is_file() and (group != 'source' or path.suffix == '.py'):
                    rows[str(path)] = digest(path)
    return rows


def read_inventory(path):
    return {path: value for value, path in (line.split('\t', 1) for line in path.read_text().splitlines())}


def write_if_changed(path, content):
    if not path.is_file() or path.read_text() != content:
        path.write_text(content)


def main():
    os.chdir(str(ROOT))
    actual = {}
    for group in GROUPS:
        before = read_inventory(LOG / (group + '-before.tsv'))
        after = inventory(group)
        write_if_changed(LOG / (group + '-after.tsv'),
            ''.join(value + '\t' + path + '\n' for path, value in sorted(after.items())))
        actual[group] = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        if group == 'protected':
            assert before == after, actual[group]
            print('{} protected path digests/symlink targets unchanged'.format(len(before)))
        else:
            assert actual[group] == sorted(EXPECTED['M'] + EXPECTED['A']), actual[group]
            assert all(path in before and path in after for path in EXPECTED['M'])
            assert all(path not in before and path in after for path in EXPECTED['A'])
            print('{} existing and {} new runtime Python files'.format(len(EXPECTED['M']), len(EXPECTED['A'])))

    chunks, stats = [], []
    for status, files in EXPECTED.items():
        for name in files:
            source = LOG / 'start-source' / name
            if status == 'M':
                assert digest(source) == before[name]
                check = subprocess.run(['git', 'diff', '--no-index', '--check', '--', str(source), name],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert check.returncode in (0, 1) and not check.stdout, check.stdout
                old = source.read_text().splitlines(keepends=True)
            else:
                old = []
            current = Path(name).read_text().splitlines(keepends=True)
            delta = list(difflib.unified_diff(old, current, 'start/' + name, 'current/' + name, n=0))
            if status == 'M':
                chunks.extend(delta)
            added = sum(line.startswith('+') and not line.startswith('+++') for line in delta)
            removed = sum(line.startswith('-') and not line.startswith('---') for line in delta)
            stats.append('{}\t+{}\t-{}\t{}\n'.format(status, added, removed, name))
    write_if_changed(LOG / 'review-delta.diff', ''.join(chunks))
    write_if_changed(LOG / 'source-delta-stat.txt', ''.join(stats))
    print('incremental diff check and scope comparison passed')


if __name__ == '__main__':
    main()
