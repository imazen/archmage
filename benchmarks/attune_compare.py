#!/usr/bin/env python3
"""Compare pinned revisions with alternating cold and consumer-only checks.

Run under run-heavy. Archives, full logs, source revisions, lockfile hash and
/usr/bin/time max RSS are retained; existing output directories are rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = Path('tests/downstream-compat/compile-cost')
    metadata = {
        'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
        'cargo': subprocess.check_output(['cargo', '-V'], text=True).strip(),
        'command': __import__('sys').argv,
        'revisions': {}, 'runs': [],
    }
    for label in ('baseline', 'candidate'):
        revision = subprocess.check_output(
            ['git', 'rev-parse', getattr(args, label)], cwd=root, text=True).strip()
        metadata['revisions'][label] = revision
        archive = out / f'{label}.tar'
        with archive.open('wb') as output:
            subprocess.run(['git', 'archive', revision], cwd=root, stdout=output, check=True)
        source = out / label
        source.mkdir()
        with tarfile.open(archive) as capture:
            capture.extractall(source, filter='data')
    lock = (out / 'candidate' / fixture / 'Cargo.lock').read_bytes()
    (out / 'baseline' / fixture / 'Cargo.lock').write_bytes(lock)
    metadata['fixture_lock_sha256'] = hashlib.sha256(lock).hexdigest()
    baseline_input = (out / 'baseline' / fixture / 'src/lib.rs').read_bytes()
    assert baseline_input == (out / 'candidate' / fixture / 'src/lib.rs').read_bytes()
    metadata['fixture_source_sha256'] = hashlib.sha256(baseline_input).hexdigest()
    for feature_label, features in [('macros', ''), ('all', 'avx512,use_magetypes')]:
        for pair in range(args.pairs):
            order = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
            for label in order:
                source = out / label
                target = out / f'{feature_label}-{pair}-{label}'
                env = dict(os.environ, CARGO_TARGET_DIR=str(target))
                command = ['cargo', 'check', '--locked', '--manifest-path', str(source / fixture / 'Cargo.toml')]
                if features:
                    command += ['--features', features]
                for stage in ('cold', 'consumer'):
                    if stage == 'consumer':
                        # Re-expand/recheck only the unchanged consuming crate.
                        os.utime(source / fixture / 'src/lib.rs', None)
                    stem = f'{feature_label}-{pair}-{label}-{stage}'
                    rss = out / f'{stem}.time'
                    start = time.perf_counter()
                    with (out / f'{stem}.log').open('w') as log:
                        result = subprocess.run(['/usr/bin/time', '-v', '-o', str(rss), *command],
                                                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                    elapsed = time.perf_counter() - start
                    maximum = next(line.split(':', 1)[1].strip() for line in rss.read_text().splitlines()
                                   if 'Maximum resident set size' in line)
                    row = dict(feature=feature_label, pair=pair, revision=label, stage=stage,
                               seconds=elapsed, max_rss_kib=int(maximum), exit_code=result.returncode,
                               command=command)
                    metadata['runs'].append(row)
                    (out / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
                    print(json.dumps(row), flush=True)
                    if result.returncode:
                        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
