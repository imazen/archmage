#!/usr/bin/env python3
"""Reproducible local-on-admitted-host driver; no SSH, build or admission implicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sources():
    files = set()
    for directory in ['src', 'archmage-macros/src', 'zenmath/src', 'benchmarks/zenmath_digests_2026-11-06/src']:
        files.update(p for p in (ROOT / directory).rglob('*') if p.is_file())
    for name in ['Cargo.toml', 'Cargo.lock', 'token-registry.toml', 'archmage-macros/Cargo.toml', 'zenmath/Cargo.toml', 'benchmarks/zenmath_digests_2026-11-06/Cargo.toml', 'benchmarks/zenmath_digests_2026-11-06/Cargo.lock', 'benchmarks/zenmath_digests_2026-11-06/build.rs', 'benchmarks/zenmath_digests_2026-11-06/drive.py']:
        files.add(ROOT / name)
    return ''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in sorted(files))

def prepare(args):
    if args.out.exists():
        raise ValueError(f'build identity directory already exists: {args.out}; use a new directory')
    manifest = sources()
    compiler_env = dict(os.environ, RUSTUP_TOOLCHAIN='1.99.0')
    rustc = subprocess.check_output(['rustup', 'run', '1.99.0', 'rustc', '-vV'], text=True, env=compiler_env)
    import shlex
    command = shlex.split(args.runner) + [str(args.binary), '--build-info']
    embedded = dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())
    source_hash = hashlib.sha256(manifest.encode()).hexdigest()
    if embedded['source_sha256'] != source_hash:
        raise ValueError('binary embedded source identity differs from current sources')
    if embedded['rustc'] != rustc.splitlines()[0]:
        raise ValueError('binary compiler differs from pinned rustc1.99.0')
    target = args.target or embedded['target']
    if target != embedded['target']:
        raise ValueError('requested target differs from binary embedded target')
    identity = dict(source_revision=args.revision, source_sha256=source_hash, binary_sha256=sha(args.binary), rustc=embedded['rustc'], target=target, rounding='nearest-even;gradual-underflow', rustflags=os.environ.get('RUSTFLAGS', '<unset>'), encoded_rustflags=os.environ.get('CARGO_ENCODED_RUSTFLAGS', '<unset>'), binary=str(args.binary.resolve()), runner=args.runner)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'sources.sha256').write_text(manifest)
    (args.out / 'build.json').write_text(json.dumps(identity, indent=2) + '\n')
    (args.out / 'rustc.txt').write_text(rustc)
    print(json.dumps(identity, indent=2))

def run(args):
    identity = json.loads((args.build / 'build.json').read_text())
    if sha(Path(identity['binary'])) != identity['binary_sha256']:
        raise ValueError('binary changed after prepare')
    if hashlib.sha256(sources().encode()).hexdigest() != identity['source_sha256']:
        raise ValueError('source changed after prepare; create separate build/run')
    args.out.mkdir(parents=True, exist_ok=True)
    command = [identity['binary'], '--op', args.op, '--backend', args.backend, '--threads', str(args.threads), '--start', str(args.start), '--count', str(args.count), '--chunk', str(args.chunk), '--out', str(args.out), '--identity', str(args.out / 'identity.txt')]
    if args.compare:
        command.append('--compare')
    text = ''.join(f'{k}={v}\n' for k, v in identity.items()) + 'command=' + json.dumps(command) + '\n'
    identity_path = args.out / 'identity.txt'
    if identity_path.exists() and identity_path.read_text() != text:
        raise ValueError('existing run identity mismatch')
    identity_path.write_text(text)
    if args.resume:
        command.append('--resume')
    if identity['runner']:
        import shlex
        command = shlex.split(identity['runner']) + command
    with (args.out / 'invocations.jsonl').open('a') as stream:
        stream.write(json.dumps(command) + '\n')
    import fcntl
    with (args.out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            summary=args.out/'summary.txt'
            if summary.exists():
                import time
                summary.rename(args.out/f'summary.txt.{time.time_ns()}.bak')
            pending=args.out/'FAILURE.pending'
            pending.write_text(f'child returncode={error.returncode}; signal={-error.returncode if error.returncode < 0 else "none"}\ncommand={json.dumps(command)}\n')
            pending.replace(args.out/'FAILURE.txt')
            raise

def compare(args):
    records = []
    for path in args.runs:
        subprocess.run([str(args.auditor), '--audit', str(path)], check=True)
        if (path / 'FAILURE.txt').exists():
            raise ValueError(f'failed run {path}')
        meta = (path / 'metadata.txt').read_text()
        config = dict(line.split('=', 1) for line in meta.splitlines()[1:] if '=' in line)
        summary = dict(line.split('=', 1) for line in (path / 'summary.txt').read_text().splitlines())
        rows = [line.split('\t') for line in (path / 'chunks.tsv').read_text().splitlines()]
        records.append((path, config, summary, rows))
    first = records[0]
    for path, config, summary, rows in records:
        if (path / 'metadata.txt').read_text().splitlines()[0] != (first[0] / 'metadata.txt').read_text().splitlines()[0]:
            raise ValueError(f'incompatible hash scheme {path}')
        for field in ['op', 'start', 'count', 'chunk', 'source_revision', 'source_sha256']:
            if config[field] != first[1][field]:
                raise ValueError(f'incompatible source/domain {path}: {field}')
        if summary['count'] != config['count']:
            raise ValueError(f'count mismatch {path}')
        if len(rows) != len(first[3]):
            raise ValueError(f'chunk count mismatch {path}')
        mismatches = [i for i, (a, b) in enumerate(zip(first[3], rows)) if a[:3] != b[:3] or a[4] != b[4]]
        if mismatches:
            raise ValueError(f'canonical mismatch {path}: chunks {mismatches}; rerun with --compare and narrowed domains in NEW directories')
        if summary['canonical_aggregate_of_chunks'] != first[2]['canonical_aggregate_of_chunks']:
            raise ValueError(f'aggregate mismatch {path}')
        print(f'MATCH {path}: {summary["status"]} count={summary["count"]} canonical={summary["canonical_aggregate_of_chunks"]}; raw_equal={summary["raw_aggregate_of_chunks"] == first[2]["raw_aggregate_of_chunks"]}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    p = subs.add_parser('stamp')
    p.set_defaults(func=lambda _: print(hashlib.sha256(sources().encode()).hexdigest()))
    p = subs.add_parser('prepare')
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--target')
    p.add_argument('--runner', default='')
    p.set_defaults(func=prepare)
    p = subs.add_parser('run')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--op', required=True)
    p.add_argument('--backend', required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=2**32)
    p.add_argument('--chunk', type=int, default=2**20)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--compare', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.set_defaults(func=run)
    p = subs.add_parser('compare')
    p.add_argument('--auditor', required=True, type=Path)
    p.add_argument('runs', nargs='+', type=Path)
    p.set_defaults(func=compare)
    try:
        args = parser.parse_args()
        args.func(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))

if __name__ == "__main__":
    main()
