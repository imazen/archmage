#!/usr/bin/env python3
"""Run rustdoc on the actual website Markdown, not copies of its examples.

Usage: python3 xtask/check_docs.py [--features avx512] [page.md ...]
Scratch files stay under target/; pass CARGO_TARGET_DIR to reuse build artifacts.
Every Rust fence is tested. Syntax-only fragments must use a text fence;
compile_fail fences must fail as specified. No implicit imports are supplied.
"""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / 'docs/site/content'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='')
    parser.add_argument('--target')
    parser.add_argument('pages', nargs='*')
    args = parser.parse_args()
    pages = [ROOT / p for p in args.pages] if args.pages else [
        *sorted(CONTENT.rglob('*.md')),
        *(ROOT / p for p in ('README.md', 'README.crates.md', 'magetypes/README.md', 'magetypes/README.crates.md', 'archmage-macros/README.md')),
    ]
    scratch = ROOT / 'target/docs-check'
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / 'src').mkdir(exist_ok=True)
    manifest = '''[package]
name = "archmage-website-examples"
version = "0.0.0"
edition = "2024"
publish = false
[workspace]
[features]
default = []
avx512 = ["archmage/avx512", "magetypes/avx512"]
[dependencies]
'''
    manifest += f'archmage = {{ path = {json.dumps(str(ROOT))}, features = ["testable_dispatch"] }}\n'
    manifest += f'magetypes = {{ path = {json.dumps(str(ROOT / "magetypes"))} }}\n'
    (scratch / 'Cargo.toml').write_text(manifest)
    (scratch / 'src/lib.rs').write_text('\n'.join(
        f'#[doc = include_str!({json.dumps(str(p.resolve()))})]\npub mod page_{i} {{}}'
        for i, p in enumerate(pages)
    ))
    command = ['cargo', 'test', '--doc', '--manifest-path', str(scratch / 'Cargo.toml')]
    if args.target:
        command += ['--target', args.target]
    if args.features:
        command += ['--features', args.features]
    env = os.environ.copy()
    env.setdefault('TMPDIR', str(Path.home() / 'tmp'))
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    env.setdefault('CARGO_TARGET_DIR', str(ROOT / 'target'))
    print(f'Testing Markdown from {len(pages)} pages', flush=True)
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    if result.returncode == 0 and not re.search(r'running [1-9][0-9]* tests?', result.stdout):
        print('ERROR: rustdoc did not execute examples; refusing a vacuous pass.')
        return 1
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
