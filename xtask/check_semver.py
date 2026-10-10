#!/usr/bin/env python3
"""Run ordinary semver checks plus Magetypes' concrete-impl feature check.

Keep each invocation's rustdoc artifacts in a fresh target directory: stale or
ambiguous cache entries must never turn a missing comparison into a pass.
Without --baseline-version, cargo-semver-checks selects the published baseline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

from check_target_features import compare

ROOT = Path(__file__).resolve().parents[1]


def only(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {description}, found {paths}')
    return paths[0]


def baseline_artifact(directory):
    return only((p for p in (directory / 'semver-checks/cache').glob('magetypes-*.json')
                 if not p.name.endswith('.metadata.json')), 'baseline rustdoc')


def check_artifacts(directory, version, baseline_version=None):
    cache = directory / 'semver-checks'
    baseline = baseline_artifact(directory)
    current = directory / 'doc/magetypes.json'
    before, after = (json.loads(p.read_text()) for p in (baseline, current))
    for doc in (before, after):
        if doc['index'][str(doc['root'])]['name'] != 'magetypes':
            raise ValueError('Expected Magetypes rustdoc')
        if doc['format_version'] != 60:
            raise ValueError('Review the precise checker for this new rustdoc format')
    if before['target'] != after['target']:
        raise ValueError('Rustdoc target contexts differ')
    if after['crate_version'] != version:
        raise ValueError('Current rustdoc version does not match the workspace')
    if baseline_version and before['crate_version'] != baseline_version:
        raise ValueError('Baseline rustdoc version does not match the requested version')

    # Registry source is removed after caching. Its Cargo metadata records the
    # resolved features; resolve the surviving current placeholder the same way.
    metadata = json.loads(baseline.with_suffix('.metadata.json').read_text())
    manifest = only(cache.glob('local-magetypes-*/Cargo.toml'), 'current placeholder manifest')
    current_metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--format-version', '1', '--manifest-path', str(manifest)],
        cwd=ROOT))
    selected = [selected_features(m) for m in (metadata, current_metadata)]
    if selected[0] != selected[1] or 'avx512' not in selected[0]:
        raise ValueError('Feature selections must match and include avx512')

    failures, count, featured = compare(before, after)
    print(f'Precise target-feature check: {before["crate_version"]} -> '
          f'{after["crate_version"]}: {count} safe methods, '
          f'{featured} with target features, {len(failures)} failures', flush=True)
    for failure in failures:
        print(failure, flush=True)
    if not featured:
        raise ValueError('Expected target-feature methods; refusing an empty feature check')
    return bool(failures)


def selected_features(metadata):
    package = only((p for p in metadata['packages'] if p['name'] == 'magetypes'),
                   'Magetypes package')
    node = only((n for n in metadata['resolve']['nodes'] if n['id'] == package['id']),
                'Magetypes feature resolution')
    return sorted(node['features'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', choices=['archmage', 'magetypes'])
    parser.add_argument('--baseline-version')
    args = parser.parse_args()
    output = ROOT / 'target' / 'semver-guard'
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f'{args.package}-', dir=output))
    print(f'Semver artifacts: {directory}', flush=True)
    command = ['cargo', 'semver-checks', 'check-release', '-p', args.package]
    if args.baseline_version:
        command += ['--baseline-version', args.baseline_version]
    environment = dict(os.environ, CARGO_TARGET_DIR=str(directory))
    subprocess.run(command, cwd=ROOT, check=True, env=environment)
    if args.package == 'magetypes':
        # CARGO_TARGET_DIR causes baseline generation to overwrite doc/magetypes.json.
        # Rebuild the current JSON against the now-frozen baseline. This second
        # pass retains manifest lint configuration and cannot rebuild the baseline.
        subprocess.run(['cargo', 'semver-checks', 'check-release', '-p', 'magetypes',
                        '--baseline-rustdoc', str(baseline_artifact(directory))],
                       cwd=ROOT, check=True, env=environment)
        version = tomllib.loads((ROOT / 'Cargo.toml').read_text())['workspace']['package']['version']
        return check_artifacts(directory, version, args.baseline_version)
    return False


if __name__ == '__main__':
    raise SystemExit(main())
