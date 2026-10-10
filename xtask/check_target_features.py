#!/usr/bin/env python3
"""Compare safe inherent methods by concrete impl, including generic arguments.

Required companion to cargo-semver-checks. Its target-feature lint in 0.51.0
matches methods without their impl's type arguments, confusing V4 and V4x.
For Magetypes only, this replaces that lint; all other lints remain enabled.
Run check_semver.py to perform both checks on the same fresh rustdoc artifacts.
Inputs must be rustdoc JSON built for the same target and Cargo feature set.
"""
import argparse
import json
from pathlib import Path


def canonical(value, paths):
    """Resolve rustdoc-local IDs to stable paths; retain all type arguments."""
    if isinstance(value, list):
        return [canonical(item, paths) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: canonical(item, paths) for key, item in value.items()
              if key not in ('id', 'path') or 'id' not in value}
    if 'id' in value:
        # Missing IDs fail closed rather than silently dropping type identity.
        result['path'] = paths[str(value['id'])]['path']
    return result


def safe_methods(document):
    methods = {}
    for item in document['index'].values():
        impl = item['inner'].get('impl')
        if impl is None or impl['trait'] is not None:
            continue
        identity = canonical([impl['for'], impl['generics']], document['paths'])
        for method_id in impl['items']:
            method = document['index'][str(method_id)]
            function = method['inner'].get('function')
            if (method['visibility'] != 'public' or function is None
                    or function['header']['is_unsafe']):
                continue
            key = json.dumps([*identity, method['name']], sort_keys=True)
            if key in methods:
                raise ValueError(f'Duplicate method identity: {key}')
            features = {feature for attr in method['attrs']
                        for feature in attr.get('target_feature', {}).get('enable', [])}
            methods[key] = features
    if not methods:
        raise ValueError('No public safe inherent methods; refusing an empty comparison')
    return methods


def compare(baseline, current):
    before, after = safe_methods(baseline), safe_methods(current)
    failures = []
    for key, features in before.items():
        if key not in after:
            failures.append(f'Safe method missing or impl changed: {key}')
        elif added := after[key] - features:
            failures.append(f'Added target features {sorted(added)}: {key}')
    return failures, len(before), sum(bool(features) for features in before.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    failures, count, featured = compare(
        json.loads(args.baseline.read_text()), json.loads(args.current.read_text()))
    for failure in failures:
        print(failure)
    print(f'Compared {count} safe methods ({featured} with target features); '
          f'{len(failures)} failures')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
