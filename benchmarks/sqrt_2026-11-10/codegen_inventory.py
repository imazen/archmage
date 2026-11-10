#!/usr/bin/env python3
"""Inventory every changed probe without changing or bypassing the CI gate.

Usage: python3 -B codegen_inventory.py CODEGEN_RUN_DIRECTORY OUTPUT_JSON
The run directory is printed by `just codegen-diff BASE HEAD`.
"""
import collections
import importlib.util
import json
from pathlib import Path
import sys


def main():
    run, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('codegen', root / 'xtask/codegen.py')
    codegen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codegen)
    result = {}
    for target in ['x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu',
                   'wasm32-unknown-unknown']:
        before, after = [codegen.parse((run / f'{side}-{target}.s').read_text())
                         for side in ['before', 'after']]
        names = {name for name in before if not name.startswith('_')}
        assert names == {name for name in after if not name.startswith('_')}
        changed = []
        for name in sorted(names):
            old, new = codegen.expand(name, before), codegen.expand(name, after)
            if old == new:
                continue
            old_instructions = [line for line in old if not line.startswith('LABEL ')]
            new_instructions = [line for line in new if not line.startswith('LABEL ')]
            changed.append({
                'name': name, 'before_entries': len(old_instructions),
                'after_entries': len(new_instructions),
                'new_ops': dict(collections.Counter(map(codegen.opcode, new_instructions))),
                'before': old, 'after': new,
            })
        result[target] = {'total': len(names), 'unchanged': len(names) - len(changed),
                          'changes': changed}
        print(f'{target}: {len(changed)}/{len(names)} changed')
        for item in changed:
            print(f"  {item['name']}: {item['before_entries']} -> {item['after_entries']}")
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
