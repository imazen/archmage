#!/usr/bin/env python3
"""Audit prelude names, including secondary re-export paths in API snapshots."""
import argparse
import json
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', default='1a0ea59')
p.add_argument('--expanded', default='7d557b90')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

def names(text):
    result = set()
    for line in text.splitlines():
        match = re.search(r'pub (?:type|struct|enum|trait|mod|use) ([\w:]+)', line)
        if not match:
            continue
        path = match[1]
        aliases = re.search(r'\[also: (.*?)\]', line)
        if path.startswith('prelude::') and path.count('::') == 1:
            result.add(path.split('::')[-1])
        if aliases and 'prelude' in aliases[1].split(', '):
            result.add(path.split('::')[-1])
    return result

report = {'baseline': a.baseline, 'expanded': a.expanded, 'targets': {},
          'scope': 'Prelude top-level names; not a complete semver proof or a count of unique inherent methods'}
for target in ['x86_64', 'aarch64', 'wasm32']:
    path = f'docs/public-api/{target}/magetypes.txt'
    read = lambda revision: subprocess.check_output(['git', 'show', f'{revision}:{path}'], text=True)
    original, expanded, current = names(read(a.baseline)), names(read(a.expanded)), names(Path(path).read_text())
    report['targets'][target] = {'original_names': sorted(original),
        'accidental_additions': sorted(expanded-original),
        'remaining_additions': sorted(current-original), 'missing_original_names': sorted(original-current)}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
assert all(not t['missing_original_names'] and not t['remaining_additions'] for t in report['targets'].values()), report
print(json.dumps(report, indent=2))
