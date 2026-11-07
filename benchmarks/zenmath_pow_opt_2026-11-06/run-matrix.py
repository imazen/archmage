#!/usr/bin/env python3
"""Run only retained immutable executables, preserving per-stage logs/status."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

snapshot = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=False)
identity = json.loads((snapshot / 'identity.json').read_text())
programs = {Path(row['path']).name: snapshot / row['archived_as'] for row in identity['binaries']}
stages = [('pow', 'timing', 60), ('transfer', 'transfer', 48), ('feedback', 'feedback', 4)]
status = dict(source_identity=str(snapshot / 'identity.json'), source_manifest_sha256=identity['source_manifest_sha256'], stages=[])
for stage, binary, groups in stages:
    entry = dict(stage=stage, expected_groups=groups, start=datetime.now(timezone.utc).isoformat(), state='starting')
    status['stages'].append(entry)
    with (root / (stage + '.log')).open('w') as log:
        child = subprocess.Popen([str(programs[binary]), str(root / (stage + '.json'))], stdout=log, stderr=subprocess.STDOUT)
        entry.update(pid=child.pid, state='running')
        (root / 'progress.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(entry), flush=True)
        result = child.wait()
    entry.update(returncode=result, state='failed' if result else 'complete', end=datetime.now(timezone.utc).isoformat())
    (root / 'progress.json').write_text(json.dumps(status, indent=2) + '\n')
    assert result == 0, entry
    suite = json.loads((root / (stage + '.json')).read_text())
    assert not suite['unreliable'] and not suite['incomplete'], entry
    assert len(suite['comparisons']) == groups
    entry['dirty_groups'] = [g['group_name'] for g in suite['comparisons'] if any(r['gate_clean'] is not True for r in g['samples'])]
    (root / 'progress.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(entry), flush=True)
