#!/usr/bin/env python3
"""Read actual controller/PID/child artifacts; never infer progress from age."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('source', type=Path)
a = p.parse_args()
state = json.loads((a.root / 'progress.json').read_text())
state['observed_utc'] = datetime.now(timezone.utc).isoformat()
if state['stage'] not in ('complete', 'failed-no-packet'):
    pid = state['wrapper_pid']
    try:
        os.kill(pid, 0)
        state['wrapper_alive'] = True
        state['live_command'] = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    except ProcessLookupError:
        state['wrapper_alive'] = False
    log = (a.root / (state['stage'] + '.log')).read_text()
    processes = re.findall(r'\[zenbench\] process (\d+)/(\d+)', log)
    if processes:
        state['process'], state['required_processes'] = map(int, processes[-1])
    children = re.findall(r'\[zenbench\] child log → (.+)', log)
    if children:
        child = a.source / children[-1]
        state['child_log'] = str(child)
        with child.open() as stream:
            state['completed_groups_in_current_process'] = sum(
                bool(re.match(r'^\s+(?:wasm-)?[a-z0-9]+/(?:min|max|recip|rsqrt)/f(?:32|64)x\d+/\S+\s+50 rounds', line))
                for line in stream)
        state['required_groups_per_process'] = 66
(a.root / 'LIVE-PROGRESS.json').write_text(json.dumps(state, indent=2) + '\n')
line = json.dumps(state)
with (a.root / 'progress-history.jsonl').open('a') as stream:
    stream.write(line + '\n')
print(line)
