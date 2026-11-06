#!/usr/bin/env python3
"""Inspect durable matrix state together with actual recorded processes/chunks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
a=p.parse_args()
records=[]
for progress in sorted(a.root.glob('*/progress.json')):
    state=json.loads(progress.read_text())
    item=dict(group=progress.parent.name,state=state)
    for key in ['pid','child_pid']:
        if key in state:
            proc=Path(f'/proc/{state[key]}')
            item[key+'_live']=proc.exists()
            if proc.exists():
                item[key+'_cgroup']=(proc/'cgroup').read_text().strip()
                children=proc/'task'/str(state[key])/'children'
                item[key+'_children']=children.read_text().split() if children.exists() else []
    if state.get('cell'):
        cell=progress.parent/state['cell']
        item['current_chunks']=len(list(cell.glob('chunk-*.tsv')))
        failure=cell/'FAILURE.txt'
        if failure.exists():item['failure']=failure.read_text()
    item['summaries']={path.parent.name:path.read_text() for path in sorted(progress.parent.glob('*/summary.txt'))}
    records.append(item)
print(json.dumps(dict(checked=datetime.now(timezone.utc).isoformat(),root=str(a.root),groups=records),indent=2))
