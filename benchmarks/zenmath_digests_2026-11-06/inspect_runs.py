#!/usr/bin/env python3
"""Inspect durable matrix state together with actual recorded processes/chunks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def process_identity(pid):
    proc=Path(f'/proc/{pid}')
    result=dict(live=proc.exists())
    if not result['live']:
        return result
    try:
        # stat field22: starttime in clock ticks since boot. Comm may contain spaces.
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        result['starttime_ticks']=int(stat[19])
        result['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        result['cmdline']=(proc/'cmdline').read_bytes().decode('utf-8',errors='replace').rstrip('\0').split('\0')
        result['executable']=os.readlink(proc/'exe')
        result['cgroup']=(proc/'cgroup').read_text().strip()
        children=proc/'task'/str(pid)/'children'
        result['children']=children.read_text().split() if children.exists() else []
    except (FileNotFoundError,ProcessLookupError,PermissionError) as error:
        result['inspection_race_or_permission']=str(error)
        result['live']=proc.exists()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args()
    records=[]
    for progress in sorted(a.root.glob('*/progress.json')):
        state=json.loads(progress.read_text())
        item=dict(group=progress.parent.name,state=state)
        for key in ['pid','child_pid']:
            if key in state:
                identity=process_identity(state[key])
                item[key+'_identity']=identity
                item[key+'_live']=identity['live']
        if state.get('cell'):
            cell=progress.parent/state['cell']
            item['current_chunks']=len(list(cell.glob('chunk-*.tsv')))
            failure=cell/'FAILURE.txt'
            if failure.exists():
                item['failure']=failure.read_text(errors='replace')
        item['summaries']={path.parent.name:path.read_text() for path in sorted(progress.parent.glob('*/summary.txt'))}
        records.append(item)
    print(json.dumps(dict(checked=datetime.now(timezone.utc).isoformat(),root=str(a.root),groups=records),indent=2))


if __name__=='__main__':
    main()
