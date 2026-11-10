#!/usr/bin/env python3
"""Untimed complete trajectories and final-batch parity for every provider cell."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from external import cells, load_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('directory',type=Path)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
manifest=load_manifest(a.manifest)
summary={'cells':0,'provider_cells':0,'batch_last_lane_mismatches':0,'budget_bytes':512*1024*1024,
         'semantics':'streams are raw powers; chains contain normalized states and may have different inputs after divergence',
         'edges':{}}
used=0
with gzip.open(a.directory/'all-trajectories.jsonl.gz','wt') as out:
    for cell in cells():
        record=dict(cell,outputs={})
        for provider in manifest['providers']:
            if provider['lanes']>cell['width']:continue
            worker=manifest['workers'][provider['worker']]
            assert hashlib.sha256(Path(worker['binary']).read_bytes()).hexdigest()==worker['sha256']
            fields=f'{provider["method"]} {cell["width"]} {cell["size"]} {int(cell["chain"])} {cell["exponent"]}'
            request=f'T {fields} 1\nB {fields} {1048576//cell["size"]}\n'
            run=subprocess.run([worker['binary']],input=request,capture_output=True,text=True,timeout=120)
            (a.directory/f'{summary["provider_cells"]:03d}.stderr').write_text(run.stderr)
            run.check_returncode()
            trace,final=[line.split() for line in run.stdout.splitlines()]
            assert len(trace)==cell['size'] and len(final)==cell['width']
            assert trace[-cell['width']:]==final, 'single-buffer trace versus repeated batch mismatch'
            assert all(0< int(v,16)&0x7f800000 <0x7f800000 for v in trace), 'ordinary timing input produced a non-normal value'
            record['outputs'][provider['label']]=trace
            summary['provider_cells']+=1
        names=list(record['outputs'])
        for i,left in enumerate(names):
            for right in names[i+1:]:
                count=sum(x!=y for x,y in zip(record['outputs'][left],record['outputs'][right],strict=True))
                edge=left+' / '+right
                summary['edges'][edge]=summary['edges'].get(edge,0)+count
        text=json.dumps(record,separators=(',',':'))+'\n';used+=len(text.encode())
        if used>summary['budget_bytes']:raise RuntimeError('trace budget exceeded; partial preserved')
        out.write(text);summary['cells']+=1
        (a.directory/'progress.json').write_text(json.dumps({'completed_cells':summary['cells'],'provider_cells':summary['provider_cells'],'last':cell['group']})+'\n')
        print(json.dumps({'completed_cells':summary['cells'],'last':cell['group']}),flush=True)
summary['uncompressed_bytes']=used
(a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
