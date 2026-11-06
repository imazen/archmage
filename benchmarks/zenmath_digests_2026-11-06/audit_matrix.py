#!/usr/bin/env python3
"""Read-only full-matrix audit using the frozen Zenhash auditor/comparator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SOURCE='59e78dde6dee12778662beff2c215a88029e73381bc159fb5dcc77f47a46f64c'
ARM='aarch64-unknown-linux-gnu'
X86='x86_64-unknown-linux-gnu'
WASM='wasm32-wasip1'


def validate_cell_labels(fields,expected_op,expected_backend,expected_target):
    for key,expected in [('op',expected_op),('backend',expected_backend),('target',expected_target)]:
        if fields.get(key)!=expected:
            raise ValueError(f'mislabeled {key}: expected {expected!r}, found {fields.get(key)!r}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['arm','x86','wasm','v4x','auditor','driver','out']:
        p.add_argument('--'+flag,required=True,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    plans=[]
    def cells(root,group,op,backends,target):
        return [dict(path=root/group/f'{op}-{backend}',backend=backend,target=target,op=op.replace('pow-','pow:',1)) for backend in backends]
    for op in ['exp','ln','log2','pow-4019999a']:
        plan=cells(a.arm,'portable-top',op,['reference','packed-scalar','neon','scalar-neon'],ARM)
        plan+=cells(a.x86,'x86-v3-top',op,['v3','scalar-v3'],X86)
        plan+=cells(a.wasm,'wasm-top',op,['wasm128','scalar-wasm128'],WASM)
        plan+=cells(a.v4x,'v4x-top',op,['v4x','v4x8','scalar-v4x'],X86)
        plans.append((op,plan))
    for op in ['sqrt','floor','ceil','trunc','round','roundeven']:
        plan=cells(a.arm,'portable-primitives',op,['reference','neon'],ARM)
        plan+=cells(a.x86,'x86-v3-primitives',op,['v1' if op=='sqrt' else 'v2'],X86)
        plan+=cells(a.wasm,'wasm-primitives',op,['wasm128'],WASM)
        plans.append((op,plan))
    for op in ['pow-3f000000','pow-40000000','pow-bf800000']:
        plan=cells(a.arm,'portable-pow-families',op,['reference','neon'],ARM)
        plan+=cells(a.x86,'x86-v3-pow-families',op,['v3'],X86)
        plan+=cells(a.wasm,'wasm-pow-families',op,['wasm128'],WASM)
        plan+=cells(a.v4x,'v4x-pow-families',op,['v4x','v4x8','scalar-v4x'],X86)
        plans.append((op,plan))
    records=[]
    seen=set()
    try:
        with (a.out/'audit-comparison.log').open('x') as log:
            for op,plan in plans:
                summaries=[]
                for cell in plan:
                    path=cell['path']
                    if str(path.resolve()) in seen:
                        raise ValueError(f'duplicate cell {path}')
                    seen.add(str(path.resolve()))
                    summary=dict(line.split('=',1) for line in (path/'summary.txt').read_text().splitlines())
                    if summary.get('status')!='COMPLETE-F32-DOMAIN' or summary.get('count')!='4294967296' or summary.get('chunks')!='4096':
                        raise ValueError(f'partial or wrong domain/count {path}')
                    meta=(path/'metadata.txt').read_text()
                    fields=dict(line.split('=',1) for line in meta.splitlines()[1:] if '=' in line)
                    validate_cell_labels(fields,cell['op'],cell['backend'],cell['target'])
                    if fields.get('source_revision')!='5a9cc685' or fields.get('source_sha256')!=SOURCE:
                        raise ValueError(f'wrong frozen source {path}')
                    summaries.append((path,summary,fields))
                command=[sys.executable,str(a.driver),'compare','--auditor',str(a.auditor),*[str(cell['path']) for cell in plan]]
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                for path,summary,fields in summaries:
                    records.append(dict(operation=op,path=str(path),summary=summary,source_sha256=fields['source_sha256'],binary_sha256=fields['binary_sha256'],target=fields['target'],backend=fields['backend']))
                log.flush()
        if len(records)!=89:
            raise ValueError(f'matrix incomplete {len(records)}/89')
        index=dict(status='ALL-89-CELLS-AUDITED',count_per_cell=2**32,chunk_size=2**20,hash_kind='ordered aggregate of chunk hashes, not raw stream',source_revision='5a9cc685',source_sha256=SOURCE,cells=records)
        (a.out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
        print('89/89 full-domain cells audited; labels, every chunk and canonical aggregate match across selected paths')
    except BaseException as error:
        (a.out/'FAILURE.txt').write_text(str(error)+'\n')
        raise


if __name__=='__main__':
    main()
