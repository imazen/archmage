#!/usr/bin/env python3
"""Bounded upstream hard-to-round search; this is not a maximum-error proof."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('crate',type=Path);p.add_argument('directory',type=Path)
p.add_argument('--include',type=Path,action='append',default=[])
p.add_argument('--sonames',action='store_true',help='link installed Linux runtime sonames when dev symlinks are isolated')
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
source=a.crate/'vendor/src/binary32/pow/worst.c';assert source.is_file()
shutil.copy2(source,a.directory/'worst.c')
flags=['-std=c11','-O2','-DCORE_MATH_NO_OPENMP','-frounding-math','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard']
args=['-m','16','-d','2','-xmin','0x1p-1','-xmax','0x1p+1','-ymin','0x1.333334p+1','-ymax','0x1.333336p+1']
manifest={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'arguments':args,
          'scope':'x in [0.5,2], one f32 exponent 4019999a; target 16 identical bits after rounding bit',
          'logical_pair_budget':20_000_000,'wall_limit_seconds':60,'OpenMP':False,
          'inclusive_domain_pairs':0x40000000-0x3f000000+1,
          'claim':'upstream hard-to-round search in a bounded rectangle; not global maximum approximation error'}
for mode,extra in [('count',['-DCOUNT']),('search',[])]:
    binary=(a.directory/f'worst-{mode}').resolve()
    includes=[value for path in a.include for value in ['-I',str(path.resolve())]]
    libraries=['-l:libmpfr.so.6','-l:libgmp.so.10']if a.sonames else['-lmpfr','-lgmp']
    command=['cc',*flags,*extra,*includes,str(source.resolve()),*libraries,'-lm','-o',str(binary)]
    with (a.directory/f'{mode}-build.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (a.directory/f'{mode}.stdout').open('w')as out,(a.directory/f'{mode}.stderr').open('w')as err:
        run=subprocess.run(['timeout','60s',str(binary),*args],stdout=out,stderr=err)
    manifest[mode]={'command':command,'returncode':run.returncode}
    (a.directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if run.returncode:raise SystemExit('bounded search failed/timed out; partial evidence retained')
    if mode=='count':
        text=(a.directory/'count.stdout').read_text();count=int(re.search(r'tot_checks=(\d+)',text)[1])
        manifest['upstream_COUNT_pairs']=count
        # Upstream COUNT omits the final inclusive endpoint in its last binade.
        # Admit using the independently counted fixed rectangle, not that value.
        assert manifest['inclusive_domain_pairs']<=manifest['logical_pair_budget'],'budget admission failed before search'
    else:
        output=(a.directory/'search.stdout').read_text()
        manifest['search_logical_checks']=int(re.search(r'# tot_checks=(\d+)',output)[1])
        assert manifest['search_logical_checks']==manifest['inclusive_domain_pairs']
        rows=[]
        for line in output.splitlines():
            if line.startswith('#')or not line.strip():continue
            fields=line.split('#',1)[0].strip().split(',')
            assert len(fields)==2,('unrecognized upstream witness',line)
            bits=[struct.unpack('<I',struct.pack('<f',float.fromhex(value.strip())))[0]for value in fields]
            rows.append(f'{bits[0]:08x} {bits[1]:08x} 2\n')
        (a.directory/'witnesses.txt').write_text(''.join(rows))
        manifest['witness_records']=len(rows);manifest['unique_pairs']=len(set(rows))
        manifest['witness_sha256']=hashlib.sha256((a.directory/'witnesses.txt').read_bytes()).hexdigest()
(a.directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
