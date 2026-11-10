#!/usr/bin/env python3
"""Audit local glibc tags, preserving exact hashes and narrowly normalized diffs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('repository',type=Path);p.add_argument('directory',type=Path)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
def git(*args):return subprocess.check_output(['git','-C',str(a.repository),*args])
def sha(data):return hashlib.sha256(data).hexdigest()
paths=[f'sysdeps/ieee754/{precision}/{name}.c'for precision,names in [
    ('flt-32',['e_expf','e_exp2f','e_logf','e_log2f','e_powf']),
    ('dbl-64',['e_exp','e_exp2','e_log','e_log2','e_pow'])]for name in names]
report={'tags':{tag:git('rev-parse',tag+'^{}').decode().strip()for tag in ['glibc-2.27','glibc-2.28','glibc-2.29','glibc-2.35','glibc-2.39','glibc-2.41']},
        'mirror_head':git('log','-1','--format=%H %cI %s').decode().strip(),
        'normalization':'only Copyright (C) start/end years; no operands, comments or other tokens changed',
        'files':[]}
for path in paths:
    left=git('show','glibc-2.35:'+path);right=git('show','glibc-2.39:'+path)
    def normalized(value):return re.sub(rb'Copyright \(C\) [0-9]{4}(?:-[0-9]{4})?',b'Copyright (C) YEARS',value)
    report['files'].append({'path':path,'sha256_2_35':sha(left),'sha256_2_39':sha(right),
                            'equal_after_copyright_years':normalized(left)==normalized(right)})
    # Negative control: a changed arithmetic operand must not normalize away.
    assert normalized(b'x * 2.0')!=normalized(b'x * 3.0')
(a.directory/'scalar-tag-diff.txt').write_bytes(git('diff','glibc-2.35','glibc-2.39','--',*paths))
for name,start,end in [('before-2.39','glibc-2.35','glibc-2.39'),('after-2.39','glibc-2.39','glibc-2.41')]:
    (a.directory/f'core-math-{name}.txt').write_bytes(git('log','--format=%H %ad %s','--date=short','--regexp-ignore-case','--grep=CORE.MATH',start+'..'+end))
for commit in ['4ea49f4c08','424c4f60ed','f41b0a43e4','e70c176825','f6b10ed8e9','29531b122c']:
    (a.directory/f'{commit}.txt').write_bytes(git('show','--format=fuller','--stat',commit))
for path in ['NEWS','sysdeps/x86_64/fpu/multiarch/e_powf.c','sysdeps/x86_64/fpu/multiarch/ifunc-fma.h','sysdeps/x86_64/fpu/multiarch/Makefile']:
    tag='glibc-2.41'if path=='NEWS'else'glibc-2.39'
    (a.directory/(path.replace('/','__')+'.txt')).write_bytes(git('show',tag+':'+path))
(a.directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
