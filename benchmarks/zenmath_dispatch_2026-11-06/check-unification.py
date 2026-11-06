#!/usr/bin/env python3
"""Check the real external baseline consumer, with archmage/std unified."""
from pathlib import Path
import re,sys
p=Path(sys.argv[1]);s=p.read_text();count=0
for name in ['compile_only_floor','compile_only_exp']:
    blocks=[b.split('\n}')[0]for b in s.split('\ndefine ')if name in b.split('\n')[0]]
    assert len(blocks)==1,(name,len(blocks))
    b=blocks[0]
    forbidden=['archmage','summon','detect','cpuid','_v3','_v4x','_v2','OnceLock']
    assert not any(x in b for x in forbidden),(name,'runtime/proof selection survived in compile-only consumer')
    count+=1
print('PASS',count,'real external consumer bodies: no runtime feature selection with archmage/std unified')
