#!/usr/bin/env python3
"""No implicit runtime power probes with only zenmath/archmage enabled."""
from pathlib import Path
import sys
blocks=[]
for filename in sys.argv[1:]:
    s=Path(filename).read_text()
    blocks.extend(b.split('\n}')[0]for b in s.split('\ndefine ')if 'calculate' in b.split('\n')[0])
assert len(blocks)==9,len(blocks)
for body in blocks:
    for forbidden in ['summon','detect','cpuid','OnceLock','native_v3','native_neon','powf_v3','powf_neon']:
        assert forbidden not in body,(forbidden,body.splitlines()[0])
print('PASS 9 real external power-policy bodies with archmage/std unified; no runtime/proof entry references')
