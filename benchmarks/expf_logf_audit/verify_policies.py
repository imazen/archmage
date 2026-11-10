#!/usr/bin/env python3
"""Stream actual option-loop outputs against independent rational graphs."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_quality as model
model.LOG_LOWP_RATIONAL=True
p=argparse.ArgumentParser(description=__doc__);p.add_argument('prefix',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();assert not a.output.exists()
@lru_cache(maxsize=65536)
def mirror(log,raw,flags,fused):return model.core_mirror('L'if log else'E',raw,flags,fused)
errors=[];count=0;segments=json.loads(a.prefix.with_suffix('.json').read_text())
with a.prefix.with_suffix('.bin').open('rb')as f:
 for s in segments:
    assert f.tell()==s['offset'];raw=f.read(s['size']*4);assert len(raw)==s['size']*4
    # Array Caller starts with ScalarToken regardless of its surrounding frame;
    # the explicit token lane-map is the separate proof-preserving alternative.
    native=s['token_lanes']and s['context']=='native'or not s['token_lanes']and not s['flags']&4
    fused=not s['flags']&16 or native
    flags=s['flags']&72
    acc=[model.fp.bits(.75)]*s['width']
    for i,(got,)in enumerate(struct.iter_unpack('<I',raw)):
        lane=i%s['width'];x=acc[lane]if s['chain']else model.fp.bits(.5+(i%256)/512.)
        expected=mirror(s['log'],x,flags,fused)
        if got!=expected:errors.append(dict(segment=s,index=i,actual=f'{got:08x}',expected=f'{expected:08x}'))
        if s['chain']:acc[lane]=0x3f000000|(expected&0x007fffff)
        count+=1
 assert not f.read(1)
report=dict(segments=len(segments),outputs=count,mismatches=len(errors),examples=errors[:20],claim='zero-bit graph identity on frozen ordinary-input loops; not a mathematical accuracy bound')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if errors:raise SystemExit(1)
