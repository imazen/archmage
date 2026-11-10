#!/usr/bin/env python3
"""Aggregate broad policy target equality, separately checking canonical NaNs."""
import array,csv,sys
from pathlib import Path
root=Path(sys.argv[1]);reference=root/'conditional';rows=[]
files=sorted(reference.glob('*.bin'));assert len(files)==18
for target in ['native','arm','v3','nehalem','wasm']:
    for f in files:
        a=array.array('I');a.frombytes(f.read_bytes());b=array.array('I');b.frombytes((root/target/f.name).read_bytes());assert len(a)==len(b)==909487
        mismatch=0;nan_difference=0;zero=0
        for x,y in zip(a,b):
            if x==y:continue
            if (x&0x7fffffff)>0x7f800000 and (y&0x7fffffff)>0x7f800000:nan_difference+=1
            else:
                mismatch+=1
                if x&0x7fffffff==0 and y&0x7fffffff==0:zero+=1
        if f.stem=='canonical':assert nan_difference==0,'canonical NaN mismatch'
        rows.append(dict(target=target,policy=f.stem,records=len(a),non_nan_mismatch=mismatch,nan_payload_difference=nan_difference,signed_zero_mismatch=zero))
        assert mismatch==0,(target,f.name,mismatch)
with (root/'cross-equality.csv').open('w') as out:
    writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Broad equality:',len(rows),'policy/target comparisons,909487records each, zero non-NaN differences')
