#!/usr/bin/env python3
from pathlib import Path
import csv,struct,sys
root=Path(sys.argv[1]);rows=[]
for target in ['i265','arm','wasm']:
 for op,band,label,file,ty in csv.reader((root/'r7900x/values/inventory.tsv').open(),delimiter='\t'):
  if label not in ['std/baseline','zenmath_correct/baseline','libm/baseline','fpmath/baseline','micromath/baseline','metallic/baseline']:continue
  q=root/target/'values'/file
  if not q.exists():
   assert target=='wasm'and op=='powf'and band=='special'and label=='metallic/baseline';continue
  fmt='<III' if ty=='f32' else '<QQQ'
  a=list(struct.iter_unpack(fmt,(root/'r7900x/values'/file).read_bytes()));b=list(struct.iter_unpack(fmt,q.read_bytes()))
  assert len(a)==len(b)==1024
  def nan(v):return(v&((1<<(31 if ty=='f32' else 63))-1))>(0x7f800000 if ty=='f32' else 0x7ff0000000000000)
  assert all(x[:2]==y[:2]for x,y in zip(a,b))
  m=sum(x[2]!=y[2]and not(nan(x[2])and nan(y[2]))for x,y in zip(a,b))
  rows.append(dict(target=target,operation=op,band=band,candidate=label,records=len(a),non_nan_mismatch=m))
p=Path(sys.argv[2]);f=p.open('w');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.close()
print('comparisons',len(rows),'mismatches',[(r['target'],r['operation'],r['band'],r['candidate'],r['non_nan_mismatch'])for r in rows if r['non_nan_mismatch']]);assert p.stat().st_size<=30000
