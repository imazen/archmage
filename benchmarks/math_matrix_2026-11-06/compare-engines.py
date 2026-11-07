#!/usr/bin/env python3
"""Byte and non-NaN output comparison of identical guest binaries/corpora."""
import csv,hashlib,json,struct,sys
from pathlib import Path
old,new,out=map(Path,sys.argv[1:]);a=list(csv.reader((old/'inventory.tsv').open(),delimiter='\t'));b=list(csv.reader((new/'inventory.tsv').open(),delimiter='\t'));assert a==b
rows=[]
for op,band,label,file,ty in a:
 aa=(old/file).read_bytes();bb=(new/file).read_bytes();fmt='<III'if ty=='f32'else'<QQQ'
 av=list(struct.iter_unpack(fmt,aa));bv=list(struct.iter_unpack(fmt,bb));assert len(av)==len(bv)==1024
 def isnan(v):return (v&((1<<(31 if ty=='f32'else 63))-1))>(0x7f800000 if ty=='f32'else 0x7ff0000000000000)
 mismatches=sum(x[2]!=y[2]and not(isnan(x[2])and isnan(y[2]))for x,y in zip(av,bv));assert all(x[:2]==y[:2]for x,y in zip(av,bv))
 rows.append(dict(operation=op,band=band,candidate=label,records=1024,exact_file_equal=aa==bb,non_nan_mismatch=mismatches))
assert len(rows)==1046
with out.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(files=len(rows),records=sum(r['records']for r in rows),non_nan_mismatch=sum(r['non_nan_mismatch']for r in rows),different_files=sum(not r['exact_file_equal']for r in rows))
print(json.dumps(summary));assert summary['non_nan_mismatch']==0
