#!/usr/bin/env python3
"""Check every timed math cell has an independently checked output file."""
from pathlib import Path
import csv,json,sys
root=Path(sys.argv[1]);coverage=list(csv.DictReader(Path(sys.argv[2]).open()));assert len(coverage)==10
expected={'i265':('timing-i265.json',1047),'r7900x':('timing-r7900x-final.json',1428),'arm':('timing-arm-final.json',1047),'wasm':('timing-wasm.json',1046)}
allcells=0
for target,(file,count)in expected.items():
 d=json.loads((root/target/file).read_text());assert not d['incomplete']and not d['unreliable'];cells={(g['group_name'],b['name'])for g in d['comparisons']if not g['group_name'].startswith('algebraic/')for b in g['benchmarks']}
 values={(op+'/'+band,label)for op,band,label,_,_ in csv.reader((root/target/'values/inventory.tsv').open(),delimiter='\t')}
 assert cells==values,(target,'timed-only',cells-values,'validation-only',values-cells)
 assert len(cells)==count,(target,len(cells));allcells+=len(cells)
 assert all(b['summary']['n']>=30 for g in d['comparisons']for b in g['benchmarks'])
 print(target,'all timed math cells independently validated',len(cells))
assert allcells==4568
assert sum(int(r['cells'])for r in coverage)==5091
traps=(root/'wasm/wasm-traps.tsv').read_text().splitlines();assert traps==['powf/special\tmetallic/baseline']
accuracy=list(csv.DictReader((root/'accuracy.csv').open()));assert len(accuracy)==4568
assert not [r for r in accuracy if ('zenmath_correct'in r['candidate']or'magemath_correct'in r['candidate'])and r['correct']!=r['n']]
print('FINAL AUDIT PASS: 4568 validated throughput cells +180 algebraic +343 feedback =5091; one explicit WASM trap')
