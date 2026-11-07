#!/usr/bin/env python3
"""Render retained zenbench paired CIs; require every final platform and replacement."""
import csv
from pathlib import Path
import sys
root=Path(sys.argv[1]);out=Path(sys.argv[2])
def rows(names):
    result={}
    for name in names:
        with (root/'sources/native'/name).open() as f:
            for r in csv.DictReader(f):
                if r['eligible']=='true':result[(r['group'],r['baseline'],r['candidate'])]=r
    return result
native=rows(['gamma-native-v6-pairs.csv','gamma-native-repeat-v6-pairs.csv'])
arm=rows(['gamma-arm-v6-pairs.csv'])
wasm=rows(['gamma-wasm-r7900x-v5-pairs.csv','gamma-wasm-r7900x-repeat-v5-pairs.csv'])
policy=rows(['policy-native-all-pairs.csv','policy-native-repeat-pairs.csv'])
assert len({k[0]for k in native})==len({k[0]for k in arm})==28
assert len({k[0]for k in wasm})==3
lines=['# Detailed paired tables','','All values are ns/value. Differences are candidate minus the named control.','Intervals are direct all-round 95% bootstrap intervals using zenbench MeanCi,','with no extra exclusions. Dirty groups are replaced in full; their original','rows remain archived. An interval crossing zero does not establish an order.','']
def table(title,items):
    lines.extend([title,'','| Platform / workload | Control | Candidate | Control mean | Candidate mean | Difference 95% CI |','|---|---|---|---:|---:|---:|'])
    for platform,data,g,b,c in items:
        r=data[(g,b,c)];assert r['eligible']=='true' and int(r['dirty'])==0
        lines.append(f"| {platform}: {g} | {b} | {c} | {float(r['baseline_mean']):.4f} | {float(r['candidate_mean']):.4f} | [{float(r['lo']):+.4f}, {float(r['hi']):+.4f}] |")
    lines.append('')
table('Final scalar ordinary (all 28 groups per native platform)',[(p,d,g,'std','current/ordinary')for p,d in [('V3',native),('NEON',arm)]for g in sorted({k[0]for k in d})])
table('Final matched once-buffer controls',[(p,d,g,b,'current/slice-context')for p,d in [('V3',native),('NEON',arm)]for g in ['dynamic/decode/values/4096','dynamic/encode/values/4096','dynamic/general/65536','dynamic/historical/65536']for b in ['before/slice-context','std/slice-context']])
table('Final WASM ordinary and packed controls',[( 'SIMD128',wasm,g,b,c)for g in sorted({k[0]for k in wasm})for b,c in [('std/control','ordinary/correct'),('old-ordinary/correct','ordinary/correct'),('old-packed-wasm128/correct','packed-wasm128/correct')]])
policies=['correct','midp-full-fused','lowp-full-fused','midp-full-separate','lowp-full-separate','midp-pre-fused','lowp-pre-fused','midp-pre-separate','lowp-pre-separate']
table('Initial routing stage, before gamma specialization (decode uniform, 1,024 values)',[('V3',policy,'gamma-decode/uniform/stream1024','old-scalar/'+p,'current/'+p)for p in policies])
out.write_text('\n'.join(lines)+'\n')
print('PASS mandatory native28, NEON28, WASM3 and nine-policy table inputs')
