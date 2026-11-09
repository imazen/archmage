#!/usr/bin/env python3
"""Actual outputs, MPFR-256 errors and a complete reconstructible variance ledger."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'powf_module_2026-11-07'))
import rounding_accuracy as reference
import rounding_oracle as model
from replacement_accuracy import rounded_reference

def classify(b):
    v = model.fp.value(b)
    return 'nan' if math.isnan(v) else 'infinity' if math.isinf(v) else 'zero' if v==0 else 'subnormal' if abs(v)<2**-126 else 'normal'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('fixtures',type=Path);p.add_argument('directory',type=Path)
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text())
    rows=[line.split()[:3] for line in a.fixtures.read_text().splitlines()]
    # Fixtures can be the original 27,810 rows or the already expanded .inputs.
    if len(rows)==27810:
        rows += [[f'{b:08x}',f'{n:08x}','3'] for b in [0,0x80000000,1,0x007fffff,0x7f7fffff,0x7f800000,0xff800000,0x7fc00001,0x7f800001,0xbf800000,0xc0000000]
                 for n in [0,0x80000000,1,0x3f000000,0x40000000,0x40400000,0xbf800000,0x7f800000,0x7fc00001]]
    assert len(rows)==27909
    (a.directory/'inputs.txt').write_text(''.join(' '.join(r)+'\n' for r in rows))
    refs=[]
    for xb,nb,_ in rows:
        x,n=model.fp.value(int(xb,16)),model.fp.value(int(nb,16))
        refs.append((rounded_reference(int(xb,16),int(nb,16)),reference.reference(int(xb,16),int(nb,16)))
                    if x>0 and math.isfinite(x) and math.isfinite(n) else None)
    report={'cases':len(rows),'precision':256,'claim':'sampled positive-finite MPFR accuracy; special values compared by observed bits/classes',
            'providers':{},'mixed_lane_variances':0,'ledger_budget_bytes':512*1024*1024}
    all_outputs={}
    for provider in manifest['providers']:
        label=provider['label'];worker=manifest['workers'][provider['worker']]
        assert hashlib.sha256(Path(worker['binary']).read_bytes()).hexdigest()==worker['sha256']
        for width in [1,4,8,16]:
            if width<provider['lanes']:continue
            key=f'{label}/x{width}'
            command=[]
            for row in rows:command.append(f'E {provider["method"]} {width} '+ ' '.join(row[:2]*width))
            for start in range(0,len(rows),width):
                chunk=rows[start:start+width];chunk+= [chunk[-1]]*(width-len(chunk))
                command.append(f'E {provider["method"]} {width} '+' '.join(v for row in chunk for v in row[:2]))
            request='\n'.join(command)+'\n'
            actual=subprocess.run([worker['binary']],input=request,text=True,capture_output=True,timeout=300)
            stem=a.directory/f'{len(all_outputs):03d}'
            with gzip.open(stem.with_suffix('.actual.gz'),'wt') as f:f.write(actual.stdout)
            stem.with_suffix('.stderr').write_text(actual.stderr)
            actual.check_returncode()
            got=[[int(v,16) for v in line.split()] for line in actual.stdout.splitlines()]
            assert len(got)==len(command) and all(len(line)==width for line in got)
            broadcast=got[:len(rows)];mixed=[v for line in got[len(rows):] for v in line][:len(rows)]
            all_outputs[key]=(broadcast,mixed)
            stats={'width':width,'native_lanes':provider['lanes'],'classes':{},'rounded_differences':0,
                   'max_finite_output_ulp_upper':0.,'worst':None,'nonfinite_mismatches_against_MPFR':0,
                   'range_witnesses':[],'mixed_lane_non_NaN_differences':0,'NaN_payload_lane_differences':0}
            for index,(lanes,b) in enumerate(zip(broadcast,mixed,strict=True)):
                cls=classify(b);stats['classes'][cls]=stats['classes'].get(cls,0)+1
                for value in lanes:
                    if value!=b:
                        k='NaN_payload_lane_differences' if classify(value)==cls=='nan' else 'mixed_lane_non_NaN_differences'
                        stats[k]+=1;report['mixed_lane_variances']+=1
                if refs[index] is None:continue
                rounded,(bounds,quantum)=refs[index]
                stats['rounded_differences']+=b!=rounded
                y=model.fp.value(b)
                if not math.isfinite(y) and b!=rounded:
                    stats['nonfinite_mismatches_against_MPFR']+=1
                    if len(stats['range_witnesses'])<8:stats['range_witnesses'].append({'input':rows[index],'actual':f'{b:08x}','rounded_MPFR':f'{rounded:08x}'})
                if not math.isfinite(y) or not all(math.isfinite(z) for z in bounds):continue
                value=Fraction.from_float(y)
                error=max(abs(value-Fraction.from_float(z)) for z in bounds)/(Fraction(2)**quantum)
                error=math.nextafter(float(error),math.inf) if error else 0.
                if error>stats['max_finite_output_ulp_upper']:
                    stats.update(max_finite_output_ulp_upper=error,worst={'input':rows[index],'actual':f'{b:08x}','reference':bounds})
            report['providers'][key]=stats
            (a.directory/'progress.json').write_text(json.dumps({'completed_provider_widths':len(all_outputs),'last':key})+'\n')
            print(json.dumps({'provider':key,**stats}),flush=True)
    # Per-input complete output map preserves every pairwise bit/class/payload
    # variance without an O(providers^2) expansion. Raw outputs retain all ties.
    used=0;count=0
    with gzip.open(a.directory/'all-variances.jsonl.gz','wt') as ledger:
        for index,row in enumerate(rows):
            values={key:{'mixed':f'{mixed[index]:08x}','broadcast':[f'{b:08x}' for b in broadcast[index]]}
                    for key,(broadcast,mixed) in all_outputs.items()}
            flattened={v for item in values.values() for v in [item['mixed']]+item['broadcast']}
            if len(flattened)<=1:continue
            record={'case':index,'input':row,'outputs':values,'rounded_MPFR':f'{refs[index][0]:08x}' if refs[index] else None}
            text=json.dumps(record,separators=(',',':'))+'\n';used+=len(text.encode())
            if used>report['ledger_budget_bytes']:raise RuntimeError('ledger budget exceeded; partial retained')
            ledger.write(text);count+=1
    report.update(variance_cases=count,ledger_uncompressed_bytes=used,provider_widths=len(all_outputs))
    (a.directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='providers'},indent=2))

if __name__=='__main__':main()
