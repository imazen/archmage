#!/usr/bin/env python3
"""Complete numerical variance inventory across actual provider executions."""
from array import array
import argparse
import gzip
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'powf_module_2026-11-07'))
from replacement_accuracy import rounded_reference
import rounding_oracle as model
import rounding_accuracy as reference

def kind(left,right):
    if left==right:return None
    ln=(left&0x7fffffff)>0x7f800000;rn=(right&0x7fffffff)>0x7f800000
    return 'NaN_class' if ln!=rn else 'NaN_payload' if ln else 'non_NaN_bits'

def shape(b):
    magnitude=b&0x7fffffff
    category='nan' if magnitude>0x7f800000 else'infinity'if magnitude==0x7f800000 else'zero'if magnitude==0 else'subnormal'if magnitude<0x00800000 else'normal'
    return category if category=='nan'else(category,b>>31)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('runs',nargs='+',help='execution-name=accuracy-directory')
    p.add_argument('--expected-cases',type=int,default=27909,help='explicit size of a separately identified bounded witness set')
    p.add_argument('--reference-suffix',default='/std_Rust_baseline/x1',help='explicit observed reference; independent MPFR accuracy is unchanged')
    a=p.parse_args();a.directory.mkdir(exist_ok=False,parents=True)
    contexts={};inputs=None;intra={};mixed={}
    for spec in a.runs:
        execution,path=spec.split('=',1);path=Path(path)
        rows=[line.split() for line in (path/'inputs.txt').read_text().splitlines()]
        if inputs is None:inputs=rows
        assert rows==inputs
        report=json.loads((path/'summary.json').read_text())
        for index,(name,stats) in enumerate(report['providers'].items()):
            key=execution+'/'+name;width=stats['width'];broadcast=array('I');mix=array('I');exceptions={}
            with gzip.open(path/f'{index:03d}.actual.gz','rt') as actual:
                for row in range(len(rows)):
                    values=[int(x,16) for x in next(actual).split()];assert len(values)==width
                    broadcast.append(values[0])
                    if any(v!=values[0] for v in values[1:]):exceptions[row]=values
                for line in actual:mix.extend(int(x,16) for x in line.split())
            del mix[len(rows):];assert len(mix)==len(broadcast)==len(rows)
            contexts[key]=(broadcast,mix);intra[key]=exceptions
            counts={'unique_input_cases':0,'non_NaN_bits':0,'NaN_class':0,'NaN_payload':0}
            for x,y in zip(broadcast,mix,strict=True):
                category=kind(x,y)
                if category:counts['unique_input_cases']+=1;counts[category]+=1
            mixed[key]=counts
    assert len(inputs)==a.expected_cases
    names=list(contexts);base=next(k for k in names if k.endswith(a.reference_suffix))
    # One per-input complete map preserves every pairwise variance without a
    # quadratic ledger. Raw .actual.gz files retain every repeated broadcast.
    used=0;records=0;budget=512*1024*1024
    against_base={name:{'non_NaN_bits':0,'NaN_class':0,'NaN_payload':0}for name in names}
    gamma={name:{'positive_cases':0,'zero_cases':0,'rounded_differences':0,'class_or_sign_mismatches':0,
                 'max_finite_output_ulp_upper':0.,'worst':None}for name in names}
    with gzip.open(a.directory/'all-variances.jsonl.gz','wt') as ledger:
        for index,row in enumerate(inputs):
            xb_bits,nb_bits=map(lambda s:int(s,16),row[:2]);xb,nb=model.fp.value(xb_bits),model.fp.value(nb_bits)
            rounded=rounded_reference(int(row[0],16),int(row[1],16)) if xb>0 and math.isfinite(xb) and math.isfinite(nb) else None
            in_gamma=model.preflight(xb_bits,nb_bits,2)==0
            if in_gamma:
                if xb_bits==0:
                    rounded=0x3f800000 if nb==0 else 0
                else:
                    bounds,quantum=reference.reference(xb_bits,nb_bits)
                errors={}
                for name,(_,actual)in contexts.items():
                    stats=gamma[name];stats['zero_cases'if xb_bits==0 else'positive_cases']+=1
                    b=actual[index];stats['rounded_differences']+=b!=rounded
                    stats['class_or_sign_mismatches']+=shape(b)!=shape(rounded)
                    y=model.fp.value(b)
                    if xb_bits==0:
                        continue
                    if not math.isfinite(y):continue
                    if b not in errors:
                        error=max(abs(Fraction.from_float(y)-Fraction.from_float(v))for v in bounds)/(Fraction(2)**quantum)
                        errors[b]=math.nextafter(float(error),math.inf)if error else 0.
                    if errors[b]>stats['max_finite_output_ulp_upper']:
                        stats.update(max_finite_output_ulp_upper=errors[b],worst={'input':row,'actual':f'{b:08x}','rounded_MPFR':f'{rounded:08x}'})
            outputs={name:{'broadcast':f'{b[index]:08x}','mixed':f'{m[index]:08x}'}for name,(b,m)in contexts.items()}
            for name in names:
                if index in intra[name]:outputs[name]['broadcast_lanes']=[f'{v:08x}'for v in intra[name][index]]
                category=kind(contexts[base][1][index],contexts[name][1][index])
                if category:against_base[name][category]+=1
            distinct={v for output in outputs.values()for value in output.values()for v in (value if isinstance(value,list)else[value])}
            if rounded is not None:distinct.add(f'{rounded:08x}')
            if len(distinct)<=1:continue
            record={'case':index,'input':row,'outputs':outputs,'rounded_MPFR':f'{rounded:08x}'if rounded is not None else None}
            text=json.dumps(record,separators=(',',':'))+'\n';used+=len(text.encode())
            if used>budget:raise RuntimeError('variance budget exceeded; partial retained')
            ledger.write(text);records+=1
    # Compare matching provider and width across executions explicitly.
    cross=[]
    for i,left in enumerate(names):
        for right in names[i+1:]:
            if left.split('/',1)[1]!=right.split('/',1)[1]:continue
            counts={'non_NaN_bits':0,'NaN_class':0,'NaN_payload':0}
            for x,y in zip(contexts[left][1],contexts[right][1],strict=True):
                category=kind(x,y)
                if category:counts[category]+=1
            cross.append({'left':left,'right':right,**counts})
    summary={'cases':len(inputs),'provider_width_executions':len(contexts),'variance_cases':records,
             'observed_reference':base,'reference_is_not_mathematical_truth':True,
             'all_variances':'all-variances.jsonl.gz','ledger_uncompressed_bytes':used,'ledger_budget_bytes':budget,
             'mixed_lane_unique_input_counts':mixed,'against_observed_reference':against_base,'cross_execution':cross,
             'Gamma_preset':gamma,'Gamma_predicate':'unchanged independent oracle preflight(base, exponent, Gamma); normal-or-zero exponent, positive-zero-or-bounded-normal base',
             'broadcast_lane_variance_case_counts':{k:len(v)for k,v in intra.items()}}
    (a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items()if k not in ['mixed_lane_unique_input_counts','against_observed_reference','cross_execution','broadcast_lane_variance_case_counts']},indent=2))

if __name__=='__main__':main()
