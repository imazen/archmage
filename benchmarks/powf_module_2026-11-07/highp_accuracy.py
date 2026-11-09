#!/usr/bin/env python3
"""Public HIGHP actual outputs; independent directed MPFR accuracy and classes."""
import argparse,json,math,subprocess,shlex
from fractions import Fraction
from pathlib import Path
import rounding_accuracy as reference
import rounding_oracle as model
from replacement_accuracy import rounded_reference
p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('fixtures',type=Path);p.add_argument('output',type=Path);p.add_argument('--tier',choices=['scalar','native'],default='native');p.add_argument('--runner',default='');a=p.parse_args();assert not a.output.exists()
request=''.join(' '.join(row.split()[:3])+'\n'for row in a.fixtures.read_text().splitlines())
# Full-domain witnesses exercise std signs and special classes outside approximation domains.
request+=''.join(f'{b:08x} {n:08x} 3\n'for b in [0,0x80000000,1,0x007fffff,0x7f7fffff,0x7f800000,0xff800000,0x7fc00001,0x7f800001,0xbf800000,0xc0000000]for n in [0,0x80000000,1,0x3f000000,0x40000000,0x40400000,0xbf800000,0x7f800000,0x7fc00001])
run=subprocess.run(shlex.split(a.runner)+[str(a.binary.resolve()),a.tier],input=request,text=True,capture_output=True,check=True);a.output.with_suffix('.actual').write_text(run.stdout);a.output.with_suffix('.inputs').write_text(request)
report={'cases':len(request.splitlines()),'methods':'all16 publicHIGHP configurations; widths1/4/8/16; vector/array and const/runtime std-equivalence','tier':a.tier,'runner':a.runner,'precision':256,'claim':'sampled error, std precision unspecified','classes':{},'max_ulp_upper':0.,'rounded_differences':0,'worst':None}
for row,out in zip(request.splitlines(),run.stdout.splitlines(),strict=True):
 fields=row.split();got=out.split();assert got[:3]==fields
 xb,nb=int(fields[0],16),int(fields[1],16);bits=int(got[3],16);x,n,y=model.fp.value(xb),model.fp.value(nb),model.fp.value(bits)
 cls='nan'if math.isnan(y)else'infinity'if math.isinf(y)else'zero'if y==0 else'subnormal'if abs(y)<2**-126 else'normal'
 report['classes'][cls]=report['classes'].get(cls,0)+1
 if not(x>0 and math.isfinite(x)and math.isfinite(n)):continue
 rounded=rounded_reference(xb,nb);report['rounded_differences']+=bits!=rounded
 bounds,q=reference.reference(xb,nb)
 if not math.isfinite(y)or not all(math.isfinite(z)for z in bounds):continue
 quantum=Fraction(2)**q;value=Fraction.from_float(y)
 error=max(abs(value-Fraction.from_float(z))for z in bounds)/quantum
 error=math.nextafter(float(error),math.inf)if error else 0.
 if error>report['max_ulp_upper']:report.update(max_ulp_upper=error,worst={'input':row,'actual':got[3],'reference':bounds})
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
