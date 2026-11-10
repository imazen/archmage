#!/usr/bin/env python3
"""Independent MPFR checks of all retained x/y/output records; never std oracle."""
import ctypes as C,ctypes.util,csv,math,struct,sys
from pathlib import Path
class V(C.Structure):_fields_=[('prec',C.c_long),('sign',C.c_int),('exp',C.c_long),('limbs',C.c_void_p)]
lib=C.CDLL(ctypes.util.find_library('mpfr'));P=C.POINTER(V)
def bind(n,args,ret=C.c_int):
 f=getattr(lib,'mpfr_'+n);f.argtypes=args;f.restype=ret;return f
init=bind('init2',[P,C.c_long],None);setd=bind('set_d',[P,C.c_double,C.c_int]);getd=bind('get_d',[P,C.c_int],C.c_double);getf=bind('get_flt',[P,C.c_int],C.c_float)
for n in ['exp','log','log2','log10','sqrt']:bind(n,[P,P,C.c_int])
bind('pow',[P,P,P,C.c_int]);bind('rint',[P,P,C.c_int])
for n in ['floor','ceil','round']:bind(n,[P,P])
x,y,lo,hi=[V() for _ in range(4)]
for v in [x,y,lo,hi]:init(C.byref(v),256)
cache={}
def oracle(op,a,b,ty):
 key=(op,bits(a,ty),bits(b,ty),ty)
 if key in cache:return cache[key]
 setd(C.byref(x),a,0);setd(C.byref(y),b,0)
 n={'ln':'log','powf':'pow','pow_fixed':'pow','log2_f64':'log2','roundeven':'rint'}.get(op,op)
 f=getattr(lib,'mpfr_'+n)
 for v,rnd in [(lo,3),(hi,2)]:
  args=[C.byref(v),C.byref(x)]
  if n=='pow':args+=[C.byref(y)]
  if n not in ['floor','ceil','round']:args+=[0 if n=='rint' else rnd]
  f(*args)
 getter=getf if ty=='f32' else getd
 aa=float(getter(C.byref(lo),0));bb=float(getter(C.byref(hi),0))
 assert aa==bb or math.isnan(aa)and math.isnan(bb),(key,aa,bb)
 low=getd(C.byref(lo),3);high=getd(C.byref(hi),2)
 cache[key]=(aa,low,high);return cache[key]
def bits(v,ty):return int.from_bytes(struct.pack('<f' if ty=='f32' else '<d',v),'little')
def order(v,w):return (~v&((1<<w)-1))if v>>(w-1)else(v|(1<<(w-1)))
root=Path(sys.argv[1]);rows=[]
for inv in sorted(root.glob('*/values/inventory.tsv')):
 for op,band,label,file,ty in csv.reader(inv.open(),delimiter='\t'):
  n=correct=special=zero=0;maxsteps=0;maxerr=0.;worst='';finite=0
  width=32 if ty=='f32' else 64
  for a,b,out in struct.iter_unpack('<fff' if ty=='f32' else '<ddd',(inv.parent/file).read_bytes()):
   n+=1;ref,low,high=oracle(op,a,b,ty);ob=bits(out,ty);rb=bits(ref,ty)
   same=ob==rb or math.isnan(out)and math.isnan(ref)
   correct+=same
   if out==0 and ref==0 and ob!=rb:zero+=1
   if not(math.isfinite(out)and math.isfinite(ref)):
    if not same:special+=1
    continue
   finite+=1;steps=abs(order(ob,width)-order(rb,width));maxsteps=max(maxsteps,steps)
   if math.isfinite(low)and math.isfinite(high):
    exp=(math.frexp(abs(low))[1]-1) if low else (-126 if ty=='f32' else -1022)
    ulp=math.ldexp(1.,max(-149 if ty=='f32' else -1074,exp-(23 if ty=='f32' else 52)))
    err=math.nextafter(math.nextafter(max(abs(out-low),abs(out-high)),math.inf)/ulp,math.inf)
    if err>maxerr:maxerr=err;worst=f'{bits(a,ty):x},{bits(b,ty):x},{ob:x},{rb:x}'
  rows.append(dict(platform=inv.parent.parent.name,operation=op,band=band,candidate=label,n=n,correct=correct,finite=finite,max_representable_steps=maxsteps,true_error_upper_f64_endpoints=maxerr,special_mismatch=special,signed_zero_mismatch=zero,worst=worst))
  if len(rows)%100==0:print('checked',len(rows),'files',flush=True)
with(root/'accuracy.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Strict gate only for candidates that actually promise correct rounding.
bad=[r for r in rows if ('zenmath_correct' in r['candidate']or'magemath_correct'in r['candidate'])and r['correct']!=r['n']]
print('files',len(rows),'correct-contract findings',bad,flush=True)
if bad:sys.exit(1)
