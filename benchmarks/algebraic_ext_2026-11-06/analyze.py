#!/usr/bin/env python3
"""Independent MPFR observations; no algebraic precision/parity bound asserted."""
from pathlib import Path
import csv,ctypes as C,ctypes.util,math,struct,sys
class V(C.Structure):_fields_=[('p',C.c_long),('sign',C.c_int),('exp',C.c_long),('limbs',C.c_void_p)]
l=C.CDLL(ctypes.util.find_library('mpfr'));P=C.POINTER(V)
def bind(n,args,ret=C.c_int):
 f=getattr(l,'mpfr_'+n);f.argtypes=args;f.restype=ret;return f
init=bind('init2',[P,C.c_long],None);setd=bind('set_d',[P,C.c_double,C.c_int]);getd=bind('get_d',[P,C.c_int],C.c_double);getf=bind('get_flt',[P,C.c_int],C.c_float)
for op in ['add','mul','div','fmod']:bind(op,[P,P,P,C.c_int])
a,b,c,t,z,lo,hi=[V()for _ in range(7)]
for v in[a,b,c,t,z,lo,hi]:init(C.byref(v),512)
ref=lambda v:C.byref(v)
def bits(v,ty):return int.from_bytes(struct.pack('<f'if ty=='f32'else'<d',v),'little')
def ordered(v,ty):
 n=32 if ty=='f32'else 64;return (~v&((1<<n)-1))if v>>(n-1)else v|(1<<(n-1))
cache={}
def oracle(group,band,rows):
 op,ty=group.split('/');key=(group,band)
 if key in cache:return cache[key]
 getter=getf if ty=='f32'else getd
 out=[]
 if op=='dot':
  setd(ref(z),0.,0)
  for x,y,_ in rows:
   setd(ref(a),x,0);setd(ref(b),y,0);assert l.mpfr_mul(ref(t),ref(a),ref(b),0)==0;assert l.mpfr_add(ref(z),ref(z),ref(t),0)==0
  out=[getter(ref(z),0)]
 else:
  for x,y,_ in rows:
   setd(ref(a),x,0);setd(ref(b),y,0)
   if op=='poly':
    co=struct.unpack('<f',struct.pack('<f',1/6))[0]if ty=='f32'else 1/6
    setd(ref(z),co,0)
    for cc in[.5,1.,1.]:
     assert l.mpfr_mul(ref(z),ref(z),ref(a),0)==0;setd(ref(c),cc,0);assert l.mpfr_add(ref(z),ref(z),ref(c),0)==0
    out.append(getter(ref(z),0))
   else:
    f=l.mpfr_div if op=='div'else l.mpfr_fmod
    f(ref(lo),ref(a),ref(b),3);f(ref(hi),ref(a),ref(b),2)
    x1=getter(ref(lo),0);x2=getter(ref(hi),0);assert bits(x1,ty)==bits(x2,ty),(group,x,y,x1,x2);out.append(x1)
 cache[key]=out;return out
root=Path(sys.argv[1]);result=[]
for inv in sorted(root.glob('outputs-*/inventory.tsv')):
 for group,band,name,ty,file in csv.reader(inv.open(),delimiter='\t'):
  rows=list(struct.iter_unpack('<fff'if ty=='f32'else'<ddd',(inv.parent/file).read_bytes()));expected=oracle(group,band,rows);observed=[r[2]for r in rows[:len(expected)]]
  err=[abs(ordered(bits(x,ty),ty)-ordered(bits(y,ty),ty))for x,y in zip(observed,expected)if math.isfinite(x)and math.isfinite(y)]
  result.append(dict(mode=inv.parent.name,group=group,band=band,candidate=name,outputs=len(expected),correctly_rounded=sum(bits(x,ty)==bits(y,ty)for x,y in zip(observed,expected)),nonfinite=sum(not math.isfinite(v)for v in observed),max_representable_steps=max(err),max_abs_from_rounded_ref=max(abs(x-y)for x,y in zip(observed,expected))))
with(root/'accuracy.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
print('Observed',len(result),'case/band/mode results; max steps',max(r['max_representable_steps']for r in result),'nonfinite',sum(r['nonfinite']for r in result));print('No universal algebraic bound or bit-parity assertion.')
