#!/usr/bin/env python3
"""Migration-only extraction; not the normal regeneration command.

Magemath src/legacy_impl.rs becomes canonical after this import. Remove this
extractor when the deprecated Magetypes implementations are retired.

Move the actual 35 Magetypes bodies into one width-parameterized template.

Only proof/representation access and self-calls change. Source arithmetic stays
in its original order. All three generated widths must normalize identically.
"""
from pathlib import Path
import hashlib
import json
import re

root=Path(__file__).resolve().parents[2]
out=root/'magemath/src'
names=None
bodies={}
for width in [4,8,16]:
    source=(root/f'magetypes/src/simd/generic/generated/transcendentals_f32x{width}.rs').read_text()
    methods=[]
    for m in re.finditer(r'^    pub fn (\w+)\(self(, n: f32)?\) -> Self \{',source,re.M):
        end=source.index('\n    }',m.end())
        body=source[m.end():end]
        methods.append((m[1],bool(m[2]),body))
    assert len(methods)==35,(width,len(methods))
    if names is None:names=[x[0]for x in methods]
    assert [x[0]for x in methods]==names
    normalized=[]
    for name,argument,body in methods:
        s=body.replace(f'f32x{width}','FloatVec').replace(f'i32x{width}','IntVec').replace(f'F32x{width}Backend','FloatBackend').replace(f'I32x{width}Backend','IntBackend')
        s=re.sub(r'\[f32;\s*'+str(width)+r'\]', '[f32; $lanes]',s)
        # Comments/formatting are not part of the arithmetic equivalence gate.
        tokens=re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))
        tokens=re.sub(r',(?=\))','',tokens)  # rustfmt trailing argument commas
        normalized.append((name,argument,tokens))
    bodies[width]=(methods,normalized)
assert bodies[4][1]==bodies[8][1]==bodies[16][1],'width source arithmetic differs: inspect before migrating'
known=set(names)

def replace_calls(s):
    pattern=re.compile(r'\.('+ '|'.join(sorted(known,key=len,reverse=True))+r')\(')
    while m:=pattern.search(s):
        end=m.end();depth=1
        while depth:
            depth+=(s[end]=='(')-(s[end]==')');end+=1
        args=s[m.end():end-1]
        last=m.start()-1
        while s[last].isspace():last-=1
        if s[last]==')':
            start=last;depth=1
            while depth:
                start-=1;depth+=(s[start]==')')-(s[start]=='(')
        else:
            start=last
            while start>0 and (s[start-1].isalnum()or s[start-1]=='_'):start-=1
        receiver=s[start:last+1]
        if receiver.startswith('(')and receiver.endswith(')'):receiver=receiver[1:-1]
        replacement=f'{m[1]}::<T>(token, {receiver}'+(', '+args if args.strip()else '')+')'
        s=s[:start]+replacement+s[end:]
    return s

header='''//! Exact legacy backend rounding and specials, moved from Magetypes.\n//! Canonical Magemath arithmetic, initially imported from the 35 legacy bodies.\n//! No scalarization: each width uses its original vector operations.\nmacro_rules! legacy_bodies {\n($float:ident,$int:ident,$float_backend:ident,$convert:ident,$int_backend:ident,$lanes:literal)=>{\nuse magetypes::simd::backends::$convert as FloatConvert;\nuse magetypes::simd::generic::{$float as FloatVec,$int as IntVec};\n#[inline(always)]fn splat_i32<T:FloatConvert>(token:T,v:i32)->IntVec<T>{IntVec::splat_t(token,v)}\n#[inline(always)]fn splat_f32<T:FloatConvert>(token:T,v:f32)->FloatVec<T>{FloatVec::splat_t(token,v)}\n'''
rows=[]
for name,argument,original in bodies[8][0]:
    body=original.replace('f32x8','FloatVec').replace('i32x8','IntVec').replace('F32x8Backend','FloatBackend').replace('I32x8Backend','IntBackend')
    body=re.sub(r'\[f32;\s*8\]', '[f32; $lanes]',body)
    body=replace_calls(body)
    body=body.replace('self.1','token')
    body=re.sub(r'\bself\b','x',body).replace('Self::','FloatVec::')
    body=re.sub(r'FloatVec::from_repr_unchecked\(\s*token,\s*<T as FloatBackend>::from_array\(\s*token,\s*approx_arr\s*\)\s*,?\s*\)', 'FloatVec::from_array_t(token, approx_arr)',body)
    assert 'from_repr_unchecked' not in body,name
    assert not re.search(r'\.(?:'+ '|'.join(known)+r')\(',body),name
    header+='\n#[inline(always)]\npub(crate) fn '+name+'<T:FloatConvert>(token:T,x:FloatVec<T>'+(',n:f32'if argument else '')+')->FloatVec<T>{'+body+'\n}\n'
    rows.append(dict(name=name,exponent=argument,body_sha256=hashlib.sha256(next(s for n,a,s in bodies[8][1]if n==name).encode()).hexdigest()))
header+='};}\n'
for width in [4,8,16]:
    if width==16:header+='#[cfg(feature="w512")]\n'
    header+=f'pub(crate) mod width{width} {{legacy_bodies!(f32x{width},i32x{width},F32x{width}Backend,F32x{width}Convert,I32x{width}Backend,{width});}}\n'
(out/'legacy_impl.rs').write_text(header)
(root/'magemath/docs/legacy-inventory.json').write_text(json.dumps(dict(source='Magetypes generated bodies from draft 4ccaf1cf; deprecation attributes do not change the normalized gate',widths=[4,8,16],methods=rows),indent=2)+'\n')
print('PASS actual 35 methods x 3 width-equivalent arithmetic bodies; one private template')


import subprocess,sys
subprocess.run([sys.executable,str(root/"magemath/tools/generate-legacy.py")],check=True)
