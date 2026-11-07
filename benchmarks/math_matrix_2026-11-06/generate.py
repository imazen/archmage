#!/usr/bin/env python3
"""Generate independent measurement kernels; never edits production sources."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parent
ops=['powf','pow_fixed','exp','ln','log2','log10','sqrt','floor','ceil','round','roundeven']
core=dict(powf='powf',pow_fixed='powf',exp='expf',ln='logf',log2='log2f',log10='log10f',sqrt='sqrtf',floor='floorf',ceil='ceilf',round='roundf',roundeven='roundevenf')
src=['use archmage::prelude::*;\nuse crate::Case;']
regs=[]; rows=[]
def add(op,label,expr,width=1,ty='f32',vector=False,context=False,buffer=False):
    name=f'k{len(rows)}'; n=width
    rows.append(dict(operation=op,candidate=label,width=n,dtype=ty,source_expression=expr))
    if not vector and not context and not buffer:
        body=f'for ((o,&a),&b) in out.iter_mut().zip(x).zip(y) {{ let _=b; *o = {expr}; }}'
        src.append(f'fn {name}(x: &[{ty}], y: &[{ty}], out: &mut [{ty}]) {{ {body} }}')
        regs.append(f'v.push(Case {{ op:"{op}", label:"{label}/baseline", run:Box::new({name}) }});')
    else:
        vt=f'{ty}x{width}'
        body=f'for ((o,a),b) in out.as_chunks_mut::<{n}>().0.iter_mut().zip(x.as_chunks::<{n}>().0.iter()).zip(y.as_chunks::<{n}>().0.iter()) {{ let a={vt}::load_t(t,a); let b={vt}::load_t(t,b); let _=b; let r={expr}; r.store(o); }}'
        body=f'assert_eq!(x.len(),y.len());assert_eq!(x.len(),out.len());assert_eq!(x.len()%{n},0);'+body
        if context: body=f'for ((o,&a),&b) in out.iter_mut().zip(x).zip(y) {{ let _=b; *o = {expr}; }}'
        if buffer: body='assert_eq!(x.len(),y.len());assert_eq!(x.len(),out.len());'+expr
        define=f',define({vt})' if vector and not buffer else ''
        src.append(f'#[attune(_*_t,+v4x(cfg(avx512)){define})]\nfn {name}(t:Token,x:&[{ty}],y:&[{ty}],out:&mut[{ty}]) {{ let _=t; let _=y; {body} }}')
        for tier,token,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','all(target_arch="wasm32",target_feature="simd128")')]:
            pre=f'#[cfg({cfg})]' if cfg else ''
            regs.append(f'{pre} if let Some(t)={token}::summon() {{ v.push(Case{{op:"{op}",label:"{label}/{tier}/x{n}",run:Box::new(move |x,y,o| {name}_{tier}_t(t,x,y,o))}}); }}')
for op in ops:
    powop=op.startswith('pow'); arg='2.4' if op=='pow_fixed' else 'b'
    method='powf' if powop else ('round_ties_even' if op=='roundeven' else op)
    add(op,'std',f'a.{method}({arg if powop else ""})')
    add(op,'std_loop_context',f'a.{method}({arg if powop else ""})',context=True)
    add(op,'zenmath_correct_loop_context',f'zenmath::scalar::{core[op]}(a{", "+arg if powop else ""})',context=True)
    add(op,'zenmath_correct',f'zenmath::scalar::{core[op]}(a{", "+arg if powop else ""})')
    lm=core[op];lm='rintf' if op=='roundeven' else lm
    add(op,'libm',f'libm::{lm}(a{", "+arg if powop else ""})')
    if op!='roundeven':
        fm='pow' if powop else op
        add(op,'fpmath',f'fpmath::{fm}(a{", "+arg if powop else ""})')
        add(op,'micromath',f'micromath::F32(a).{method}({"micromath::F32("+arg+")" if powop else ""}).0')
    # metallic exposes round/sqrt and all transcendentals; floor/ceil are absent.
    if op not in ['floor','ceil','roundeven','sqrt']:
        add(op,'metallic',f'metallic::{core[op]}(a{", "+arg if powop else ""})')
    if op in ['exp','ln','log2','log10','powf','pow_fixed']:
        approx='pow' if powop else op
        for precision in ['midp','lowp']:
            for surface,prefix in [('portable',''),('unfused','unfused::')]:
                add(op,f'zenmath_{surface}_{precision}',f'zenmath::scalar::{prefix}{approx}_{precision}_f32(a{", "+arg if powop else ""})')
            for surface in ['portable','unfused','relaxed']:
                if op!='powf':
                    add(op,f'magemath_{surface}_{precision}',f'magemath::{surface}::kernels::{approx}_{precision}(t,a{", 2.4" if powop else ""})',8,vector=True)
        if op in ['exp','ln','log2','powf','pow_fixed']:
            mm='pow' if powop else op
            args=', y' if op=='powf' else ', &vec![2.4; x.len()]' if op=='pow_fixed' else ''
            expr=f'out.copy_from_slice(x); archmage::attuned!(magemath::portable::slice::{mm}_f32(out{args}));'
            # Fixed exponents are already supplied by the shared input generator.
            expr=expr.replace('&vec![2.4; x.len()]','y')
            add(op,'magemath_correct_proof',expr,8,vector=True,buffer=True)
            name=f'k{len(rows)}';rows.append(dict(operation=op,candidate='magemath_correct_dispatch',width=8,dtype='f32',source_expression=expr))
            src.append(f'fn {name}(x:&[f32],y:&[f32],out:&mut[f32]) {{let _=y;out.copy_from_slice(x); magemath::portable::slice::{mm}_f32(out{args.replace("&vec![2.4; x.len()]","y")});}}')
            regs.append(f'v.push(Case{{op:"{op}",label:"magemath_correct_dispatch/runtime/x8",run:Box::new({name})}});')
        if op=='powf':
            for p in ['MIDP','LOWP']:
                add(op,f'magemath_policy_{p.lower()}',f'out.copy_from_slice(x); archmage::attuned!(magemath::portable::slice::pow_f32_with::<{{magemath::PowBehavior::{p}.to_u64()}}>(out,y));',8,vector=True,buffer=True)
    elif op!='round':
        for surface in ['portable','unfused','relaxed']:
            add(op,f'magemath_{surface}',f'magemath::{surface}::kernels::{op}(t,a)',8,vector=True)
    for w in [4,8,16]:
        if op=='round':continue # magetypes round is ties-even, recorded separately.
        if op=='powf':continue # no packed per-lane exponent API on magetypes.
        if op in ['sqrt','floor','ceil','roundeven']:
            add(op,'magetypes_default',f'a.{"round" if op=="roundeven" else op}()',w,vector=True)
        else:
            mm='pow' if powop else op
            for p in ['midp','lowp']:
                add(op,f'magetypes_{p}',f'a.{mm}_{p}({"2.4" if powop else ""})',w,vector=True)
            if not powop:
                add(op,'magetypes_midp_precise',f'a.{mm}_midp_precise()',w,vector=True)
# f64 log2 baseline and available vector lowp.
for label,e in [('std','a.log2()'),('zenmath_correct','zenmath::scalar::log2(a)'),('libm','libm::log2(a)'),('fpmath','fpmath::log2(a)'),('metallic','metallic::log2(a)'),('zenmath_lowp','zenmath::scalar::log2_lowp_f64(a)')]:
    add('log2_f64',label,e,ty='f64')
    if label in ['std','zenmath_correct']: add('log2_f64',label+'_loop_context',e,ty='f64',context=True)
# No public generic f64 vector log2 method at this revision (compile verified).
src.append('pub fn cases32()->Vec<Case<f32>> { let mut v=Vec::new();'+''.join(r for r in regs if '"log2_f64"' not in r)+'v }')
src.append('pub fn cases64()->Vec<Case<f64>> { let mut v=Vec::new();'+''.join(r for r in regs if '"log2_f64"' in r)+'v }')
(root/'src/generated.rs').write_text('\n'.join(src)+'\n')
with (root/'api-matrix.csv').open('w') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('kernel families',len(rows))
