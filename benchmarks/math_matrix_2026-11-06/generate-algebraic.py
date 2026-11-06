from pathlib import Path
import re
p=Path(__file__).resolve().parent
src=['//! Algebraic experiments only: no portable-bit or ULP contract.','use archmage::prelude::*;','use crate::Case;']
regs=[]
for ty,w in [('f32',8),('f64',4)]:
    vt=f'{ty}x{w}'
    bodies={
    'dot_ordinary':f'let mut sum=0.0; for (&a,&b) in x.iter().zip(y){{sum+=a*b}} out[0]=sum;',
    'dot_algebraic':f'let mut sum=0.0_{ty}; for (&a,&b) in x.iter().zip(y){{sum=sum.algebraic_add(a.algebraic_mul(b))}} out[0]=sum;',
    'poly_ordinary':'for (o,&a) in out.iter_mut().zip(x){*o=((0.125*a+0.25)*a+0.5)*a+1.0;}',
    'poly_algebraic':f'for (o,&a) in out.iter_mut().zip(x){{*o=0.125_{ty}.algebraic_mul(a).algebraic_add(0.25).algebraic_mul(a).algebraic_add(0.5).algebraic_mul(a).algebraic_add(1.0);}}',
    'poly_vector':f'for (o,a) in out.as_chunks_mut::<{w}>().0.iter_mut().zip(x.as_chunks::<{w}>().0.iter()){{let a={vt}::load_t(t,a);let c=|v|{vt}::splat_t(t,v);(((c(0.125)*a+c(0.25))*a+c(0.5))*a+c(1.0)).store(o);}}',
    'poly_array_algebraic':f'for (o,a) in out.as_chunks_mut::<{w}>().0.iter_mut().zip(x.as_chunks::<{w}>().0.iter()){{let a={vt}::load_t(t,a).to_array();let r=a.map(|a|0.125_{ty}.algebraic_mul(a).algebraic_add(0.25).algebraic_mul(a).algebraic_add(0.5).algebraic_mul(a).algebraic_add(1.0));{vt}::from_array_t(t,r).store(o);}}',
    'dot_vector':f'let mut sum={vt}::splat_t(t,0.);for (a,b) in x.as_chunks::<{w}>().0.iter().zip(y.as_chunks::<{w}>().0.iter()){{sum+={vt}::load_t(t,a)*{vt}::load_t(t,b);}}out[0]=sum.reduce_add();',
    'dot_array_algebraic':f'let mut sum=[0.0_{ty};{w}];for (a,b) in x.as_chunks::<{w}>().0.iter().zip(y.as_chunks::<{w}>().0.iter()){{let a={vt}::load_t(t,a).to_array();let b={vt}::load_t(t,b).to_array();sum=core::array::from_fn(|i|sum[i].algebraic_add(a[i].algebraic_mul(b[i])));}}out[0]=sum.into_iter().fold(0.0_{ty},|s,a|s.algebraic_add(a));',
    }
    bodies['poly_vector_fused']=bodies['poly_vector'].replace('(((c(0.125)*a+c(0.25))*a+c(0.5))*a+c(1.0))','c(0.125).mul_add_fused(a,c(0.25)).mul_add_fused(a,c(0.5)).mul_add_fused(a,c(1.0))')
    bodies['dot_vector_fused']=bodies['dot_vector'].replace('sum+=','sum=').replace('*'+vt+'::load_t(t,b);',' .mul_add_fused('+vt+'::load_t(t,b),sum);')
    for name,body in bodies.items():
        # Cubic Taylor approximation of exp on [-0.5,0.5], coefficients 1/6,1/2,1,1.
        # Non-power-of-two leading coefficient; this measures arithmetic, not an exp accuracy claim.
        if name.startswith('poly'): body=re.sub(r'0\.125|0\.25|0\.5|1\.0',lambda m:{'0.125':('0.16666667' if ty=='f32' else '0.16666666666666666'),'0.25':'0.5','0.5':'1.0','1.0':'1.0'}[m[0]],body)
        fn=f'{name}_{ty}'
        src.append(f'#[attune(_*_t,+v4x(cfg(avx512)),define({vt}))]\npub fn {fn}(t:Token,x:&[{ty}],y:&[{ty}],out:&mut[{ty}]){{assert_eq!(x.len(),y.len());assert_eq!(x.len(),out.len());assert_eq!(x.len()%{w},0);let _=t;let _=y;{body}}}')
        for tier,tok,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','all(target_arch="wasm32",target_feature="simd128")')]:
            pre=f'#[cfg({cfg})]' if cfg else ''
            regs.append((ty,f'{pre} if let Some(t)={tok}::summon(){{v.push(Case{{op:"{name}",label:"{name}/{tier}",run:Box::new(move|x,y,o|{fn}_{tier}_t(t,x,y,o))}});}}'))
for ty in ['f32','f64']:
    src.append(f'pub fn cases_{ty}()->Vec<Case<{ty}>>{{let mut v=Vec::new();'+''.join(s for t,s in regs if t==ty)+'v}')
(p/'src/algebraic.rs').write_text('\n'.join(src))
