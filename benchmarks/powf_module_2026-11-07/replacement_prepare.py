"""Generate equal-work legacy/std/runtime/const consumer loops for the frozen study."""
from pathlib import Path
import sys
out=Path(sys.argv[1])
code=''
for n in [4,8,16]:
    code+=r"""
#[inline(always)]
fn prevalidated_call@N@<T:magetypes::simd::backends::F32x@N@Convert>(token:T,x:[f32;@N@],n:f32,domain:PowfDomain)->[f32;@N@]{
    macro_rules! call {($a:expr)=>{match domain {
        PowfDomain::PositiveNormal=>prevalidated::core@N@::<T,{$a},false,false>(token,x,n).unwrap(),
        PowfDomain::NonNegativeNormal=>prevalidated::core@N@::<T,{$a},true,false>(token,x,n).unwrap(),
        PowfDomain::Gamma=>prevalidated::core@N@::<T,{$a},true,true>(token,x,n).unwrap(),
    }}}
    if core::any::TypeId::of::<T>()==core::any::TypeId::of::<archmage::ScalarToken>(){call!(prevalidated::SOFTWARE)}else{call!(prevalidated::NATIVE)}
}
""".replace('@N@',str(n))
for width in [1,4,8,16]:
    block=r'''
#[inline(always)]
fn replacement_core@N@<T:magetypes::simd::backends::F32x@N@Convert,const ARM:usize,const CALLER:u32,const AUTO:u32,const EXP:u32,const DYNAMIC:bool>(
    token:T,input:&[[f32;@N@]],output:&mut[[f32;@N@]],chain:bool,dynamic_options:PowfOptions,
) {
    use magetypes::simd::generic::f32x@N@ as V;
    let n=if DYNAMIC {std::hint::black_box(f32::from_bits(EXP))} else {f32::from_bits(EXP)};
    let domain=match CALLER&3 {0=>PowfDomain::PositiveNormal,1=>PowfDomain::NonNegativeNormal,2=>PowfDomain::Gamma,_=>unreachable!()};
    let mut acc=input[0];
    for (base,out) in input.iter().zip(output) {
        let x=if chain {acc} else {*base};
        let v=V::from_array_t(token,x);
        let y=match ARM {
            0=>v.pow_midp().to_array(),
            1=>v.pow_midp_unchecked(n).to_array(),
            2=>v.pow_midp_precise(n).to_array(),
            3=>v.pow_lowp(n).to_array(),
            4=>v.pow_lowp_unchecked(n).to_array(),
            5=>x.map(|x|x.powf(n)),
            6=>x.map(|x|(f64::from(x).powf(f64::from(n)))as f32),
            7=>v.try_powf_runtime(n,PowfOptions{domain,execution:PowfExecution::Caller}).unwrap().to_array(),
            8=>v.try_powf_runtime(n,dynamic_options).unwrap().to_array(),
            9=>v.try_powf_runtime(n,PowfOptions{domain,execution:PowfExecution::Automatic}).unwrap().to_array(),
            10=>v.try_powf_runtime(n,dynamic_options).unwrap().to_array(),
            11=>v.try_powf_with::<CALLER>(n).unwrap().to_array(),
            12=>v.try_powf_with::<AUTO>(n).unwrap().to_array(),
            13=>prevalidated_call@N@(token,x,n,domain),
            14=>x.map(|x|x.powi(f32::from_bits(EXP)as i32)),
            _=>unreachable!(),
        };
        acc=if chain {y.map(normalize)} else {y}; *out=acc;
    }
}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
fn replacement_loop@N@<const ARM:usize,const CALLER:u32,const AUTO:u32,const EXP:u32,const DYNAMIC:bool>(
    token:Token,input:&[[f32;@N@]],output:&mut[[f32;@N@]],chain:bool,options:PowfOptions,
) { replacement_core@N@::<Token,ARM,CALLER,AUTO,EXP,DYNAMIC>(token,input,output,chain,options); }

fn replacement_execute@N@<const CALLER:u32,const AUTO:u32,const EXP:u32,const DYNAMIC:bool>(
    arm:usize,context:&str,input:&[[f32;@N@]],output:&mut[[f32;@N@]],chain:bool,
) {
    let domain=match CALLER&3 {0=>PowfDomain::PositiveNormal,1=>PowfDomain::NonNegativeNormal,2=>PowfDomain::Gamma,_=>unreachable!()};
    let options=std::hint::black_box(PowfOptions{domain,execution:if arm==10{PowfExecution::Automatic}else{PowfExecution::Caller}});
    macro_rules! go {($f:ident,$t:expr)=>{match arm {
        0=>$f::<0,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        1=>$f::<1,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        2=>$f::<2,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        3=>$f::<3,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        4=>$f::<4,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        5=>$f::<5,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        6=>$f::<6,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        7=>$f::<7,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        8=>$f::<8,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        9=>$f::<9,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        10=>$f::<10,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        11=>$f::<11,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        12=>$f::<12,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        13=>$f::<13,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        14=>$f::<14,CALLER,AUTO,EXP,DYNAMIC>($t,input,output,chain,options),
        _=>unreachable!(),
    }}}
    match context {
        "scalar"=>go!(replacement_loop@N@_scalar,archmage::ScalarToken),
        #[cfg(target_arch="x86_64")]
        "v3"=>go!(replacement_loop@N@_v3,archmage::X64V3Token::summon().expect("required V3")),
        #[cfg(all(target_arch="x86_64",feature="avx512"))]
        "v4"=>go!(replacement_loop@N@_v4,archmage::X64V4Token::summon().expect("required V4")),
        #[cfg(all(target_arch="x86_64",feature="avx512"))]
        "v4x"=>go!(replacement_loop@N@_v4x,archmage::X64V4xToken::summon().expect("required V4x")),
        #[cfg(target_arch="aarch64")]
        "neon"=>go!(replacement_loop@N@_neon,archmage::NeonToken::summon().expect("required NEON")),
        #[cfg(target_arch="wasm32")]
        "wasm128"=>go!(replacement_loop@N@_wasm128,archmage::Wasm128Token::summon().expect("required WASM SIMD")),
        other=>panic!("unsupported requested context {other}"),
    }
}

fn replacement_register@N@<const CALLER:u32,const AUTO:u32,const EXP:u32,const DYNAMIC:bool>(
    suite:Option<&mut zenbench::Suite>,cell:&Cell,
) {
    if !matches_filter(&cell.name){return;}
    let input:Vec<[f32;@N@]>=(0..cell.size/@N@).map(|i|core::array::from_fn(|j|value(i*@N@+j,cell.distribution))).collect();
    let mut expected=vec![[0.0;@N@];input.len()];
    let mut actual=expected.clone();
    replacement_execute@N@::<CALLER,AUTO,EXP,DYNAMIC>(13,cell.context,&input,&mut expected,cell.chain);
    let mut midp=None;
    let mut lowp=None;
    let count=if cell.integer{15}else{14};
    for arm in 0..count {
        replacement_execute@N@::<CALLER,AUTO,EXP,DYNAMIC>(arm,cell.context,&input,&mut actual,cell.chain);
        let bits=actual.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>();
        assert!(actual.iter().flatten().all(|x|x.is_finite()),"nonfinite ordinary output {} arm{arm}",cell.name);
        if arm==0{midp=Some(bits.clone());}
        if arm==2{assert_eq!(Some(&bits),midp.as_ref(),"midp_precise alias {}",cell.name);}
        if arm==3{lowp=Some(bits.clone());}
        if arm==4{assert_eq!(Some(&bits),lowp.as_ref(),"lowp_unchecked alias {}",cell.name);}
        if (7..=13).contains(&arm) {
            assert_eq!(bits,expected.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),"canonical graph {} arm{arm}",cell.name);
        }
    }
    let Some(suite)=suite else{return;};
    suite.group(&cell.name,|g|{
        g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(120));
        for (arm,label) in LABELS.iter().take(count).enumerate(){
            let input=input.clone();let context=cell.context;let chain=cell.chain;
            g.bench(*label,move|b|{
                let mut output=vec![[0.0;@N@];input.len()];
                replacement_execute@N@::<CALLER,AUTO,EXP,DYNAMIC>(arm,context,&input,&mut output,chain);
                b.iter(||replacement_execute@N@::<CALLER,AUTO,EXP,DYNAMIC>(arm,context,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));
                std::hint::black_box(output);
            });
        }
    });
}

fn replacement_dispatch@N@(suite:Option<&mut zenbench::Suite>,cell:&Cell){
    match (cell.domain,cell.exponent,cell.dynamic) {
@CONFIGS@
        _=>panic!("unregistered frozen configuration {}",cell.name),
    }
}
'''.replace('v.pow_midp().to_array()','v.pow_midp(n).to_array()')
    if width==1:
        block=block.replace('F32x@N@Convert','F32x4Convert').replace('f32x@N@ as V','f32x4 as V')
        block=block.replace('let v=V::from_array_t(token,x);','let v=V::splat_t(token,x[0]);')
        block=block.replace('5=>x.map(|x|x.powf(n)),','5=>x[0].powf(n),')
        block=block.replace('6=>x.map(|x|(f64::from(x).powf(f64::from(n)))as f32),','6=>(f64::from(x[0]).powf(f64::from(n)))as f32,')
        block=block.replace('7=>v.try_powf_runtime(n,PowfOptions{domain,execution:PowfExecution::Caller}).unwrap().to_array(),','7=>magetypes::powf::try_powf_runtime(x[0],n,PowfOptions{domain,execution:PowfExecution::Caller}).unwrap(),')
        block=block.replace('8=>v.try_powf_runtime(n,dynamic_options).unwrap().to_array(),','8=>magetypes::powf::try_powf_runtime(x[0],n,dynamic_options).unwrap(),')
        block=block.replace('9=>v.try_powf_runtime(n,PowfOptions{domain,execution:PowfExecution::Automatic}).unwrap().to_array(),','9=>magetypes::powf::try_powf_runtime(x[0],n,PowfOptions{domain,execution:PowfExecution::Automatic}).unwrap(),')
        block=block.replace('10=>v.try_powf_runtime(n,dynamic_options).unwrap().to_array(),','10=>magetypes::powf::try_powf_runtime(x[0],n,dynamic_options).unwrap(),')
        block=block.replace('11=>v.try_powf_with::<CALLER>(n).unwrap().to_array(),','11=>magetypes::powf::try_powf_with::<CALLER>(x[0],n).unwrap(),')
        block=block.replace('12=>v.try_powf_with::<AUTO>(n).unwrap().to_array(),','12=>magetypes::powf::try_powf_with::<AUTO>(x[0],n).unwrap(),')
        block=block.replace('13=>prevalidated_call@N@(token,x,n,domain),','13=>prevalidated_call4(token,[x[0];4],n,domain)[0],')
        block=block.replace('14=>x.map(|x|x.powi(f32::from_bits(EXP)as i32)),','14=>x[0].powi(f32::from_bits(EXP)as i32),')
        block=block.replace('.to_array(),','.to_array()[0],')
        block=block.replace('acc=if chain {y.map(normalize)} else {y}; *out=acc;','let y=[y]; acc=if chain {y.map(normalize)} else {y}; *out=acc;')
        block=block.replace('#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]','#[archmage::magetypes(scalar)]')
        a=block.index('    match context {'); b=block.index('\n}\n\nfn replacement_register',a)
        block=block[:a]+'    assert_eq!(context,"scalar"); go!(replacement_loop@N@_scalar,archmage::ScalarToken);'+block[b:]
    block=block.replace('\nfn replacement_register','\n#[cfg(not(target_arch="wasm32"))]\nfn replacement_register')
    block=block.replace('\nfn replacement_dispatch','\n#[cfg(not(target_arch="wasm32"))]\nfn replacement_dispatch')
    configs=[]
    for domain in range(3):
        for exp in [0x4019999a,0x3ed55555,0x3f000000]:
            configs.append((domain,exp,False))
    configs += [(0,0x3e232000,False),(0,0x429db000,False),(2,0x4019999a,True),(2,0x40000000,False),(2,0x40800000,False)]
    lines='\n'.join(f'        ({d},0x{e:08x},{str(dynamic).lower()})=>replacement_register{width}::<{d|4},{d},0x{e:08x},{str(dynamic).lower()}>(suite,cell),' for d,e,dynamic in configs)
    code+=block.replace('@N@',str(width)).replace('@CONFIGS@',lines)
(out/'replacement_kernels.rs').write_text(code)
