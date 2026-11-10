"""Equal-work controlled Gamma loops; literal flags survive to feature kernels."""
from pathlib import Path
import sys

out = Path(sys.argv[1])
code = ''
for width in [4, 8, 16]:
    block = r'''
#[archmage::magetypes(v3,neon,scalar)]
#[inline(never)]
fn rounding_loop@N@<const ARM:usize,const EXP:u32,const CHECKED:bool>(
    token:Token,input:&[[f32;@N@]],output:&mut[[f32;@N@]],chain:bool,
) {
    use magetypes::simd::generic::f32x@N@ as V;
    let n=f32::from_bits(EXP);
    let mut acc=input[0];
    for (base,out) in input.iter().zip(output) {
        let x=if chain {acc} else {*base};
        let v=V::from_array_t(token,x);
        let y=match ARM {
            0=>archmage_powf_study::rounding::kernel@N@::<Token,false,false,CHECKED,2>(token,v,n).unwrap().to_array(),
            1=>archmage_powf_study::rounding::kernel@N@::<Token,false,true,CHECKED,2>(token,v,n).unwrap().to_array(),
            2=>archmage_powf_study::rounding::kernel@N@::<Token,true,false,CHECKED,2>(token,v,n).unwrap().to_array(),
            3=>archmage_powf_study::rounding::kernel@N@::<Token,true,true,CHECKED,2>(token,v,n).unwrap().to_array(),
            4=>v.pow_lowp(n).to_array(),
            5=>v.pow_midp(n).to_array(),
            6=>x.map(|x|x.powf(n)),
            _=>unreachable!(),
        };
        acc=if chain {y.map(normalize)} else {y};
        *out=acc;
    }
}
fn execute@N@<const EXP:u32,const CHECKED:bool>(arm:usize,context:&str,input:&[[f32;@N@]],output:&mut[[f32;@N@]],chain:bool){
    macro_rules! go {($f:ident,$t:expr)=>{match arm {
        0=>$f::<0,EXP,CHECKED>($t,input,output,chain),
        1=>$f::<1,EXP,CHECKED>($t,input,output,chain),
        2=>$f::<2,EXP,CHECKED>($t,input,output,chain),
        3=>$f::<3,EXP,CHECKED>($t,input,output,chain),
        4=>$f::<4,EXP,CHECKED>($t,input,output,chain),
        5=>$f::<5,EXP,CHECKED>($t,input,output,chain),
        6=>$f::<6,EXP,CHECKED>($t,input,output,chain),
        _=>unreachable!(),
    }}}
    match context {
        "scalar"=>go!(rounding_loop@N@_scalar,archmage::ScalarToken),
        #[cfg(target_arch="x86_64")]
        "v3"=>go!(rounding_loop@N@_v3,archmage::X64V3Token::summon().expect("required V3")),
        #[cfg(target_arch="aarch64")]
        "neon"=>go!(rounding_loop@N@_neon,archmage::NeonToken::summon().expect("required NEON")),
        _=>panic!("unsupported explicit context"),
    }
}
fn register@N@<const EXP:u32,const CHECKED:bool>(suite:Option<&mut zenbench::Suite>,cell:&Cell){
    let input:Vec<[f32;@N@]>=(0..cell.size/@N@).map(|i|core::array::from_fn(|j|(((i*@N@+j)%255)+1)as f32/255.)).collect();
    let mut output=vec![[0.;@N@];input.len()];
    for arm in 0..7 {
        execute@N@::<EXP,CHECKED>(arm,cell.context,&input,&mut output,cell.chain);
        assert!(output.iter().flatten().all(|v|v.is_finite()));
        if arm<4 {
            let mut other=output.clone();
            execute@N@::<EXP,true>(arm,cell.context,&input,&mut other,cell.chain);
            assert_eq!(output.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),other.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),"checked/assumed {}",cell.name);
        }
    }
    let Some(suite)=suite else{return;};
    suite.group(&cell.name,|g|{
        g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(120));
        for (arm,label) in LABELS.iter().enumerate(){
            let input=input.clone();let context=cell.context;let chain=cell.chain;
            g.bench(*label,move|b|{
                let mut output=vec![[0.;@N@];input.len()];
                execute@N@::<EXP,CHECKED>(arm,context,&input,&mut output,chain);
                b.iter(||execute@N@::<EXP,CHECKED>(arm,context,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));
                std::hint::black_box(output);
            });
        }
    });
}
fn dispatch@N@(suite:Option<&mut zenbench::Suite>,cell:&Cell){
    match (cell.exponent,cell.checked) {
        (0x4019999a,true)=>register@N@::<0x4019999a,true>(suite,cell),
        (0x4019999a,false)=>register@N@::<0x4019999a,false>(suite,cell),
        (0x3f000000,true)=>register@N@::<0x3f000000,true>(suite,cell),
        (0x3f000000,false)=>register@N@::<0x3f000000,false>(suite,cell),
        _=>unreachable!(),
    }
}
'''
    code += block.replace('@N@',str(width))
(out/'rounding_bench.rs').write_text(code)
