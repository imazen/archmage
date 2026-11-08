#![forbid(unsafe_code)]
#[cfg(not(target_arch="wasm32"))]
use zenbench::{GateConfig, Suite};
#[allow(dead_code)]
mod kernels {
    #[cfg(not(target_arch="wasm32"))]
    use super::{matches_filter, value};
    use archmage::SimdToken;
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}
#[cfg(not(target_arch="wasm32"))]
mod fma_kernels {
    use archmage::SimdToken;
    include!(concat!(env!("OUT_DIR"), "/fma_kernels.rs"));
}
#[cfg(not(target_arch="wasm32"))]
fn matches_filter(id: &str) -> bool {
    std::env::var("STUDY_FILTER").map_or(true, |f| id.contains(&f))
}
fn value(i:usize, distribution:&str, op:usize) -> f32 {
    let bits = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
    let x = match distribution {
        "ordinary" => 0.125 + (i%97) as f32 / 97.0,
        "bits" => f32::from_bits(0x3d000000 + bits % 0x04800000),
        "sparse" if i%64==0 => f32::from_bits(0x7f800001 | bits&0x7fffff),
        "dense" if i%2==0 => f32::from_bits(0x7f800001 | bits&0x7fffff),
        "sparse" | "dense" => 0.125 + (i%97) as f32/97.0,
        _ => panic!("unknown distribution {distribution}"),
    };
    // Log and pow ordinary inputs are positive normal. Exp/activation also
    // exercise signs. The feedback graph is documented separately.
    if op < 8 && op != 2 && op != 5 && i%2==0 { -x } else { x }
}
fn oracle_value(i:usize) -> f32 {
    const SPECIAL:&[u32]=&[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x3f000000,0x3f800000,0x3f800001,0x3f7fffff,0xbf800000,
        0x42ff0000,0x43000000,0xc2fc0000,0x7f7fffff,0xff7fffff,
        0x7f800000,0xff800000,0x7fc00000,0xffc00001,0x7f800001];
    if i<SPECIAL.len() {f32::from_bits(SPECIAL[i])}
    else if i%3==0 {f32::from_bits((i as u32).wrapping_mul(747796405).wrapping_add(2891336453))}
    else { (i as f32-512.0)/8.0 }
}
#[cfg(not(target_arch="wasm32"))]
fn main() {
    if let Ok(context) = std::env::var("STUDY_DUMP") { kernels::dump(&context); return; }
    if let Ok(context) = std::env::var("STUDY_VERIFY") {
        kernels::verify(&context);
        return;
    }
    if std::env::var_os("STUDY_ADMISSION").is_some() {
        let gate=zenbench::ResourceGate::new(GateConfig::strict());
        let mut clean=true;
        for _ in 0..10 { let reason=gate.check(); println!("admission: {reason:?}"); clean &= reason.is_none(); std::thread::sleep(std::time::Duration::from_millis(500)); }
        if !clean {std::process::exit(2);}
        return;
    }
    let mut suite=Suite::new();
    #[cfg(feature="wasm-host")]
    if let Ok(path)=std::env::var("STUDY_WASM_HOST") {
        wasm_host::register(&mut suite,&path);
    }

    let contexts=std::env::var("STUDY_CONTEXTS").unwrap_or_else(|_|"scalar,scalar-v3,v3,baseline-v3".into());
    if std::env::var_os("STUDY_WASM_HOST").is_none() {
    for context in contexts.split(',') {
        for n in [64,4096] { for kind in ["stream","chain"] {
            for dist in ["ordinary","bits","sparse","dense"] {
                if std::env::var("STUDY_FMA").is_ok() {
                    fma_kernels::register(&mut suite,context,n,kind,dist);
                } else {
                    kernels::register(&mut suite,context,n,kind,dist);
                }
            }
        }}
    }
    }
    let result=zenbench::run_gated(GateConfig::strict(), move |s| *s=suite);
    result.save(std::env::var("STUDY_OUTPUT").expect("required STUDY_OUTPUT")).unwrap();
    for group in &result.comparisons {
        let clean=!result.unreliable && group.samples.len()>=50 && group.samples.iter().all(|s|s.gate_clean==Some(true));
        for arm in 1..group.benchmarks.len() {
            let ratios:Vec<f64>=group.samples.iter().map(|s|s.compensated_ns[arm] as f64/s.compensated_ns[0] as f64).collect();
            let ci=zenbench::MeanCi::from_samples(&ratios,100_000).expect("paired rounds");
            println!("RATIO {} {} clean={} n={} R={} CI=[{},{}]", group.group_name, group.benchmarks[arm].name,clean,ratios.len(),ci.median,ci.lower,ci.upper);
        }
    }
}

#[cfg(target_arch="wasm32")]
fn main() {
    std::hint::black_box(guest::guest_probe as extern "C" fn()->i32);
    std::hint::black_box(guest::guest_prepare as extern "C" fn(i32,i32,i32));
    std::hint::black_box(guest::guest_run as extern "C" fn(i32,i32,i32,i32,i32)->i64);
    if let Ok(context)=std::env::var("STUDY_DUMP") {kernels::dump(&context);}
    else {kernels::verify("wasm128");}
}

#[cfg(all(not(target_arch="wasm32"),feature="wasm-host"))]
mod wasm_host;
#[cfg(target_arch="wasm32")]
mod guest;
