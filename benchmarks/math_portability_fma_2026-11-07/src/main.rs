#![forbid(unsafe_code)]
use zenbench::{GateConfig, Suite};
mod kernels {
    use super::{matches_filter, value};
    use archmage::SimdToken;
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}
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
fn main() {
    if let Ok(context) = std::env::var("STUDY_VERIFY") {
        kernels::verify(&context);
        return;
    }
    let mut suite=Suite::new();
    let contexts=std::env::var("STUDY_CONTEXTS").unwrap_or_else(|_|"scalar,scalar-v3,v3,baseline-v3".into());
    for context in contexts.split(',') {
        for n in [64,4096] { for kind in ["stream","chain"] {
            for dist in ["ordinary","bits","sparse","dense"] {
                kernels::register(&mut suite,context,n,kind,dist);
            }
        }}
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
