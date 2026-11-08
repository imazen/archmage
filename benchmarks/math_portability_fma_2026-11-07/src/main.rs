#![forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
use zenbench::{GateConfig, Suite};
#[allow(dead_code)]
mod kernels {
    #[cfg(not(target_arch = "wasm32"))]
    use super::{matches_filter, value};
    use archmage::SimdToken;
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}
#[allow(dead_code)]
mod fma_kernels {
    #[cfg(not(target_arch = "wasm32"))]
    use archmage::SimdToken;
    include!(concat!(env!("OUT_DIR"), "/fma_kernels.rs"));
}
#[cfg(not(target_arch = "wasm32"))]
fn matches_filter(id: &str) -> bool {
    std::env::var("STUDY_FILTER").map_or(true, |f| id.contains(&f))
}
fn value(i: usize, distribution: &str, op: usize) -> f32 {
    let bits = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
    let x = match distribution {
        "ordinary" => 0.125 + (i % 97) as f32 / 97.0,
        "bits" => f32::from_bits(0x3d000000 + bits % 0x04800000),
        "sparse" if i.is_multiple_of(64) => f32::from_bits(0x7f800001 | bits & 0x7fffff),
        "dense" if i.is_multiple_of(2) => f32::from_bits(0x7f800001 | bits & 0x7fffff),
        "sparse" | "dense" => 0.125 + (i % 97) as f32 / 97.0,
        _ => panic!("unknown distribution {distribution}"),
    };
    // Log and pow ordinary inputs are positive normal. Exp/activation also
    // exercise signs. The feedback graph is documented separately.
    if op < 8 && op != 2 && op != 5 && i.is_multiple_of(2) {
        -x
    } else {
        x
    }
}
fn fma_value_f32(i: usize, dist: &str) -> f32 {
    if dist != "bits" {
        return value(i, dist, 0);
    }
    let x = f32::from_bits((i as u32).wrapping_mul(747796405).wrapping_add(2891336453));
    if x.is_finite() { x } else { 1.0 }
}
fn fma_value_f64(i: usize, dist: &str) -> f64 {
    if dist != "bits" {
        return value(i, dist, 0) as f64;
    }
    let x = f64::from_bits(
        (i as u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(0xd1b54a32d192ed03),
    );
    if x.is_finite() { x } else { 1.0 }
}
fn oracle_value(i: usize) -> f32 {
    const SPECIAL: &[u32] = &[
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f000000, 0x3f800000, 0x3f800001,
        0x3f7fffff, 0xbf800000, 0x42ff0000, 0x43000000, 0xc2fc0000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001,
    ];
    if i < SPECIAL.len() {
        f32::from_bits(SPECIAL[i])
    } else if i.is_multiple_of(3) {
        f32::from_bits((i as u32).wrapping_mul(747796405).wrapping_add(2891336453))
    } else {
        (i as f32 - 512.0) / 8.0
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Ok(kind) = std::env::var("STUDY_FIRST_USE") {
        match kind.as_str() {
            "f32" => {
                let x = magetypes::simd::generic::f32x4::from_array_t(
                    archmage::ScalarToken,
                    std::hint::black_box([f32::from_bits(0x3f800800); 4]),
                );
                std::hint::black_box(
                    x.mul_add_fused(
                        x,
                        magetypes::simd::generic::f32x4::splat_t(
                            archmage::ScalarToken,
                            f32::from_bits(0xbf801000),
                        ),
                    )
                    .to_array(),
                );
            }
            "f64" => {
                let x = magetypes::simd::generic::f64x4::from_array_t(
                    archmage::ScalarToken,
                    std::hint::black_box([f64::from_bits(0x3ff0000002000000); 4]),
                );
                std::hint::black_box(
                    x.mul_add_fused(
                        x,
                        magetypes::simd::generic::f64x4::splat_t(
                            archmage::ScalarToken,
                            f64::from_bits(0xbff0000004000000),
                        ),
                    )
                    .to_array(),
                );
            }
            "empty" => {
                std::hint::black_box([0u64; 4]);
            }
            _ => panic!("unknown first-use diagnostic"),
        }
        return;
    }
    if let Ok(context) = std::env::var("STUDY_DUMP") {
        kernels::dump(&context);
        return;
    }
    if let Ok(context) = std::env::var("STUDY_VERIFY") {
        kernels::verify(&context);
        return;
    }
    if std::env::var_os("STUDY_ADMISSION").is_some() {
        let gate = zenbench::ResourceGate::new(GateConfig::strict());
        let mut clean = true;
        for _ in 0..10 {
            let reason = gate.check();
            println!("admission: {reason:?}");
            clean &= reason.is_none();
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        if !clean {
            std::process::exit(2);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::env::var_os("STUDY_DISABLE_V3").is_some() {
        use archmage::SimdToken;
        archmage::X64V3Token::dangerously_disable_token_process_wide(true)
            .expect("baseline build permits disabling V3");
        assert!(archmage::X64V3Token::summon().is_none());
        println!(
            "forced no-V3 fallback; hardware FMA present={}",
            std::is_x86_feature_detected!("fma")
        );
    }
    let mut suite = Suite::new();
    #[cfg(feature = "wasm-host")]
    if let Ok(path) = std::env::var("STUDY_WASM_HOST") {
        wasm_host::register(&mut suite, &path);
    }

    let contexts = std::env::var("STUDY_CONTEXTS")
        .unwrap_or_else(|_| "scalar,scalar-v3,v3,baseline-v3".into());
    if std::env::var_os("STUDY_WASM_HOST").is_none() {
        for context in contexts.split(',') {
            for n in [64, 4096] {
                for kind in ["stream", "chain"] {
                    for dist in ["ordinary", "bits", "sparse", "dense"] {
                        if std::env::var("STUDY_FMA").is_ok() {
                            fma_kernels::register(&mut suite, context, n, kind, dist);
                        } else {
                            kernels::register(&mut suite, context, n, kind, dist);
                        }
                    }
                }
            }
        }
    }
    let result = zenbench::run_gated(GateConfig::strict(), move |s| *s = suite);
    let destination = std::env::var("STUDY_OUTPUT").expect("required STUDY_OUTPUT");
    if destination == "-" {
        println!("STUDY_JSON {}", serde_json::to_string(&result).unwrap());
    } else {
        result.save(destination).unwrap();
    }
    for group in &result.comparisons {
        let clean = !result.unreliable
            && group.samples.len() >= 50
            && group.samples.iter().all(|s| s.gate_clean == Some(true));
        for arm in 1..group.benchmarks.len() {
            let ratios: Vec<f64> = group
                .samples
                .iter()
                .map(|s| s.compensated_ns[arm] as f64 / s.compensated_ns[0] as f64)
                .collect();
            let ci = zenbench::MeanCi::from_samples(&ratios, 100_000).expect("paired rounds");
            println!(
                "RATIO {} {} clean={} n={} R={} CI=[{},{}]",
                group.group_name,
                group.benchmarks[arm].name,
                clean,
                ratios.len(),
                ci.median,
                ci.lower,
                ci.upper
            );
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    std::hint::black_box(guest::guest_fma as extern "C" fn(i32, i32, i32, i32, i32) -> i64);
    std::hint::black_box(guest::guest_probe as extern "C" fn() -> i32);
    std::hint::black_box(guest::guest_prepare as extern "C" fn(i32, i32, i32));
    std::hint::black_box(guest::guest_run as extern "C" fn(i32, i32, i32, i32, i32) -> i64);
    if let Ok(context) = std::env::var("STUDY_DUMP") {
        kernels::dump(&context);
    } else {
        kernels::verify("wasm128");
    }
}

#[cfg(target_arch = "wasm32")]
mod guest;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasm-host"))]
mod wasm_host;
