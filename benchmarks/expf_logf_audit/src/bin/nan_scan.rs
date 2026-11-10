//! Exhaustive NaN classification, bounded arrays and no retained corpus allocation.
#![forbid(unsafe_code)]
use archmage::SimdToken;
use magetypes::{expf, logf};

#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn method(token: Token, xs: [f32; 8]) -> [f32; 8] {
    magetypes::simd::generic::f32x8::from_array_t(token, xs)
        .log2_midp()
        .to_array()
}
fn native(xs: [f32; 8]) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    return method_v3(archmage::X64V3Token::summon().expect("V3 required"), xs);
    #[cfg(target_arch = "aarch64")]
    return method_neon(archmage::NeonToken::summon().expect("NEON required"), xs);
    #[cfg(target_arch = "wasm32")]
    return method_wasm128(archmage::Wasm128Token::summon().expect("SIMD required"), xs);
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    return method_scalar(archmage::ScalarToken, xs);
}
fn main() {
    let software = std::env::var_os("NAN_SCAN_SOFTWARE").is_some();
    let labels = [
        "legacy-method",
        "configured-log-Specials-Checked",
        "configured-exp-Full-Checked",
    ];
    let mut checked = 0u64;
    let mut failures = [0u64; 3];
    let mut witnesses = Vec::new();
    for sign in [0, 0x80000000] {
        let mut payload = 1u32;
        while payload <= 0x007fffff {
            let count = (0x00800000 - payload).min(8) as usize;
            let bits: [u32; 8] =
                core::array::from_fn(|i| sign | 0x7f800000 | (payload + i as u32).min(0x007fffff));
            let xs = std::hint::black_box(bits.map(f32::from_bits));
            let outputs = if software {
                [
                    method_scalar(archmage::ScalarToken, xs),
                    logf::try_log2_array_with::<5, 8>(xs).unwrap(),
                    expf::try_exp2_array_with::<5, 8>(xs).unwrap(),
                ]
            } else {
                [
                    native(xs),
                    logf::try_log2_array_with::<1, 8>(xs).unwrap(),
                    expf::try_exp2_array_with::<1, 8>(xs).unwrap(),
                ]
            };
            for (op, values) in outputs.iter().enumerate() {
                for lane in 0..count {
                    if !values[lane].is_nan() {
                        failures[op] += 1;
                        if witnesses.len() < 16 {
                            witnesses.push(serde_json::json!({"operation":labels[op],"input":format!("{:08x}",bits[lane]),"actual":format!("{:08x}",values[lane].to_bits())}));
                        }
                    }
                }
            }
            checked += count as u64;
            payload += count as u32;
        }
    }
    assert_eq!(checked, 16777214);
    println!(
        "{}",
        serde_json::json!({"mode":if software{"ScalarToken Caller"}else{"native method / Automatic configured"},"distinct_NaNs":checked,"labels":labels,"non_NaN_results":failures,"witnesses":witnesses,"array_lanes":8})
    );
    assert_eq!(failures, [0; 3]);
}
