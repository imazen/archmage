//! Opt-in mantissa scan. Fixed-size vectors; no corpus allocation.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, SimdToken};
use magemath::recip::*;
const LOW: u32 = RecipOptions::PORTABLE_LOWP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const MID: u32 = RecipOptions::PORTABLE_MIDP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const NLOW: u32 = RecipOptions::BACKEND_LOWP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const NMID: u32 = RecipOptions::BACKEND_MIDP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn scan(t: Token, stride: usize) -> [(f64, u32); 8] {
    use magetypes::simd::generic::f32x4 as V;
    let mut worst = [(0.0_f64, 0_u32); 8];
    // All binary32 mantissas in both exponent parity classes when stride=4.
    for start in (0x3f800000_u32..0x40800000).step_by(stride) {
        let input = core::array::from_fn(|i| f32::from_bits(start + i as u32));
        let x = V::from_array_t(t, input);
        let values = [
            try_recip_f32x4_with::<LOW, _>(t, x).unwrap(),
            try_rsqrt_f32x4_with::<LOW, _>(t, x).unwrap(),
            try_recip_f32x4_with::<MID, _>(t, x).unwrap(),
            try_rsqrt_f32x4_with::<MID, _>(t, x).unwrap(),
            try_recip_f32x4_with::<NLOW, _>(t, x).unwrap(),
            try_rsqrt_f32x4_with::<NLOW, _>(t, x).unwrap(),
            try_recip_f32x4_with::<NMID, _>(t, x).unwrap(),
            try_rsqrt_f32x4_with::<NMID, _>(t, x).unwrap(),
        ];
        for (j, vector) in values.into_iter().enumerate() {
            for (i, y) in vector.to_array().into_iter().enumerate() {
                let reference = if j % 2 == 0 {
                    1.0 / input[i] as f64
                } else {
                    1.0 / (input[i] as f64).sqrt()
                };
                let error = ((y as f64 - reference) / reference).abs();
                if error > worst[j].0 {
                    worst[j] = (error, input[i].to_bits());
                }
                assert!(
                    error
                        <= if j == 0 || j == 1 || j == 4 || j == 5 {
                            6e-3
                        } else {
                            1e-4
                        },
                    "profile {j} input {:08x} error {error}",
                    input[i].to_bits()
                );
            }
        }
    }
    worst
}
fn report(name: &str, values: [(f64, u32); 8]) {
    for (profile, (error, x)) in [
        "portable LOWP recip",
        "portable LOWP rsqrt",
        "portable MIDP recip",
        "portable MIDP rsqrt",
        "backend LOWP recip",
        "backend LOWP rsqrt",
        "backend MIDP recip",
        "backend MIDP rsqrt",
    ]
    .into_iter()
    .zip(values)
    {
        println!("{name} | {profile} | max relative error {error:.12e} | witness {x:08x}");
    }
}
fn main() {
    let stride = if std::env::args().any(|v| v == "--exhaustive") {
        4
    } else {
        4096
    };
    println!(
        "x4 scan stride={stride}; domain [1,4), f64 oracle; edge exponents are directed integration tests"
    );
    report("scalar", scan_scalar(ScalarToken, stride));
    #[cfg(target_arch = "x86_64")]
    report(
        "V3",
        scan_v3(archmage::X64V3Token::summon().expect("V3 required"), stride),
    );
    #[cfg(target_arch = "aarch64")]
    report(
        "NEON",
        scan_neon(
            archmage::NeonToken::summon().expect("NEON required"),
            stride,
        ),
    );
    #[cfg(target_arch = "wasm32")]
    report(
        "WASM",
        scan_wasm128(
            archmage::Wasm128Token::summon().expect("SIMD128 required"),
            stride,
        ),
    );
}
