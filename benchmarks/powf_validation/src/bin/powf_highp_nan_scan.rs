//! Exhaustive NaN-family classification, bounded to one f32 operand at a time.
#![allow(deprecated)] // Deliberately validates the retained Magetypes compatibility implementation.
#![forbid(unsafe_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use magetypes::powf::{PowfExecution, PowfOptions};
const FLAGS: u32 = PowfOptions::PORTABLE_HIGHP
    .with_execution(PowfExecution::Caller)
    .bits();
#[archmage::magetypes(v4x(cfg(avx512)), v3, neon, wasm128, scalar)]
fn scan(token: Token) -> (u64, u64) {
    use magetypes::simd::generic::f32x4;
    let mut families = 0u64;
    let mut mismatches = 0u64;
    for sign in [0u32, 0x8000_0000] {
        for payload in 1u32..0x0080_0000 {
            let nan = std::hint::black_box(f32::from_bits(sign | 0x7f80_0000 | payload));
            let signaling = payload & 0x0040_0000 == 0;
            let xs = f32x4::splat_t(token, nan);
            let zero = xs.try_powf_with::<FLAGS>(0.).unwrap().to_array();
            let ordinary = xs.try_powf_with::<FLAGS>(2.4).unwrap().to_array();
            let exp_nan = f32x4::from_array_t(token, [1., 0., -1., 2.])
                .try_powf_with::<FLAGS>(nan)
                .unwrap()
                .to_array();
            for value in zero {
                mismatches += u64::from(if signaling {
                    !value.is_nan()
                } else {
                    value.to_bits() != 1f32.to_bits()
                });
            }
            for value in ordinary {
                mismatches += u64::from(!value.is_nan());
            }
            for (i, value) in exp_nan.into_iter().enumerate() {
                mismatches += u64::from(if i == 0 && !signaling {
                    value.to_bits() != 1f32.to_bits()
                } else {
                    !value.is_nan()
                });
            }
            families += 1;
        }
    }
    (families, mismatches)
}
fn main() {
    let context = std::env::args()
        .nth(1)
        .expect("explicit scalar/native/native-v4x context");
    let (families, mismatches) = match context.as_str() {
        "scalar" => scan_scalar(archmage::ScalarToken),
        #[cfg(target_arch = "x86_64")]
        "native" => scan_v3(archmage::X64V3Token::summon().expect("required V3")),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "native-v4x" => scan_v4x(archmage::X64V4xToken::summon().expect("required V4x")),
        #[cfg(target_arch = "aarch64")]
        "native" => scan_neon(archmage::NeonToken::summon().expect("required NEON")),
        #[cfg(target_arch = "wasm32")]
        "native" => scan_wasm128(archmage::Wasm128Token::summon().expect("required SIMD128")),
        _ => panic!("unsupported context"),
    };
    println!(
        "{{\"context\":\"{context}\",\"nan_operand_bits\":{families},\"lane_checks\":{},\"classification_mismatches\":{mismatches}}}",
        families * 12
    );
    assert_eq!(families, 16_777_214);
    assert_eq!(mismatches, 0);
}
