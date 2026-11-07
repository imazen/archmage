//! Negation flips exactly the sign bit, including zero and NaN payloads.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, SimdToken};
use std::hint::black_box;

const F32_BITS: &[u32] = &[
    0,
    0x8000_0000,
    1,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0123,
    0xffc0_0456,
    0x7f80_0001,
    0xff80_0001,
];
const F64_BITS: &[u64] = &[
    0,
    0x8000_0000_0000_0000,
    1,
    0x8000_0000_0000_0001,
    0x000f_ffff_ffff_ffff,
    0x0010_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0123,
    0xfff8_0000_0000_0456,
    0x7ff0_0000_0000_0001,
    0xfff0_0000_0000_0001,
];
macro_rules! check {
    ($fn:ident, $v:ident, $bound:ident, $n:literal, $float:ident, $uint:ident, $bits:ident) => {
        fn $fn<T: magetypes::simd::backends::$bound>(t: T) {
            for offset in 0..$bits.len() {
                let a: [$float; $n] =
                    core::array::from_fn(|i| $float::from_bits($bits[(offset + i) % $bits.len()]));
                let va = magetypes::simd::generic::$v::from_array_t(t, black_box(a));
                let neg = (-va).to_array();
                let twice = (-black_box(-va)).to_array();
                let sign = (1 as $uint) << ($uint::BITS - 1);
                for i in 0..$n {
                    assert_eq!(
                        neg[i].to_bits(),
                        a[i].to_bits() ^ sign,
                        "{} {} lane {i}: negate {:x}",
                        T::NAME,
                        stringify!($v),
                        a[i].to_bits()
                    );
                    assert_eq!(twice[i].to_bits(), a[i].to_bits());
                }
            }
        }
    };
}
check!(x4, f32x4, F32x4Backend, 4, f32, u32, F32_BITS);
check!(x8, f32x8, F32x8Backend, 8, f32, u32, F32_BITS);
check!(d2, f64x2, F64x2Backend, 2, f64, u64, F64_BITS);
check!(d4, f64x4, F64x4Backend, 4, f64, u64, F64_BITS);
#[cfg(feature = "w512")]
check!(x16, f32x16, F32x16Backend, 16, f32, u32, F32_BITS);
#[cfg(feature = "w512")]
check!(d8, f64x8, F64x8Backend, 8, f64, u64, F64_BITS);
macro_rules! tier {
    ($name:ident, $token:ty) => {
        #[test]
        fn $name() {
            let token = <$token>::summon();
            if stringify!($name) == "scalar" || cfg!(not(target_arch = "x86_64")) {
                assert!(
                    token.is_some(),
                    "required backend {} is unavailable",
                    stringify!($name)
                );
            }
            if let Some(t) = token {
                x4(t);
                x8(t);
                d2(t);
                d4(t);
                #[cfg(feature = "w512")]
                {
                    x16(t);
                    d8(t);
                }
            }
        }
    };
}
tier!(scalar, ScalarToken);
#[cfg(target_arch = "x86_64")]
tier!(v3, archmage::X64V3Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
tier!(v4, archmage::X64V4Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
tier!(v4x, archmage::X64V4xToken);
#[cfg(target_arch = "aarch64")]
tier!(neon, archmage::NeonToken);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
tier!(wasm, archmage::Wasm128Token);

// Named no-inline feature body for inspecting the generated negation instruction.
#[archmage::attune(_v3_t, inline(never), define(f32x8))]
fn neg8_codegen(input: [f32; 8]) -> [f32; 8] {
    let t = Token::from_context();
    (-f32x8::from_array_t(t, input)).to_array()
}

#[cfg(target_arch = "x86_64")]
#[test]
fn neg8_codegen_contract() {
    if let Some(t) = archmage::X64V3Token::summon() {
        let input = core::array::from_fn(|i| f32::from_bits(F32_BITS[i]));
        let output = neg8_codegen_v3_t(t, black_box(input));
        assert_eq!(
            output.map(f32::to_bits),
            input.map(|v| v.to_bits() ^ 0x8000_0000)
        );
    }
}
