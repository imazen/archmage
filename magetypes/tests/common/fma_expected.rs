//! Independent rounding oracles shared by arithmetic and documentation tests.
#![forbid(unsafe_code)]
use archmage::SimdToken;

pub trait FmaExpected: Copy {
    /// One rounding: what `mul_add_portable` returns on every backend.
    fn fused_expected(self, b: Self, c: Self) -> Self;
    /// What `mul_add` returns for `token`: one rounding with hardware FMA (x86,
    /// NEON), the engine's choice with relaxed WASM, and two roundings on the
    /// scalar backend and strict WASM.
    fn mul_add_expected<T: SimdToken>(self, b: Self, c: Self, token: T) -> Self;
}

macro_rules! oracle {
    ($float:ty, $raw:ident) => {
        impl FmaExpected for $float {
            fn fused_expected(self, b: Self, c: Self) -> Self {
                // A nonzero exact product plus zero needs just one multiply.
                // This preserves negative underflow zero, which Rust 1.98.1's
                // WASM std f64::mul_add loses for min_subnormal * -min_subnormal.
                // Exclude zero multiplicands: adding opposite signed zeros
                // follows addition's sign rules, not multiplication's.
                if c == 0.0 && self != 0.0 && b != 0.0 {
                    self * b
                } else {
                    self.mul_add(b, c)
                }
            }
            fn mul_add_expected<T: SimdToken>(self, b: Self, c: Self, _token: T) -> Self {
                use core::any::TypeId;
                if TypeId::of::<T>() == TypeId::of::<archmage::ScalarToken>() {
                    return self * b + c;
                }
                #[cfg(target_arch = "wasm32")]
                if TypeId::of::<T>() == TypeId::of::<archmage::Wasm128Token>() {
                    #[cfg(target_feature = "relaxed-simd")]
                    return relaxed::$raw(self, b, c);
                    #[cfg(not(target_feature = "relaxed-simd"))]
                    return self * b + c;
                }
                self.fused_expected(b, c)
            }
        }
    };
}
oracle!(f32, expected32);
oracle!(f64, expected64);

#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
mod relaxed {
    use core::arch::wasm32::*;

    // The test build itself enables relaxed SIMD. Use its raw instruction as
    // the engine oracle, independently of magetypes' vector implementation.
    pub(super) fn expected32(a: f32, b: f32, c: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_relaxed_madd(
            f32x4_splat(a),
            f32x4_splat(b),
            f32x4_splat(c),
        ))
    }
    pub(super) fn expected64(a: f64, b: f64, c: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_relaxed_madd(
            f64x2_splat(a),
            f64x2_splat(b),
            f64x2_splat(c),
        ))
    }
}
