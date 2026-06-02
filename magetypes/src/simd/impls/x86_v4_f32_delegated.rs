//! W128 / W256 f32 backend delegation for `X64V4Token` / `X64V4xToken` /
//! `Avx512Fp16Token`.
//!
//! ## Why this file exists
//!
//! `X64V4Token` and friends advertise AVX-512 (W512) support but do not
//! ship their own implementations of the narrower [`F32x4Backend`] /
//! [`F32x8Backend`] traits. Without these, generic code parameterized on
//! `T: F32x8Backend` cannot accept a V4 token, and the cross-width
//! [`F32x16FromHalves`](crate::simd::generic::F32x16FromHalves) primitive
//! cannot accept a V4 token either — even though V4's native AVX-512
//! `_mm512_insertf32x8` is the intended optimal path.
//!
//! ## Deductive safety proof
//!
//! `X64V4Token`'s feature set is, per the registry:
//!
//!   `V4 = {sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b,`
//!         `avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe,`
//!         `pclmulqdq, aes, avx512f, avx512bw, avx512cd, avx512dq, avx512vl}`
//!
//! `X64V3Token`'s feature set is:
//!
//!   `V3 = {sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b,`
//!         `avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe}`
//!
//! Therefore `V3 ⊊ V4` (strict subset). Any instruction encoding safe to
//! execute on a CPU presenting V4's features is also safe under V3's
//! features — the V4 platform is a strict superset of V3 capabilities.
//!
//! [`F32x4Backend`] / [`F32x8Backend`] for `X64V3Token` use intrinsics
//! drawn from the V3 feature set (SSE / AVX / AVX2 / FMA, no AVX-512).
//! Delegating V4's narrower-width methods to V3's implementations is
//! therefore sound: every call site that was sound under a V3 token is
//! sound under a V4 token by the subset relation, and the `Self::Repr`
//! associated type stays identical (`__m128` / `__m256`) so caller code
//! does not observe a representation change.
//!
//! Each delegating method calls the target V3 method with the downcast
//! token supplied by the guaranteed `.v3()` extractor — the soundness
//! chain stays explicit at every call.
//!
//! `X64V4xToken` and `Avx512Fp16Token` are V4 plus additional AVX-512
//! sub-features (BITALG / VPOPCNTDQ for V4x; FP16 for Avx512Fp16Token);
//! they are also strict supersets of V3 by the same argument.
//!
//! ## What this enables
//!
//! After this file:
//! - `f32x4<X64V4Token>`, `f32x8<X64V4Token>` are constructible.
//! - The cross-width `from_halves` / `low` / `high` family lights up
//!   for V4 tokens at every width (W128↔W256 inherits V3's AVX behavior;
//!   W256↔W512 uses V4's native `_mm512_insertf32x8`).
//! - Generic `T: F32x8Backend` code accepts V4 tokens unchanged.

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

use crate::simd::backends::{F32x4Backend, F32x8Backend};

/// Delegate every [`F32x4Backend`] method on `$token` to `X64V3Token`'s
/// impl. `Self::Repr` is set to V3's repr (`__m128`); all method bodies
/// forward through the trait's UFCS path, using the guaranteed `.v3()`
/// extractor on the passed token.
macro_rules! delegate_f32x4_to_v3 {
    ($token:ty) => {
        impl F32x4Backend for $token {
            type Repr = <archmage::X64V3Token as F32x4Backend>::Repr;

            // Construction
            #[inline(always)]
            fn splat(self, v: f32) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::splat(self.v3(), v)
            }
            #[inline(always)]
            fn zero(self) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::zero(self.v3())
            }
            #[inline(always)]
            fn load(self, d: &[f32; 4]) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::load(self.v3(), d)
            }
            #[inline(always)]
            fn from_array(self, a: [f32; 4]) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::from_array(self.v3(), a)
            }
            #[inline(always)]
            fn store(self, r: Self::Repr, o: &mut [f32; 4]) {
                <archmage::X64V3Token as F32x4Backend>::store(self.v3(), r, o)
            }
            #[inline(always)]
            fn to_array(self, r: Self::Repr) -> [f32; 4] {
                <archmage::X64V3Token as F32x4Backend>::to_array(self.v3(), r)
            }

            // Arithmetic
            #[inline(always)]
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::add(self.v3(), a, b)
            }
            #[inline(always)]
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::sub(self.v3(), a, b)
            }
            #[inline(always)]
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::mul(self.v3(), a, b)
            }
            #[inline(always)]
            fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::div(self.v3(), a, b)
            }
            #[inline(always)]
            fn neg(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::neg(self.v3(), a)
            }

            // Math
            #[inline(always)]
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::min(self.v3(), a, b)
            }
            #[inline(always)]
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::max(self.v3(), a, b)
            }
            #[inline(always)]
            fn sqrt(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::sqrt(self.v3(), a)
            }
            #[inline(always)]
            fn abs(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::abs(self.v3(), a)
            }
            #[inline(always)]
            fn floor(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::floor(self.v3(), a)
            }
            #[inline(always)]
            fn ceil(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::ceil(self.v3(), a)
            }
            #[inline(always)]
            fn round(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::round(self.v3(), a)
            }
            #[inline(always)]
            fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::mul_add(self.v3(), a, b, c)
            }
            #[inline(always)]
            fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::mul_sub(self.v3(), a, b, c)
            }

            // Comparisons
            #[inline(always)]
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::simd_eq(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::simd_ne(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::simd_lt(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::simd_le(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::simd_gt(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::simd_ge(self.v3(), a, b)
            }
            #[inline(always)]
            fn blend(self, m: Self::Repr, t: Self::Repr, f: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::blend(self.v3(), m, t, f)
            }

            // Reductions
            #[inline(always)]
            fn reduce_add(self, a: Self::Repr) -> f32 {
                <archmage::X64V3Token as F32x4Backend>::reduce_add(self.v3(), a)
            }
            #[inline(always)]
            fn reduce_min(self, a: Self::Repr) -> f32 {
                <archmage::X64V3Token as F32x4Backend>::reduce_min(self.v3(), a)
            }
            #[inline(always)]
            fn reduce_max(self, a: Self::Repr) -> f32 {
                <archmage::X64V3Token as F32x4Backend>::reduce_max(self.v3(), a)
            }

            // Approximations (override defaults to preserve V3's hardware-rcp/rsqrt)
            #[inline(always)]
            fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::rcp_approx(self.v3(), a)
            }
            #[inline(always)]
            fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::rsqrt_approx(self.v3(), a)
            }
            #[inline(always)]
            fn recip(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::recip(self.v3(), a)
            }
            #[inline(always)]
            fn rsqrt(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::rsqrt(self.v3(), a)
            }

            // Bitwise
            #[inline(always)]
            fn not(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::not(self.v3(), a)
            }
            #[inline(always)]
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::bitand(self.v3(), a, b)
            }
            #[inline(always)]
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::bitor(self.v3(), a, b)
            }
            #[inline(always)]
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::bitxor(self.v3(), a, b)
            }

            // Default-having helpers (override to keep V3's hardware path)
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x4Backend>::clamp(self.v3(), a, lo, hi)
            }
        }
    };
}

/// Same delegation pattern for [`F32x8Backend`].
macro_rules! delegate_f32x8_to_v3 {
    ($token:ty) => {
        impl F32x8Backend for $token {
            type Repr = <archmage::X64V3Token as F32x8Backend>::Repr;

            #[inline(always)]
            fn splat(self, v: f32) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::splat(self.v3(), v)
            }
            #[inline(always)]
            fn zero(self) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::zero(self.v3())
            }
            #[inline(always)]
            fn load(self, d: &[f32; 8]) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::load(self.v3(), d)
            }
            #[inline(always)]
            fn from_array(self, a: [f32; 8]) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::from_array(self.v3(), a)
            }
            #[inline(always)]
            fn store(self, r: Self::Repr, o: &mut [f32; 8]) {
                <archmage::X64V3Token as F32x8Backend>::store(self.v3(), r, o)
            }
            #[inline(always)]
            fn to_array(self, r: Self::Repr) -> [f32; 8] {
                <archmage::X64V3Token as F32x8Backend>::to_array(self.v3(), r)
            }

            #[inline(always)]
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::add(self.v3(), a, b)
            }
            #[inline(always)]
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::sub(self.v3(), a, b)
            }
            #[inline(always)]
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::mul(self.v3(), a, b)
            }
            #[inline(always)]
            fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::div(self.v3(), a, b)
            }
            #[inline(always)]
            fn neg(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::neg(self.v3(), a)
            }

            #[inline(always)]
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::min(self.v3(), a, b)
            }
            #[inline(always)]
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::max(self.v3(), a, b)
            }
            #[inline(always)]
            fn sqrt(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::sqrt(self.v3(), a)
            }
            #[inline(always)]
            fn abs(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::abs(self.v3(), a)
            }
            #[inline(always)]
            fn floor(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::floor(self.v3(), a)
            }
            #[inline(always)]
            fn ceil(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::ceil(self.v3(), a)
            }
            #[inline(always)]
            fn round(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::round(self.v3(), a)
            }
            #[inline(always)]
            fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::mul_add(self.v3(), a, b, c)
            }
            #[inline(always)]
            fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::mul_sub(self.v3(), a, b, c)
            }

            #[inline(always)]
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::simd_eq(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::simd_ne(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::simd_lt(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::simd_le(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::simd_gt(self.v3(), a, b)
            }
            #[inline(always)]
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::simd_ge(self.v3(), a, b)
            }
            #[inline(always)]
            fn blend(self, m: Self::Repr, t: Self::Repr, f: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::blend(self.v3(), m, t, f)
            }

            #[inline(always)]
            fn reduce_add(self, a: Self::Repr) -> f32 {
                <archmage::X64V3Token as F32x8Backend>::reduce_add(self.v3(), a)
            }
            #[inline(always)]
            fn reduce_min(self, a: Self::Repr) -> f32 {
                <archmage::X64V3Token as F32x8Backend>::reduce_min(self.v3(), a)
            }
            #[inline(always)]
            fn reduce_max(self, a: Self::Repr) -> f32 {
                <archmage::X64V3Token as F32x8Backend>::reduce_max(self.v3(), a)
            }

            #[inline(always)]
            fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::rcp_approx(self.v3(), a)
            }
            #[inline(always)]
            fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::rsqrt_approx(self.v3(), a)
            }
            #[inline(always)]
            fn recip(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::recip(self.v3(), a)
            }
            #[inline(always)]
            fn rsqrt(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::rsqrt(self.v3(), a)
            }

            #[inline(always)]
            fn not(self, a: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::not(self.v3(), a)
            }
            #[inline(always)]
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::bitand(self.v3(), a, b)
            }
            #[inline(always)]
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::bitor(self.v3(), a, b)
            }
            #[inline(always)]
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::bitxor(self.v3(), a, b)
            }

            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
                <archmage::X64V3Token as F32x8Backend>::clamp(self.v3(), a, lo, hi)
            }
        }
    };
}

delegate_f32x4_to_v3!(archmage::X64V4Token);
delegate_f32x4_to_v3!(archmage::X64V4xToken);
delegate_f32x4_to_v3!(archmage::Avx512Fp16Token);

delegate_f32x8_to_v3!(archmage::X64V4Token);
delegate_f32x8_to_v3!(archmage::X64V4xToken);
delegate_f32x8_to_v3!(archmage::Avx512Fp16Token);
