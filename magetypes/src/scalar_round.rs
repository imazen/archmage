//! Scalar rounding using the CPU features guaranteed by the build.
#![forbid(unsafe_code)]

pub(crate) use selected::{ceil, ceilf, floor, floorf, roundeven, roundevenf, trunc, truncf};

mod software {
    pub(crate) use crate::nostd_math::{ceil, ceilf, floor, floorf, roundeven, roundevenf};

    macro_rules! trunc {
        ($name:ident, $float:ty, $bits:ty, $mask:literal, $bias:literal, $mantissa:literal) => {
            #[inline]
            pub(crate) fn $name(x: $float) -> $float {
                let bits = x.to_bits();
                let exponent = ((bits >> $mantissa) as i32 & $mask) - $bias;
                if exponent < 0 {
                    let zero: $float = 0.0;
                    zero.copysign(x)
                } else if exponent >= $mantissa {
                    x
                } else {
                    let one: $bits = 1;
                    let fraction = (one << ($mantissa - exponent)) - 1;
                    <$float>::from_bits(bits & !fraction)
                }
            }
        };
    }

    trunc!(truncf, f32, u32, 0xff, 127, 23);
    trunc!(trunc, f64, u64, 0x7ff, 1023, 52);
}

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
    all(
        target_arch = "x86_64",
        target_feature = "sse3",
        target_feature = "ssse3",
        target_feature = "sse4.1",
        target_feature = "sse4.2",
        target_feature = "popcnt",
        target_feature = "cmpxchg16b"
    )
))]
macro_rules! select_round {
    ($($name:ident, $hardware:ident, $method:ident, $float:ty);+ $(;)?) => {$(
        #[inline(always)]
        pub(crate) fn $name(x: $float) -> $float {
            let Some(proof) = token() else {
                return super::software::$name(x);
            };
            #[cfg(feature = "std")]
            {
                let _ = proof;
                x.$method()
            }
            #[cfg(not(feature = "std"))]
            {
                $hardware(proof, x)
            }
        }
    )+};
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod selected {
    use archmage::{NeonToken, SimdToken};
    #[cfg(not(feature = "std"))]
    use core::arch::aarch64::*;

    // The enclosing cfg proves every feature of the token. Test builds may
    // disable the token to select the software path.
    #[inline(always)]
    fn token() -> Option<NeonToken> {
        ::archmage::NeonToken::summon()
    }

    select_round! {
        floorf, floorf_hw, floor, f32;
        floor, floor_hw, floor, f64;
        ceilf, ceilf_hw, ceil, f32;
        ceil, ceil_hw, ceil, f64;
        roundevenf, roundevenf_hw, round_ties_even, f32;
        roundeven, roundeven_hw, round_ties_even, f64;
        truncf, truncf_hw, trunc, f32;
        trunc, trunc_hw, trunc, f64;
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn floorf_hw(_token: NeonToken, x: f32) -> f32 {
        vget_lane_f32::<0>(vrndm_f32(vdup_n_f32(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn floor_hw(_token: NeonToken, x: f64) -> f64 {
        vget_lane_f64::<0>(vrndm_f64(vdup_n_f64(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn ceilf_hw(_token: NeonToken, x: f32) -> f32 {
        vget_lane_f32::<0>(vrndp_f32(vdup_n_f32(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn ceil_hw(_token: NeonToken, x: f64) -> f64 {
        vget_lane_f64::<0>(vrndp_f64(vdup_n_f64(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn roundevenf_hw(_token: NeonToken, x: f32) -> f32 {
        vget_lane_f32::<0>(vrndn_f32(vdup_n_f32(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn roundeven_hw(_token: NeonToken, x: f64) -> f64 {
        vget_lane_f64::<0>(vrndn_f64(vdup_n_f64(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn truncf_hw(_token: NeonToken, x: f32) -> f32 {
        vget_lane_f32::<0>(vrnd_f32(vdup_n_f32(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn trunc_hw(_token: NeonToken, x: f64) -> f64 {
        vget_lane_f64::<0>(vrnd_f64(vdup_n_f64(x)))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod selected {
    use archmage::{SimdToken, Wasm128Token};
    #[cfg(not(feature = "std"))]
    use core::arch::wasm32::*;

    // The enclosing cfg proves every feature of the token. Test builds may
    // disable the token to select the software path.
    #[inline(always)]
    fn token() -> Option<Wasm128Token> {
        ::archmage::Wasm128Token::summon()
    }

    select_round! {
        floorf, floorf_hw, floor, f32;
        floor, floor_hw, floor, f64;
        ceilf, ceilf_hw, ceil, f32;
        ceil, ceil_hw, ceil, f64;
        roundevenf, roundevenf_hw, round_ties_even, f32;
        roundeven, roundeven_hw, round_ties_even, f64;
        truncf, truncf_hw, trunc, f32;
        trunc, trunc_hw, trunc, f64;
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn floorf_hw(_token: Wasm128Token, x: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn floor_hw(_token: Wasm128Token, x: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_floor(f64x2_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn ceilf_hw(_token: Wasm128Token, x: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_ceil(f32x4_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn ceil_hw(_token: Wasm128Token, x: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_ceil(f64x2_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn roundevenf_hw(_token: Wasm128Token, x: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_nearest(f32x4_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn roundeven_hw(_token: Wasm128Token, x: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_nearest(f64x2_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn truncf_hw(_token: Wasm128Token, x: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_trunc(f32x4_splat(x)))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn trunc_hw(_token: Wasm128Token, x: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_trunc(f64x2_splat(x)))
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse3",
    target_feature = "ssse3",
    target_feature = "sse4.1",
    target_feature = "sse4.2",
    target_feature = "popcnt",
    target_feature = "cmpxchg16b"
))]
mod selected {
    use archmage::{SimdToken, X64V2Token};
    #[cfg(not(feature = "std"))]
    use core::arch::x86_64::*;

    // The enclosing cfg proves every feature of the token. Test builds may
    // disable the token to select the software path.
    #[inline(always)]
    fn token() -> Option<X64V2Token> {
        ::archmage::X64V2Token::summon()
    }

    select_round! {
        floorf, floorf_hw, floor, f32;
        floor, floor_hw, floor, f64;
        ceilf, ceilf_hw, ceil, f32;
        ceil, ceil_hw, ceil, f64;
        roundevenf, roundevenf_hw, round_ties_even, f32;
        roundeven, roundeven_hw, round_ties_even, f64;
        truncf, truncf_hw, trunc, f32;
        trunc, trunc_hw, trunc, f64;
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn floorf_hw(_token: X64V2Token, x: f32) -> f32 {
        let v = _mm_set_ss(x);
        _mm_cvtss_f32(_mm_round_ss::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(v, v))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn floor_hw(_token: X64V2Token, x: f64) -> f64 {
        let v = _mm_set_sd(x);
        _mm_cvtsd_f64(_mm_round_sd::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(v, v))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn ceilf_hw(_token: X64V2Token, x: f32) -> f32 {
        let v = _mm_set_ss(x);
        _mm_cvtss_f32(_mm_round_ss::<{ _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC }>(v, v))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn ceil_hw(_token: X64V2Token, x: f64) -> f64 {
        let v = _mm_set_sd(x);
        _mm_cvtsd_f64(_mm_round_sd::<{ _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC }>(v, v))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn roundevenf_hw(_token: X64V2Token, x: f32) -> f32 {
        let v = _mm_set_ss(x);
        _mm_cvtss_f32(_mm_round_ss::<
            { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
        >(v, v))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn roundeven_hw(_token: X64V2Token, x: f64) -> f64 {
        let v = _mm_set_sd(x);
        _mm_cvtsd_f64(_mm_round_sd::<
            { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
        >(v, v))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn truncf_hw(_token: X64V2Token, x: f32) -> f32 {
        let v = _mm_set_ss(x);
        _mm_cvtss_f32(_mm_round_ss::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
            v, v,
        ))
    }

    #[cfg(not(feature = "std"))]
    #[archmage::attune(wrap)]
    fn trunc_hw(_token: X64V2Token, x: f64) -> f64 {
        let v = _mm_set_sd(x);
        _mm_cvtsd_f64(_mm_round_sd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
            v, v,
        ))
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
    all(
        target_arch = "x86_64",
        target_feature = "sse3",
        target_feature = "ssse3",
        target_feature = "sse4.1",
        target_feature = "sse4.2",
        target_feature = "popcnt",
        target_feature = "cmpxchg16b"
    )
)))]
use software as selected;
