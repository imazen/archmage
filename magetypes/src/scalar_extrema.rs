//! IEEE extrema for runtime scalar callers, with canonical NaNs.
#![forbid(unsafe_code)]

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub(crate) use crate::nostd_math::{maximum, maximumf, minimum, minimumf};

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) use selected::{maximum, maximumf, minimum, minimumf};

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! select_extrema {
    ($($name:ident, $hardware:ident, $float:ty);+ $(;)?) => {$(
        #[inline(always)]
        pub(crate) fn $name(a: $float, b: $float) -> $float {
            let Some(proof) = token() else {
                return crate::nostd_math::$name(a, b);
            };
            let value = $hardware(proof, a, b);
            // Preserve the scalar result bits even when hardware selects a NaN payload.
            if value.is_nan() { <$float>::NAN } else { value }
        }
    )+};
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod selected {
    use archmage::{NeonToken, SimdToken};
    use core::arch::aarch64::*;

    #[inline(always)]
    fn token() -> Option<NeonToken> {
        ::archmage::NeonToken::summon()
    }

    select_extrema! {
        minimumf, minimumf_hw, f32;
        maximumf, maximumf_hw, f32;
        minimum, minimum_hw, f64;
        maximum, maximum_hw, f64;
    }

    #[archmage::attune(wrap)]
    fn minimumf_hw(_token: NeonToken, a: f32, b: f32) -> f32 {
        vget_lane_f32::<0>(vmin_f32(vdup_n_f32(a), vdup_n_f32(b)))
    }

    #[archmage::attune(wrap)]
    fn maximumf_hw(_token: NeonToken, a: f32, b: f32) -> f32 {
        vget_lane_f32::<0>(vmax_f32(vdup_n_f32(a), vdup_n_f32(b)))
    }

    #[archmage::attune(wrap)]
    fn minimum_hw(_token: NeonToken, a: f64, b: f64) -> f64 {
        vget_lane_f64::<0>(vmin_f64(vdup_n_f64(a), vdup_n_f64(b)))
    }

    #[archmage::attune(wrap)]
    fn maximum_hw(_token: NeonToken, a: f64, b: f64) -> f64 {
        vget_lane_f64::<0>(vmax_f64(vdup_n_f64(a), vdup_n_f64(b)))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod selected {
    use archmage::{SimdToken, Wasm128Token};
    use core::arch::wasm32::*;

    #[inline(always)]
    fn token() -> Option<Wasm128Token> {
        ::archmage::Wasm128Token::summon()
    }

    select_extrema! {
        minimumf, minimumf_hw, f32;
        maximumf, maximumf_hw, f32;
        minimum, minimum_hw, f64;
        maximum, maximum_hw, f64;
    }

    #[archmage::attune(wrap)]
    fn minimumf_hw(_token: Wasm128Token, a: f32, b: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_min(f32x4_splat(a), f32x4_splat(b)))
    }

    #[archmage::attune(wrap)]
    fn maximumf_hw(_token: Wasm128Token, a: f32, b: f32) -> f32 {
        f32x4_extract_lane::<0>(f32x4_max(f32x4_splat(a), f32x4_splat(b)))
    }

    #[archmage::attune(wrap)]
    fn minimum_hw(_token: Wasm128Token, a: f64, b: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_min(f64x2_splat(a), f64x2_splat(b)))
    }

    #[archmage::attune(wrap)]
    fn maximum_hw(_token: Wasm128Token, a: f64, b: f64) -> f64 {
        f64x2_extract_lane::<0>(f64x2_max(f64x2_splat(a), f64x2_splat(b)))
    }
}
