//! Ordinary scalar API; acceleration preserves the reference bit contracts.
#[cfg(feature = "correct-rounding")]
mod precise_reference {
    // Some fallback exports are shadowed by selected overrides in the parent.
    #[allow(unused_imports)]
    pub use crate::reference::{
        acosf, asinh, atan, atan2, atan2f, atanf, atanh, cbrt, cbrtf, cos, cosf, exp, exp2, exp2f,
        expf, expm1f, hypotf, log, log1p, log1pf, log2, log2f, log10, log10f, logf, pow, powf, sin,
        sincosf, sinf, tan, tanf, tanh, tanhf,
    };
}
#[cfg(feature = "correct-rounding")]
pub use precise_reference::*;
#[cfg(feature = "configurable")]
mod configurable_reference {
    // Some fallback exports are shadowed by selected overrides in the parent.
    #[allow(unused_imports)]
    pub use crate::reference::{
        cbrt_lowp_f32, cbrt_midp_f32, cbrt_midp_precise_f32, exp_lowp_f32, exp_lowp_f64,
        exp_midp_f32, exp2_lowp_f32, exp2_lowp_f64, exp2_midp_f32, ln_lowp_f32, ln_lowp_f64,
        ln_midp_f32, log2_lowp_f32, log2_lowp_f64, log2_midp_f32, log10_lowp_f32, log10_lowp_f64,
        log10_midp_f32, pow_lowp_f32, pow_lowp_f64, pow_midp_f32, powf_with, sigmoid_midp_f32,
        silu_midp_f32, unfused,
    };
}
#[cfg(feature = "configurable")]
pub use configurable_reference::*;
mod primitive_reference {
    // Some fallback exports are shadowed by selected overrides in the parent.
    #[allow(unused_imports)]
    pub use crate::reference::{
        ceil, ceilf, floor, floorf, fma, fmaf, round, roundeven, roundevenf, roundf, sqrt, sqrtf,
    };
}
pub use primitive_reference::*;

/// Non-dispatch reference algorithms, including the historical compile-time
/// hardware FMA choice with std. No runtime CPU selection occurs here.
pub mod reference {
    #[cfg(feature = "correct-rounding")]
    pub use crate::reference::{
        acosf, asinh, atan, atan2, atan2f, atanf, atanh, cbrt, cbrtf, cos, cosf, exp, exp2, exp2f,
        expf, expm1f, hypotf, log, log1p, log1pf, log2, log2f, log10, log10f, logf, pow, powf, sin,
        sincosf, sinf, tan, tanf, tanh, tanhf,
    };
    #[cfg(feature = "configurable")]
    pub use crate::reference::{
        cbrt_lowp_f32, cbrt_midp_f32, cbrt_midp_precise_f32, exp_lowp_f32, exp_lowp_f64,
        exp_midp_f32, exp2_lowp_f32, exp2_lowp_f64, exp2_midp_f32, ln_lowp_f32, ln_lowp_f64,
        ln_midp_f32, log2_lowp_f32, log2_lowp_f64, log2_midp_f32, log10_lowp_f32, log10_lowp_f64,
        log10_midp_f32, pow_lowp_f32, pow_lowp_f64, pow_midp_f32, powf_with, sigmoid_midp_f32,
        silu_midp_f32, unfused,
    };
    pub use crate::reference::{
        ceil, ceilf, floor, floorf, fma, fmaf, round, roundeven, roundevenf, roundf, sqrt, sqrtf,
    };
}

// These overrides are deliberately selective. The dispatch study records
// mixed/loss cases as well as improvements; explicit proof APIs remain available.
#[cfg(all(
    feature = "runtime-dispatch",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
macro_rules! primitive {
    ($name:ident, $ty:ty) => {
        #[inline]
        pub fn $name(x: $ty) -> $ty {
            crate::simd::scalar::$name(x)
        }
    };
}
#[cfg(all(
    feature = "runtime-dispatch",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
mod selected {
    primitive!(sqrtf, f32);
    primitive!(sqrt, f64);
    primitive!(floorf, f32);
    primitive!(floor, f64);
    primitive!(ceil, f64);
    primitive!(roundf, f32);
    primitive!(round, f64);

    #[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
    primitive!(roundeven, f64);
    #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
    primitive!(roundevenf, f32);
    #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
    primitive!(ceilf, f32);

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn fma(x: f64, y: f64, z: f64) -> f64 {
        crate::simd::scalar::fma(x, y, z)
    }

    #[cfg(all(
        any(
            feature = "correct-rounding",
            feature = "normal",
            feature = "configurable"
        ),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        use archmage::SimdToken;
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = archmage::X64V3Token::summon() {
            return crate::simd::__scalar_transcendentals::powf_v3_t(t, x, y);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(t) = archmage::NeonToken::summon() {
            return crate::simd::__scalar_transcendentals::powf_neon_t(t, x, y);
        }
        crate::reference::powf(x, y)
    }
}
#[cfg(all(feature = "runtime-dispatch", target_arch = "x86_64"))]
pub use selected::fma;
#[cfg(all(
    feature = "correct-rounding",
    feature = "runtime-dispatch",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub use selected::powf;
#[cfg(all(
    feature = "runtime-dispatch",
    any(target_arch = "x86_64", target_arch = "wasm32")
))]
pub use selected::roundeven;
#[cfg(all(
    feature = "runtime-dispatch",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
pub use selected::{ceil, floor, floorf, round, roundf, sqrt, sqrtf};
#[cfg(all(
    feature = "runtime-dispatch",
    any(target_arch = "aarch64", target_arch = "wasm32")
))]
pub use selected::{ceilf, roundevenf};

/// Power with a compile-time policy, selecting the same core as ordinary
/// `powf` where this crate enables acceleration. Reference policy evaluation
/// remains independently available through [`reference::powf_with`].
#[cfg(all(feature = "runtime-dispatch", feature = "configurable"))]
#[inline(always)]
pub fn powf_with<const P: u64>(x: f32, y: f32) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        crate::simd::policy::select::<P>(x, y)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        crate::__pow_policy::evaluate_with_core::<P>(x, y, selected_powf)
    }
}

// Private ordinary correct core, also used when only policy/normal APIs exist.
#[cfg(any(feature = "normal", feature = "configurable"))]
#[inline(always)]
pub(crate) fn selected_powf(x: f32, y: f32) -> f32 {
    #[cfg(all(
        feature = "runtime-dispatch",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        selected::powf(x, y)
    }
    #[cfg(not(all(
        feature = "runtime-dispatch",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    {
        crate::reference::powf(x, y)
    }
}
