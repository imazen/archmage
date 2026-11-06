//! Ordinary scalar API; acceleration preserves the reference bit contracts.
pub use crate::reference::*;

/// Non-dispatch reference algorithms, including the historical compile-time
/// hardware FMA choice with std. No runtime CPU selection occurs here.
pub mod reference {
    pub use crate::reference::*;
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

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        use archmage::SimdToken;
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = archmage::X64V3Token::summon() {
            return crate::simd::scalar::powf_v3_t(t, x, y);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(t) = archmage::NeonToken::summon() {
            return crate::simd::scalar::powf_neon_t(t, x, y);
        }
        crate::reference::powf(x, y)
    }
}
#[cfg(all(feature = "runtime-dispatch", target_arch = "x86_64"))]
pub use selected::fma;
#[cfg(all(
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
/// [`powf`] where this crate enables acceleration. Reference policy evaluation
/// remains independently available through [`reference::powf_with`].
#[cfg(feature = "runtime-dispatch")]
#[inline(always)]
pub fn powf_with<const P: u64>(x: f32, y: f32) -> f32 {
    crate::__pow_policy::evaluate_with_core::<P>(x, y, powf)
}
