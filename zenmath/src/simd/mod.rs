//! Accurate packed f32 math without Magetypes. Four f32 inputs are evaluated
//! using four binary64 intermediates (256 bits on V3, two registers on NEON/WASM).
//! The four-input V4x proof reuses that same 256-bit kernel. Separate eight-input
//! V4x proof entries use eight native AVX-512 binary64 lanes.
//!
//! Arrays and slices expose direct `_v3` feature-context bodies and `_v3_t`
//! proof entries (and corresponding NEON/WASM/scalar forms). Internal kernels
//! never probe. Selection uses compile-time guarantees unless this crate's own
//! `runtime-dispatch` feature is enabled, regardless of unified Archmage features.
//! Assumes nearest-even rounding and gradual underflow; NaN payloads unspecified.
#![allow(clippy::missing_safety_doc)]
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod context;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod correct;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod gamma;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod packed;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod soft_fma;

/// Selection is outside every packed loop. This guard is deliberately owned by
/// Zenmath, rather than depending on whether Archmage's std feature is unified.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn available<T: archmage::SimdToken>() -> Option<T> {
    #[cfg(feature = "runtime-dispatch")]
    {
        T::summon()
    }
    #[cfg(not(feature = "runtime-dispatch"))]
    {
        if T::compiled_with() == Some(true) {
            T::summon()
        } else {
            None
        }
    }
}

#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
#[doc(hidden)]
#[path = "array.rs"]
pub mod __array;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
#[doc(hidden)]
#[path = "scalar-transcendentals.rs"]
pub mod __scalar_transcendentals;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
#[doc(hidden)]
#[path = "slice.rs"]
pub mod __slice;
#[cfg(feature = "correct-rounding")]
pub use __array as array;
#[cfg(feature = "correct-rounding")]
pub use __slice as slice;

/// Scalar hardware primitives and whole-algorithm feature contexts.
pub mod scalar;

#[cfg(all(
    feature = "runtime-dispatch",
    feature = "configurable",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub(crate) mod policy;
