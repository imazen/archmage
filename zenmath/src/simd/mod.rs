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
mod context;
mod correct;
mod packed;
mod soft_fma;

/// Selection is outside every packed loop. This guard is deliberately owned by
/// Zenmath, rather than depending on whether Archmage's std feature is unified.
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

pub mod array;
pub mod slice;

/// Scalar hardware primitives and whole-algorithm feature contexts.
pub mod scalar;
