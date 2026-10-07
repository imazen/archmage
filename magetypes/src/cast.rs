//! Cross-tier casting traits, deprecated since 0.9.30.
//!
//! `Downcast` and `Upcast` were meant as type-level casts between same-width
//! vectors bound to different tokens, from the time each tier had its own
//! concrete vector types. Nothing has ever implemented either one. A generic
//! vector carries its token, so moving it to another context means rebuilding
//! it with that context's token:
//!
//! - Any vector, any token: `f32x8::from_array_t(token, v.to_array())`. Token
//!   conversions supply the narrower token, for example `v4.v3()`; a wider
//!   token comes from `summon()` or from the enclosing `#[arcane]` region.
//! - Native backends also have `raw()` and `from_raw_t(token, raw)`, which
//!   unwrap and rewrap the raw intrinsic value at no cost.
//! - Width changes use `low()`, `high()`, `split()` and `from_halves_t()`.
//!
//! Both traits, and with them this module, are queued for removal in
//! magetypes 0.10.

/// Marker trait for types that can be safely downcast.
///
/// Downcasting from a wider SIMD context to a narrower one is always safe
/// because the narrower context requires fewer CPU features.
///
/// Deprecated since 0.9.30 and queued for removal in magetypes 0.10: nothing
/// has ever implemented it. See the module docs for what replaces it.
#[deprecated(
    since = "0.9.30",
    note = "never implemented; a vector carries its token, so rebuild it under the narrower token (for example `f32x8::from_array_t(v4.v3(), v.to_array())`). Will be removed in magetypes 0.10."
)]
pub trait Downcast<T> {
    /// Downcast to a narrower context type.
    ///
    /// This is always safe - no target feature requirements.
    fn downcast(self) -> T;
}

// `Upcast` declares an `unsafe fn`, so its definition lives in `simd_storage`,
// the only module allowed `unsafe`; this re-export keeps its public path.
#[allow(deprecated)]
pub use crate::simd_storage::Upcast;

// =============================================================================
// x86 implementations
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    // Cross-tier casting between same-width vectors (e.g. an SSE-context f32x4
    // used inside an AVX2 region) is a no-op on the generic types: the value
    // already carries its token, so there is nothing to convert. Width changes
    // go through `low()`, `high()`, `split()` and `from_halves_t()` in
    // `simd::generic::cross_width`.
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn downcast_is_documented() {
        // This test just verifies the module compiles and documents the casting rules.
    }
}
