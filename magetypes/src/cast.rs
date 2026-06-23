//! Cross-tier casting utilities.
//!
//! These functions allow safe casting between SIMD vectors of the same size
//! but different context (e.g., SSE vs AVX2 on x86).
//!
//! # Safety Model
//!
//! - **Downcast** (wider context → narrower): Always safe, no feature requirements
//! - **Upcast** (narrower context → wider): Requires being in the wider context
//!   (inside an `#[arcane]` function with the appropriate token)
//!
//! # Generic context
//!
//! In the generic type system every value already carries its token, so most
//! cross-context use is just passing the right `generic::fNxM<T>` around. These
//! marker traits exist for the cases that need an explicit, type-level cast
//! between same-width vectors bound to different tokens.

/// Marker trait for types that can be safely downcast.
///
/// Downcasting from a wider SIMD context to a narrower one is always safe
/// because the narrower context requires fewer CPU features.
pub trait Downcast<T> {
    /// Downcast to a narrower context type.
    ///
    /// This is always safe - no target feature requirements.
    fn downcast(self) -> T;
}

/// Marker trait for types that can be upcast with proof of context.
///
/// Upcasting requires being in the appropriate context (inside `#[arcane]`
/// with the right token).
pub trait Upcast<T> {
    /// Upcast to a wider context type.
    ///
    /// # Safety
    ///
    /// Caller must ensure they are in an appropriate SIMD context
    /// (inside `#[arcane]` function with matching token).
    unsafe fn upcast(self) -> T;
}

// =============================================================================
// x86 implementations
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    // Cross-tier casting between same-width vectors (e.g. an SSE-context f32x4
    // used inside an AVX2 region) is a no-op on the generic types: the value
    // already carries its token, so there is nothing to convert. Width changes
    // (half extraction / combination) are not currently exposed on the generic
    // types.
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn downcast_is_documented() {
        // This test just verifies the module compiles and documents the casting rules.
    }
}
