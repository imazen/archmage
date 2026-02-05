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
//! # Example
//!
//! ```rust,ignore
//! use archmage::{arcane, X64V3Token};
//! use magetypes::simd::x86::{w128, w256};
//!
//! #[arcane]
//! fn process(token: X64V3Token, input: w128::f32x4) -> w256::f32x8 {
//!     // Inside #[arcane], we have AVX2 context
//!     // Can safely use 128-bit vectors with VEX encoding
//!     let doubled = w256::f32x8::from_halves(token, input, input);
//!     doubled
//! }
//! ```
//!
//! # Cross-Tier Casting Rules
//!
//! | From | To | Method | Context Required |
//! |------|-----|--------|------------------|
//! | Any | Same type | (identity) | None |
//! | w256::* | w128::* (half) | `.low()`, `.high()` | None |
//! | w512::* | w256::* (half) | `.low()`, `.high()` | None |
//! | w128::* | w256::* | `from_halves()` | AVX2 context |
//! | w256::* | w512::* | `from_halves()` | AVX-512 context |

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
    // Note: Cross-tier casting between same-width vectors (e.g., SSE f32x4 vs AVX f32x4)
    // is a transmute of the same underlying type. The main practical casting is:
    //
    // 1. Extracting halves: w256 → two w128, w512 → two w256
    //    These are available as .low() and .high() methods on the types.
    //
    // 2. Combining halves: two w128 → w256, two w256 → w512
    //    These are available as from_halves() constructors.
    //
    // Both patterns are already implemented in the generated types.
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn downcast_is_documented() {
        // This test just verifies the module compiles and documents the casting rules.
        // Actual casting uses .low(), .high(), and from_halves() on the types.
    }
}
