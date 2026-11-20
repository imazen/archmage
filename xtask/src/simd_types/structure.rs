//! Comparison-trait and macro definitions shared by the generated SIMD module.

/// Generate comparison trait definitions
pub fn generate_comparison_traits() -> String {
    r#"
// ============================================================================
// Comparison Traits (return masks, not bool)
// ============================================================================

/// **Superseded (queued for deprecation):** Never implemented; use the inherent `v.simd_eq(rhs)` method on vector types instead.
pub trait SimdEq<Rhs = Self> {
    type Output;
    fn simd_eq(self, rhs: Rhs) -> Self::Output;
}

/// **Superseded (queued for deprecation):** Never implemented; use the inherent `v.simd_ne(rhs)` method on vector types instead.
pub trait SimdNe<Rhs = Self> {
    type Output;
    fn simd_ne(self, rhs: Rhs) -> Self::Output;
}

/// **Superseded (queued for deprecation):** Never implemented; use the inherent `v.simd_lt(rhs)` method on vector types instead.
pub trait SimdLt<Rhs = Self> {
    type Output;
    fn simd_lt(self, rhs: Rhs) -> Self::Output;
}

/// **Superseded (queued for deprecation):** Never implemented; use the inherent `v.simd_le(rhs)` method on vector types instead.
pub trait SimdLe<Rhs = Self> {
    type Output;
    fn simd_le(self, rhs: Rhs) -> Self::Output;
}

/// **Superseded (queued for deprecation):** Never implemented; use the inherent `v.simd_gt(rhs)` method on vector types instead.
pub trait SimdGt<Rhs = Self> {
    type Output;
    fn simd_gt(self, rhs: Rhs) -> Self::Output;
}

/// **Superseded (queued for deprecation):** Never implemented; use the inherent `v.simd_ge(rhs)` method on vector types instead.
pub trait SimdGe<Rhs = Self> {
    type Output;
    fn simd_ge(self, rhs: Rhs) -> Self::Output;
}

"#
    .to_string()
}
