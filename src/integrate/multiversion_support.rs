//! Integration with multiversion and multiversed crates
//!
//! Provides utilities and documentation for using archmage tokens
//! with function multiversioning.

/// Documentation module explaining multiversion integration
///
/// # Using archmage with `#[multiversed]`
///
/// The `multiversed` crate generates multiple versions of a function
/// for different CPU feature levels. archmage tokens work perfectly
/// with this pattern:
///
/// ```rust,ignore
/// use archmage::{avx2_fma_token, ops};
/// use multiversed::multiversed;
///
/// #[multiversed]
/// fn dot_product(a: &[f32], b: &[f32]) -> f32 {
///     assert_eq!(a.len(), b.len());
///
///     // Token creation is the ONLY unsafe point
///     // Justified because multiversed guarantees AVX2+FMA in this version
///     let token = avx2_fma_token!();
///
///     let mut sum = ops::zero_f32x8(token.avx2());
///
///     let chunks = a.len() / 8;
///     for i in 0..chunks {
///         let av = ops::load_f32x8(token.avx2(), a[i * 8..][..8].try_into().unwrap());
///         let bv = ops::load_f32x8(token.avx2(), b[i * 8..][..8].try_into().unwrap());
///         sum = ops::fmadd_f32x8(token.fma(), av, bv, sum);
///     }
///
///     let arr = ops::to_array_f32x8(sum);
///     let mut result: f32 = arr.iter().sum();
///
///     for i in (chunks * 8)..a.len() {
///         result += a[i] * b[i];
///     }
///
///     result
/// }
/// ```
///
/// # Using archmage with `#[multiversion]`
///
/// The lower-level `multiversion` crate also works:
///
/// ```rust,ignore
/// use archmage::{Avx2Token, ops};
/// use multiversion::multiversion;
///
/// #[multiversion(targets(
///     "x86_64+avx2+fma",
///     "x86_64+avx",
///     "x86_64+sse4.1",
/// ))]
/// fn my_kernel(data: &mut [f32]) {
///     #[cfg(target_feature = "avx2")]
///     {
///         // SAFETY: multiversion guarantees we're in AVX2 variant
///         let token = unsafe { Avx2Token::new_unchecked() };
///         // ... use token-gated operations
///     }
///
///     #[cfg(not(target_feature = "avx2"))]
///     {
///         // Scalar fallback
///     }
/// }
/// ```
///
/// # Why the Token Macros Work
///
/// The `avx2_token!()`, `fma_token!()`, etc. macros contain unsafe code,
/// but are safe to use inside multiversioned functions because:
///
/// 1. **Multiversion guarantees**: The `#[multiversed]` or `#[multiversion]`
///    macro generates separate function versions for each CPU feature level.
///
/// 2. **Target feature propagation**: Each version is compiled with the
///    appropriate `#[target_feature]` attribute, meaning AVX2 instructions
///    are available in the AVX2 version.
///
/// 3. **Runtime dispatch**: The dispatch code checks CPU features at runtime
///    and calls the appropriate version, ensuring the AVX2 version only
///    runs on CPUs with AVX2.
///
/// 4. **Token validity**: Therefore, when `avx2_token!()` executes in the
///    AVX2 version, AVX2 is guaranteed to be available, making the unsafe
///    `new_unchecked()` call valid.
///
/// # Alternative: Runtime Detection
///
/// For code not using multiversion, use runtime detection:
///
/// ```rust,ignore
/// use archmage::{Avx2FmaToken, ops};
///
/// fn my_function(data: &[f32]) -> f32 {
///     if let Some(token) = Avx2FmaToken::try_new() {
///         // AVX2+FMA path - all operations safe via token
///         let v = ops::load_f32x8(token.avx2(), data[..8].try_into().unwrap());
///         // ...
///     } else {
///         // Fallback path
///         // ...
///     }
/// }
/// ```
pub mod docs {}

// Re-export the token macros at the integration module level for convenience
// (They're already exported at crate root, but this provides discoverability)

/// Utility trait for operations that can work with either runtime detection
/// or multiversion dispatch.
pub trait TokenSource {
    /// The token type this source provides
    type Token;

    /// Get the token (may panic if feature unavailable outside multiversion context)
    fn token(&self) -> Self::Token;
}
