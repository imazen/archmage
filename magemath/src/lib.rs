//! Configurable f32 pow, exponential, logarithm, and cube-root approximations.
//!
//! Configure operations through typed profiles implementing
//! [`powf::PowfProfile`], [`expf::ExpfProfile`], [`logf::LogfProfile`], or
//! [`cbrtf::CbrtfProfile`] (using [`powf::presets`], [`expf::presets`],
//! [`logf::presets`], [`cbrtf::presets`], custom `*Spec` types, or
//! [`powf::Options`]/`*Options` encoders). Modifiers [`Caller<P>`],
//! [`Unchecked<P>`], and [`Backend<P>`] override single axes without
//! rewriting a profile.
//!
//! Scalar, array, and slice entries select a backend once per call. Typed
//! vector entries take the caller's feature proof explicitly; reuse that proof
//! with `*_with_token` for scalar tails.
//! ```
//! use archmage::ScalarToken;
//! use magetypes::simd::generic::f32x4;
//! use magemath::Caller;
//! use magemath::powf::{presets::GammaMidp, try_powf_f32x4_with};
//!
//! let input = f32x4::from_array_t(ScalarToken, [0.0, 0.25, 0.5, 1.0]);
//! let output = try_powf_f32x4_with::<Caller<GammaMidp>, _>(ScalarToken, input, 2.2).unwrap();
//! assert_eq!(output.to_array()[3], 1.0);
//! ```
//!
//! Magemath depends on Magetypes and Archmage. It does not replace vector storage
//! or primitive arithmetic, and Magetypes does not depend on Magemath.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#[cfg(test)]
extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;

/// Whole-operation feature selection, independent of numerical policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Execution {
    /// Reuse a capable caller or select a native tier once per operation.
    #[default]
    Automatic,
    /// Retain the caller's Magetypes tier without autonomous upgrades.
    /// A std HIGHP provider may still use its own internal CPU dispatch.
    Caller,
}

/// Whether accepted results must agree bit for bit across backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Portability {
    /// One fused graph, with exact software FMA when necessary. Not std power.
    /// Full HIGHP preserves NaN classification; NaN payload/sign are unspecified.
    #[default]
    BitExact,
    /// Native fused or separate graph; result bits and range rejection may vary.
    /// HIGHP also inherits std platform special-value classification differences.
    BackendDependent,
}

/// Whether to enforce the selected numerical domain at runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Validation {
    /// Reject the whole vector/slice, returning the exact mask or first failing index.
    #[default]
    Checked,
    /// Omit numerical input/output checks. The caller promises the same domain.
    /// Violations have unspecified numerical results, never memory unsoundness.
    /// Supported widths and valid option encodings are still enforced.
    AssumeValid,
}

/// Profile modifier overriding execution to [`Execution::Caller`].
pub struct Caller<P>(core::marker::PhantomData<fn() -> P>);

impl<P> Copy for Caller<P> {}

impl<P> Clone for Caller<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> core::fmt::Debug for Caller<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Caller").finish()
    }
}

/// Profile modifier overriding validation to [`Validation::AssumeValid`].
pub struct Unchecked<P>(core::marker::PhantomData<fn() -> P>);

impl<P> Copy for Unchecked<P> {}

impl<P> Clone for Unchecked<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> core::fmt::Debug for Unchecked<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Unchecked").finish()
    }
}

/// Profile modifier overriding portability to [`Portability::BackendDependent`].
pub struct Backend<P>(core::marker::PhantomData<fn() -> P>);

impl<P> Copy for Backend<P> {}

impl<P> Clone for Backend<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> core::fmt::Debug for Backend<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Backend").finish()
    }
}

/// Failure of a slice operation: the reason and the first input that caused it.
///
/// `kind` is the family's error kind (for example
/// [`powf::PowfErrorKind`]). Slice entries stop at the first rejected vector;
/// the destination's contents are unspecified after an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SliceError<K> {
    /// Which part of the contract failed.
    pub kind: K,
    /// Index of the first rejected input. Zero for a rejected uniform argument
    /// such as the exponent.
    pub index: usize,
}

impl<K: core::fmt::Debug> core::fmt::Display for SliceError<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} at index {}", self.kind, self.index)
    }
}

#[cfg(feature = "std")]
impl<K: core::fmt::Debug> std::error::Error for SliceError<K> {}

pub mod cbrtf;
pub mod expf;
pub mod logf;
pub mod powf;
mod soft_fma;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)] // The copy also contains vector-primitive helpers not used here.
mod wasm_fma;
