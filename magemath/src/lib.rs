//! Configurable `f32` pow, exponential, logarithm, and cube-root approximations.
//!
//! Configure operations with fluent `const` option builders —
//! [`powf::PowfOptions`], [`expf::ExpfOptions`], [`logf::LogfOptions`], and
//! [`cbrtf::CbrtfOptions`] — ending in `.bits()` for compile-time
//! specialization through [`powf::Powf`], [`expf::Expf`], [`logf::Logf`],
//! [`cbrtf::Cbrtf`], or inline on Magetypes SIMD vectors via [`SimdMath`].
//! Dynamic runtime configurations use `.runtime()` on the same option value.
//!
//! ```
//! use archmage::ScalarToken;
//! use magetypes::simd::generic::f32x4;
//! use magemath::{SimdMath, powf::{Powf, PowfOptions}};
//!
//! const GAMMA: u128 = PowfOptions::new().gamma().midp().bits();
//! const FAST_GAMMA: u128 = PowfOptions::new().gamma().midp().unchecked().bits();
//!
//! let src = [0.0, 0.25, 0.5, 1.0];
//! let mut dst = [0.0; 4];
//! Powf::<GAMMA>::slice(&src, &mut dst, 2.2).unwrap();
//!
//! let v = f32x4::from_array_t(ScalarToken, src);
//! let checked = v.try_powf_with::<GAMMA>(2.2).unwrap();
//! let inline = v.powf::<FAST_GAMMA>(2.2);
//! assert_eq!(checked.to_array(), dst);
//! assert_eq!(inline.to_array(), dst);
//! ```
//!
//! Magemath depends on Magetypes and Archmage. It does not replace vector
//! storage or primitive arithmetic, and Magetypes does not depend on Magemath.
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
    /// Upgrade once per operation to a wider or faster runtime SIMD tier when
    /// beneficial (for example `Scalar -> V3/V4/NEON/Wasm128`, or `V3 -> V4`
    /// on slices, 16-lane vectors, and 8-lane `Highp` power), while staying
    /// inline when the caller's token is already optimal for the width.
    #[default]
    Upgrade,
    /// Retain the caller's Magetypes tier without autonomous runtime upgrades.
    /// A `std` `Highp` provider may still use its own internal CPU dispatch.
    Caller,
}

/// Whether accepted results must agree bit for bit across backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Portability {
    /// One fused graph, with exact software FMA when necessary. Not `std` power.
    /// Full `Highp` preserves NaN classification; NaN payload/sign are unspecified.
    #[default]
    BitExact,
    /// Native fused or separate graph; result bits and range rejection may vary.
    /// `Highp` also inherits `std` platform special-value classification differences.
    BackendDependent,
}

/// How out-of-bounds inputs and intermediate ranges are handled at runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Validation {
    /// Reject the whole vector/slice on out-of-bounds inputs, returning the
    /// exact failing lane mask or first failing slice index
    /// (`.error_on_out_of_bounds()` / `.checked()`).
    #[default]
    Checked,
    /// Clamp finite out-of-bounds inputs into the configured domain/range (and
    /// saturate out-of-range outputs on `Midp`/`Lowp`), propagating `NaN` as
    /// `NaN` without returning a numerical error (`.clamp_out_of_bounds()`).
    ClampOutOfBounds,
    /// Omit numerical input/output checks and clamps. The caller promises all
    /// inputs satisfy the configured domain (`.assume_in_bounds()` / `.unchecked()`).
    /// Violations have unspecified numerical results, never memory unsoundness.
    /// Supported widths and valid option encodings are still enforced.
    AssumeValid,
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

mod private {
    pub trait Sealed: Copy {
        fn eval_powf_const<const CFG: u128>(
            self,
            exponent: f32,
        ) -> Result<Self, crate::powf::PowfError>;
        fn eval_powf_runtime(
            self,
            exponent: f32,
            options: crate::powf::PowfOptions,
        ) -> Result<Self, crate::powf::PowfError>;

        fn eval_expf_const<const CFG: u128, const OP: u8>(
            self,
        ) -> Result<Self, crate::expf::ExpfError>;
        fn eval_expf_runtime<const OP: u8>(
            self,
            options: crate::expf::ExpfOptions,
        ) -> Result<Self, crate::expf::ExpfError>;

        fn eval_logf_const<const CFG: u128, const OP: u8>(
            self,
        ) -> Result<Self, crate::logf::LogfError>;
        fn eval_logf_runtime<const OP: u8>(
            self,
            options: crate::logf::LogfOptions,
        ) -> Result<Self, crate::logf::LogfError>;

        fn eval_cbrtf_const<const CFG: u128>(self) -> Result<Self, crate::cbrtf::CbrtfError>;
        fn eval_cbrtf_runtime(
            self,
            options: crate::cbrtf::CbrtfOptions,
        ) -> Result<Self, crate::cbrtf::CbrtfError>;
    }
}

/// Sealed Magetypes `f32` SIMD vector (`f32x4`, `f32x8`, and with `w512`, `f32x16`)
/// supported by Magemath's operator structs and [`SimdMath`].
pub trait SimdVector: private::Sealed {}

/// Configurable transcendental operations inline on Magetypes `f32` vectors.
///
/// Every method reuses the capability token carried inside `self` (`self.token()`),
/// so no redundant token parameter or lane-width suffix is needed.
///
/// - `try_*` uses each family's default checked `Midp` `BitExact` options.
/// - `try_*_with::<CFG>` specializes at compile time from `*Options::bits()`.
/// - Infallible `powf::<CFG>`, `exp2::<CFG>`, `exp::<CFG>`, `log2::<CFG>`,
///   `ln::<CFG>`, `log10::<CFG>`, `log1p::<CFG>`, and `cbrt::<CFG>` return `Self`
///   directly and fail at compile time unless `CFG` is infallible
///   (`.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or
///   `.admit_all_floats()` / `.full_domain()`).
pub trait SimdMath: SimdVector {
    /// Checked `self^exponent` with default [`powf::PowfOptions::new()`].
    #[inline(always)]
    fn try_powf(self, exponent: f32) -> Result<Self, powf::PowfError> {
        self.try_powf_with::<{ powf::PowfOptions::new().bits() }>(exponent)
    }

    /// Checked `self^exponent` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_powf_with<const CFG: u128>(self, exponent: f32) -> Result<Self, powf::PowfError> {
        self.eval_powf_const::<CFG>(exponent)
    }

    /// Infallible `self^exponent` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats().highp()`:
    ///
    /// ```compile_fail
    /// use archmage::ScalarToken;
    /// use magetypes::simd::generic::f32x4;
    /// use magemath::{SimdMath, powf::PowfOptions};
    /// const CHECKED: u128 = PowfOptions::new().gamma().bits();
    /// let v = f32x4::splat_t(ScalarToken, 0.5);
    /// let _ = v.powf::<CHECKED>(2.2);
    /// ```
    #[inline(always)]
    fn powf<const CFG: u128>(self, exponent: f32) -> Self {
        const {
            let opts = powf::PowfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "SimdMath::powf::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats().highp()`; use `try_powf_with::<CFG>` for checked options"
            );
        }
        match self.eval_powf_const::<CFG>(exponent) {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `2^self` with default [`expf::ExpfOptions::new()`].
    #[inline(always)]
    fn try_exp2(self) -> Result<Self, expf::ExpfError> {
        self.try_exp2_with::<{ expf::ExpfOptions::new().bits() }>()
    }

    /// Checked `2^self` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_exp2_with<const CFG: u128>(self) -> Result<Self, expf::ExpfError> {
        self.eval_expf_const::<CFG, 0>()
    }

    /// Infallible `2^self` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn exp2<const CFG: u128>(self) -> Self {
        const {
            let opts = expf::ExpfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "SimdMath::exp2::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_exp2_with::<CFG>` for checked options"
            );
        }
        match self.eval_expf_const::<CFG, 0>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `e^self` with default [`expf::ExpfOptions::new()`].
    #[inline(always)]
    fn try_exp(self) -> Result<Self, expf::ExpfError> {
        self.try_exp_with::<{ expf::ExpfOptions::new().bits() }>()
    }

    /// Checked `e^self` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_exp_with<const CFG: u128>(self) -> Result<Self, expf::ExpfError> {
        self.eval_expf_const::<CFG, 1>()
    }

    /// Infallible `e^self` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn exp<const CFG: u128>(self) -> Self {
        const {
            let opts = expf::ExpfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "SimdMath::exp::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_exp_with::<CFG>` for checked options"
            );
        }
        match self.eval_expf_const::<CFG, 1>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `log2(self)` with default [`logf::LogfOptions::new()`].
    #[inline(always)]
    fn try_log2(self) -> Result<Self, logf::LogfError> {
        self.try_log2_with::<{ logf::LogfOptions::new().bits() }>()
    }

    /// Checked `log2(self)` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_log2_with<const CFG: u128>(self) -> Result<Self, logf::LogfError> {
        self.eval_logf_const::<CFG, 0>()
    }

    /// Infallible `log2(self)` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn log2<const CFG: u128>(self) -> Self {
        const {
            let opts = logf::LogfOptions::decode_const(CFG);
            opts.validate_op(0);
            assert!(
                opts.is_infallible(),
                "SimdMath::log2::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_log2_with::<CFG>` for checked options"
            );
        }
        match self.eval_logf_const::<CFG, 0>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `ln(self)` with default [`logf::LogfOptions::new()`].
    #[inline(always)]
    fn try_ln(self) -> Result<Self, logf::LogfError> {
        self.try_ln_with::<{ logf::LogfOptions::new().bits() }>()
    }

    /// Checked `ln(self)` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_ln_with<const CFG: u128>(self) -> Result<Self, logf::LogfError> {
        self.eval_logf_const::<CFG, 1>()
    }

    /// Infallible `ln(self)` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn ln<const CFG: u128>(self) -> Self {
        const {
            let opts = logf::LogfOptions::decode_const(CFG);
            opts.validate_op(1);
            assert!(
                opts.is_infallible(),
                "SimdMath::ln::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_ln_with::<CFG>` for checked options"
            );
        }
        match self.eval_logf_const::<CFG, 1>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `log10(self)` with default [`logf::LogfOptions::new()`].
    #[inline(always)]
    fn try_log10(self) -> Result<Self, logf::LogfError> {
        self.try_log10_with::<{ logf::LogfOptions::new().bits() }>()
    }

    /// Checked `log10(self)` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_log10_with<const CFG: u128>(self) -> Result<Self, logf::LogfError> {
        self.eval_logf_const::<CFG, 2>()
    }

    /// Infallible `log10(self)` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn log10<const CFG: u128>(self) -> Self {
        const {
            let opts = logf::LogfOptions::decode_const(CFG);
            opts.validate_op(2);
            assert!(
                opts.is_infallible(),
                "SimdMath::log10::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_log10_with::<CFG>` for checked options"
            );
        }
        match self.eval_logf_const::<CFG, 2>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `ln(1 + self)` with default [`logf::LogfOptions::new()`].
    #[inline(always)]
    fn try_log1p(self) -> Result<Self, logf::LogfError> {
        self.try_log1p_with::<{ logf::LogfOptions::new().bits() }>()
    }

    /// Checked `ln(1 + self)` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_log1p_with<const CFG: u128>(self) -> Result<Self, logf::LogfError> {
        self.eval_logf_const::<CFG, 3>()
    }

    /// Infallible `ln(1 + self)` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn log1p<const CFG: u128>(self) -> Self {
        const {
            let opts = logf::LogfOptions::decode_const(CFG);
            opts.validate_op(3);
            assert!(
                opts.is_infallible(),
                "SimdMath::log1p::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_log1p_with::<CFG>` for checked options"
            );
        }
        match self.eval_logf_const::<CFG, 3>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Checked `cbrt(self)` with default [`cbrtf::CbrtfOptions::new()`].
    #[inline(always)]
    fn try_cbrt(self) -> Result<Self, cbrtf::CbrtfError> {
        self.try_cbrt_with::<{ cbrtf::CbrtfOptions::new().bits() }>()
    }

    /// Checked `cbrt(self)` specialized for `const CFG: u128`.
    #[inline(always)]
    fn try_cbrt_with<const CFG: u128>(self) -> Result<Self, cbrtf::CbrtfError> {
        self.eval_cbrtf_const::<CFG>()
    }

    /// Infallible `cbrt(self)` specialized for `const CFG: u128`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()` / `.full_domain()`.
    #[inline(always)]
    fn cbrt<const CFG: u128>(self) -> Self {
        const {
            let opts = cbrtf::CbrtfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "SimdMath::cbrt::<CFG> requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`; use `try_cbrt_with::<CFG>` for checked options"
            );
        }
        match self.eval_cbrtf_const::<CFG>() {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }
}

macro_rules! impl_simd_vector {
    ($vec:ident, $bound:ident, $run_powf:ident, $run_expf:ident, $run_logf:ident, $run_cbrtf:ident, $lanes:literal) => {
        impl<T: magetypes::simd::backends::$bound> private::Sealed
            for magetypes::simd::generic::$vec<T>
        {
            #[inline(always)]
            fn eval_powf_const<const CFG: u128>(
                self,
                exponent: f32,
            ) -> Result<Self, powf::PowfError> {
                powf::const_api::$run_powf::<CFG, T>(self.token(), self, exponent)
            }

            #[inline(always)]
            fn eval_powf_runtime(
                self,
                exponent: f32,
                options: powf::PowfOptions,
            ) -> Result<Self, powf::PowfError> {
                let token = self.token();
                powf::const_api::runtime::<T, $lanes>(token, self.to_array(), exponent, options)
                    .map(|arr| magetypes::simd::generic::$vec::from_array_t(token, arr))
            }

            #[inline(always)]
            fn eval_expf_const<const CFG: u128, const OP: u8>(
                self,
            ) -> Result<Self, expf::ExpfError> {
                expf::const_api::$run_expf::<CFG, OP, T>(self.token(), self)
            }

            #[inline(always)]
            fn eval_expf_runtime<const OP: u8>(
                self,
                options: expf::ExpfOptions,
            ) -> Result<Self, expf::ExpfError> {
                let token = self.token();
                expf::const_api::runtime_token::<T, OP, $lanes>(token, self.to_array(), options)
                    .map(|arr| magetypes::simd::generic::$vec::from_array_t(token, arr))
            }

            #[inline(always)]
            fn eval_logf_const<const CFG: u128, const OP: u8>(
                self,
            ) -> Result<Self, logf::LogfError> {
                logf::const_api::$run_logf::<CFG, OP, T>(self.token(), self)
            }

            #[inline(always)]
            fn eval_logf_runtime<const OP: u8>(
                self,
                options: logf::LogfOptions,
            ) -> Result<Self, logf::LogfError> {
                let token = self.token();
                logf::const_api::runtime_token::<T, OP, $lanes>(token, self.to_array(), options)
                    .map(|arr| magetypes::simd::generic::$vec::from_array_t(token, arr))
            }

            #[inline(always)]
            fn eval_cbrtf_const<const CFG: u128>(self) -> Result<Self, cbrtf::CbrtfError> {
                cbrtf::const_api::$run_cbrtf::<CFG, T>(self.token(), self)
            }

            #[inline(always)]
            fn eval_cbrtf_runtime(
                self,
                options: cbrtf::CbrtfOptions,
            ) -> Result<Self, cbrtf::CbrtfError> {
                let token = self.token();
                cbrtf::const_api::runtime_token::<T, $lanes>(token, self.to_array(), options)
                    .map(|arr| magetypes::simd::generic::$vec::from_array_t(token, arr))
            }
        }

        impl<T: magetypes::simd::backends::$bound> SimdVector
            for magetypes::simd::generic::$vec<T>
        {
        }

        impl<T: magetypes::simd::backends::$bound> SimdMath for magetypes::simd::generic::$vec<T> {}
    };
}

impl_simd_vector!(f32x4, F32x4Convert, run4, run4, run4, run4, 4);
impl_simd_vector!(f32x8, F32x8Convert, run8, run8, run8, run8, 8);
#[cfg(feature = "w512")]
impl_simd_vector!(f32x16, F32x16Convert, run16, run16, run16, run16, 16);

pub mod cbrtf;
pub mod expf;
pub mod logf;
pub mod powf;
mod soft_fma;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)] // The copy also contains vector-primitive helpers not used here.
mod wasm_fma;
