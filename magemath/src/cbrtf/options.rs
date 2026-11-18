//! Fluent `const` configuration builder for the cube-root (`cbrtf`) family.

use crate::{Execution, Portability, Validation};

const MAX_BOUNDED: f32 = f32::MAX / 3.0;

const MAGIC: u128 = 0x43;
const HAS_RANGE: u128 = 1 << 8;
const FULL: u128 = 1 << 9;
const LOWP: u128 = 1 << 10;
const BACKEND: u128 = 1 << 11;
const ASSUME: u128 = 1 << 12;
const CALLER: u128 = 1 << 13;
const SUBNORM: u128 = 1 << 14;
const CLAMP_OOB: u128 = 1 << 15;
const RESERVED_MASK: u128 = ((1_u128 << 48) - 1) << 16;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

const fn canon_zero(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

const fn is_nonzero_subnormal_only_interval(lo: f32, hi: f32) -> bool {
    (lo > 0.0 && hi < f32::MIN_POSITIVE) || (lo > -f32::MIN_POSITIVE && hi < 0.0)
}

/// Approximation family, not a global mathematical error bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum CbrtfPrecision {
    /// Two Halley iterations after the SIMD Kahan seed (`<= 3.2 ULP`).
    #[default]
    Midp,
    /// One Halley iteration after the SIMD Kahan seed (`<= 259 ULP`, `~3e-5` rel err).
    Lowp,
}

/// Fluent `const` configuration builder for [`super::Cbrtf`] and [`crate::SimdMath`].
///
/// Options separate **what inputs are supported** ([`input_range`](Self::input_range),
/// [`admit_subnormals`](Self::admit_subnormals), [`unit_range`](Self::unit_range),
/// and [`admit_all_floats`](Self::admit_all_floats)) from **how out-of-bounds
/// inputs are handled**:
///
/// - [`error_on_out_of_bounds`](Self::error_on_out_of_bounds) (default, alias
///   [`checked`](Self::checked)): return `Err` on out-of-domain inputs.
/// - [`clamp_out_of_bounds`](Self::clamp_out_of_bounds): clamp non-NaN inputs
///   into the configured domain and propagate `NaN` as `NaN` (infallible).
/// - [`assume_in_bounds`](Self::assume_in_bounds) (alias [`unchecked`](Self::unchecked)):
///   omit runtime checks and clamps when the caller already guarantees bounds
///   (infallible).
///
/// Declaring [`input_range`](Self::input_range) with `lo >= f32::MIN_POSITIVE`
/// (or `hi <= -f32::MIN_POSITIVE`) proves subnormals cannot occur and elides
/// per-lane subnormal handling at compile time. When `[-1, 1]` perceptual
/// color inputs may include dark subnormals, [`unit_range`](Self::unit_range)
/// or [`admit_subnormals`](Self::admit_subnormals) adds exact `* 2^24` / `* 2^-8`
/// subnormal scaling without paying for full-domain `>= 1e36` / `±inf` handling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CbrtfOptions {
    input: Option<(u32, u32)>,
    subnormals: bool,
    full: bool,
    precision: CbrtfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl Default for CbrtfOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CbrtfOptions {
    /// Bounded domain (`±0` and normals strictly below `f32::MAX / 3`), `Midp`,
    /// `BitExact`, `Checked` (`error_on_out_of_bounds`), `Execution::Upgrade`.
    pub const fn new() -> Self {
        Self {
            input: None,
            subnormals: false,
            full: false,
            precision: CbrtfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Upgrade,
        }
    }

    /// Restrict inputs to `[lo, hi]` within the active cube-root domain
    /// (`±0` and normals strictly below `f32::MAX / 3` unless
    /// [`admit_subnormals`](Self::admit_subnormals) is enabled).
    ///
    /// Both bounds must be finite `f32`s strictly below `f32::MAX / 3` in
    /// magnitude with `lo <= hi`. Intervals containing only nonzero subnormals
    /// require [`admit_subnormals`](Self::admit_subnormals).
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) < MAX_BOUNDED && abs(hi) < MAX_BOUNDED,
            "cbrtf input range must satisfy -f32::MAX/3 < lo <= hi < f32::MAX/3"
        );
        self.input = Some((canon_zero(lo).to_bits(), canon_zero(hi).to_bits()));
        self
    }

    /// Also accept nonzero subnormal inputs (`0 < |x| < f32::MIN_POSITIVE`),
    /// rescaling them accurately by `* 2^24` before the Kahan seed and `* 2^-8`
    /// afterward without paying full-domain `>= 1e36` / `±inf` overhead.
    pub const fn admit_subnormals(mut self) -> Self {
        if !self.full {
            self.subnormals = true;
        }
        self
    }

    /// Signed unit cube-root domain `[-1.0, 1.0]` including `±0.0` and
    /// subnormals (ideal for Oklab / perceptual color transforms):
    /// `.input_range(-1.0, 1.0).admit_subnormals()`.
    pub const fn unit_range(self) -> Self {
        self.input_range(-1.0, 1.0).admit_subnormals()
    }

    /// Accept every `f32` input (rescaling subnormals and large magnitudes,
    /// propagating `±0`, `±inf`, and `NaN`).
    pub const fn admit_all_floats(mut self) -> Self {
        self.full = true;
        self.subnormals = false;
        self
    }

    /// Alias for [`admit_all_floats`](Self::admit_all_floats).
    pub const fn full_domain(self) -> Self {
        self.admit_all_floats()
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: CbrtfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Shorthand for `.precision(CbrtfPrecision::Lowp)`.
    pub const fn lowp(self) -> Self {
        self.precision(CbrtfPrecision::Lowp)
    }

    /// Shorthand for `.precision(CbrtfPrecision::Midp)`.
    pub const fn midp(self) -> Self {
        self.precision(CbrtfPrecision::Midp)
    }

    /// Select cross-backend agreement.
    pub const fn portability(mut self, value: Portability) -> Self {
        self.portability = value;
        self
    }

    /// Shorthand for `.portability(Portability::BitExact)`.
    pub const fn bit_exact(self) -> Self {
        self.portability(Portability::BitExact)
    }

    /// Shorthand for `.portability(Portability::BackendDependent)`.
    pub const fn backend_dependent(self) -> Self {
        self.portability(Portability::BackendDependent)
    }

    /// Select how out-of-bounds inputs are handled.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Return `Err` when any input falls outside the configured domain
    /// (`Validation::Checked`, default).
    pub const fn error_on_out_of_bounds(self) -> Self {
        self.validation(Validation::Checked)
    }

    /// Alias for [`error_on_out_of_bounds`](Self::error_on_out_of_bounds).
    pub const fn checked(self) -> Self {
        self.error_on_out_of_bounds()
    }

    /// Clamp finite/infinite out-of-bounds inputs into the configured domain
    /// and propagate `NaN` as `NaN` (`Validation::ClampOutOfBounds`, infallible).
    pub const fn clamp_out_of_bounds(self) -> Self {
        self.validation(Validation::ClampOutOfBounds)
    }

    /// Omit numerical input checks and clamps; caller guarantees all inputs lie
    /// within the configured domain (`Validation::AssumeValid`, infallible).
    pub const fn assume_in_bounds(self) -> Self {
        self.validation(Validation::AssumeValid)
    }

    /// Alias for [`assume_in_bounds`](Self::assume_in_bounds).
    pub const fn unchecked(self) -> Self {
        self.assume_in_bounds()
    }

    /// Select whole-operation backend dispatch.
    pub const fn execution(mut self, value: Execution) -> Self {
        self.execution = value;
        self
    }

    /// Shorthand for `.execution(Execution::Upgrade)`.
    pub const fn upgrade(self) -> Self {
        self.execution(Execution::Upgrade)
    }

    /// Shorthand for `.execution(Execution::Caller)`.
    pub const fn caller(self) -> Self {
        self.execution(Execution::Caller)
    }

    /// Whether this configuration is self-consistent.
    pub const fn is_valid(&self) -> bool {
        if self.full && self.input.is_some() {
            return false;
        }
        if !self.admits_subnormals()
            && let Some((lo_b, hi_b)) = self.input
            && is_nonzero_subnormal_only_interval(f32::from_bits(lo_b), f32::from_bits(hi_b))
        {
            return false;
        }
        true
    }

    /// Assert that this configuration is self-consistent.
    pub const fn validate(&self) {
        assert!(
            !(self.full && self.input.is_some()),
            "full_domain cannot be combined with input_range"
        );
        if !self.admits_subnormals()
            && let Some((lo_b, hi_b)) = self.input
        {
            assert!(
                !is_nonzero_subnormal_only_interval(f32::from_bits(lo_b), f32::from_bits(hi_b)),
                "cbrtf input_range contains only nonzero subnormals; add .admit_subnormals() or include 0.0 / normal bounds"
            );
        }
    }

    /// Whether the declared range excludes the subnormal band around zero
    /// (or subnormals are admitted), eliding the per-lane subnormal rejection check.
    pub const fn elides_denorm_check(&self) -> bool {
        !self.denorm_possible()
    }

    /// Whether this configuration can never return an input error
    /// (`.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or
    /// `.admit_all_floats()` / `.full_domain()`).
    pub const fn is_infallible(&self) -> bool {
        self.assume_valid() || self.clamp_oob() || self.full
    }

    /// Pack this configuration into a `u128` const-generic argument.
    pub const fn bits(self) -> u128 {
        let mut out = MAGIC;
        if let Some((lo, hi)) = self.input {
            out |= HAS_RANGE | ((lo as u128) << 64) | ((hi as u128) << 96);
        }
        if self.subnormals {
            out |= SUBNORM;
        }
        if self.full {
            out |= FULL;
        }
        if matches!(self.precision, CbrtfPrecision::Lowp) {
            out |= LOWP;
        }
        if matches!(self.portability, Portability::BackendDependent) {
            out |= BACKEND;
        }
        match self.validation {
            Validation::Checked => {}
            Validation::ClampOutOfBounds => out |= CLAMP_OOB,
            Validation::AssumeValid => out |= ASSUME,
        }
        if matches!(self.execution, Execution::Caller) {
            out |= CALLER;
        }
        out
    }

    const fn from_bits_raw(bits: u128) -> Option<Self> {
        if (bits & 0xff) != MAGIC || (bits & RESERVED_MASK) != 0 {
            return None;
        }
        if (bits & ASSUME) != 0 && (bits & CLAMP_OOB) != 0 {
            return None;
        }
        let full = (bits & FULL) != 0;
        let subnormals = (bits & SUBNORM) != 0;
        if full && subnormals {
            return None;
        }
        let lo_b = ((bits >> 64) & 0xffff_ffff) as u32;
        let hi_b = ((bits >> 96) & 0xffff_ffff) as u32;
        let input = if (bits & HAS_RANGE) != 0 {
            if lo_b == 0x8000_0000 || hi_b == 0x8000_0000 {
                return None;
            }
            let lo = f32::from_bits(lo_b);
            let hi = f32::from_bits(hi_b);
            if !(lo <= hi && abs(lo) < MAX_BOUNDED && abs(hi) < MAX_BOUNDED) {
                return None;
            }
            Some((lo_b, hi_b))
        } else {
            if lo_b != 0 || hi_b != 0 {
                return None;
            }
            None
        };
        let validation = if (bits & ASSUME) != 0 {
            Validation::AssumeValid
        } else if (bits & CLAMP_OOB) != 0 {
            Validation::ClampOutOfBounds
        } else {
            Validation::Checked
        };
        Some(Self {
            input,
            subnormals,
            full,
            precision: if (bits & LOWP) != 0 {
                CbrtfPrecision::Lowp
            } else {
                CbrtfPrecision::Midp
            },
            portability: if (bits & BACKEND) != 0 {
                Portability::BackendDependent
            } else {
                Portability::BitExact
            },
            validation,
            execution: if (bits & CALLER) != 0 {
                Execution::Caller
            } else {
                Execution::Upgrade
            },
        })
    }

    /// Decode a `u128` option word back into [`CbrtfOptions`].
    pub const fn from_bits(bits: u128) -> Option<Self> {
        let Some(opts) = Self::from_bits_raw(bits) else {
            return None;
        };
        if opts.is_valid() { Some(opts) } else { None }
    }

    pub(crate) const fn decode_const(bits: u128) -> Self {
        let Some(opts) = Self::from_bits_raw(bits) else {
            panic!("invalid cbrtf const options bit pattern");
        };
        opts.validate();
        opts
    }

    /// Create a dynamic runtime evaluator for this [`CbrtfOptions`] value.
    #[inline(always)]
    pub const fn runtime(self) -> super::CbrtfRuntime {
        super::CbrtfRuntime::new(self)
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        match self.input {
            Some((lo, hi)) => Some((f32::from_bits(lo), f32::from_bits(hi))),
            None => None,
        }
    }
    const fn range_excludes_subnormals(&self) -> bool {
        match self.input_bounds() {
            Some((lo, hi)) => lo >= f32::MIN_POSITIVE || hi <= -f32::MIN_POSITIVE,
            None => false,
        }
    }
    pub(crate) const fn denorm_possible(&self) -> bool {
        !self.range_excludes_subnormals() && !self.admits_subnormals()
    }
    pub(crate) const fn scale_subnormals(&self) -> bool {
        self.subnormals && !self.range_excludes_subnormals()
    }
    pub(crate) const fn admits_subnormals(&self) -> bool {
        self.subnormals || self.full
    }
    pub(crate) const fn is_full(&self) -> bool {
        self.full
    }
    pub(crate) const fn is_lowp(&self) -> bool {
        matches!(self.precision, CbrtfPrecision::Lowp)
    }
    pub(crate) const fn is_backend(&self) -> bool {
        matches!(self.portability, Portability::BackendDependent)
    }
    pub(crate) const fn clamp_oob(&self) -> bool {
        matches!(self.validation, Validation::ClampOutOfBounds)
    }
    pub(crate) const fn assume_valid(&self) -> bool {
        matches!(self.validation, Validation::AssumeValid)
    }
    pub(crate) const fn is_caller(&self) -> bool {
        matches!(self.execution, Execution::Caller)
    }
}
