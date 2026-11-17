//! Typed range-as-data profiles for the cube-root (`cbrtf`) family.

use super::{CbrtfDomain, CbrtfOptions, CbrtfPrecision};
use crate::{Execution, Portability, Validation};

const MAX_BOUNDED: f32 = f32::MAX / 3.0;

const fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// A cube-root configuration built in a `const`.
///
/// Fields are private so additional policy axes can be added without breaking
/// existing [`CbrtfProfile`] implementations.
#[derive(Clone, Copy, Debug)]
pub struct CbrtfSpec {
    input: Option<(f32, f32)>,
    full: bool,
    precision: CbrtfPrecision,
    portability: Portability,
    validation: Validation,
    execution: Execution,
}

impl CbrtfSpec {
    /// Bounded domain (`±0` and normals strictly below `f32::MAX / 3`), MIDP,
    /// BitExact, Checked, Automatic execution.
    pub const fn new() -> Self {
        Self {
            input: None,
            full: false,
            precision: CbrtfPrecision::Midp,
            portability: Portability::BitExact,
            validation: Validation::Checked,
            execution: Execution::Automatic,
        }
    }

    /// Translate an encoded [`CbrtfOptions`] value into an equivalent [`CbrtfSpec`].
    pub const fn from_options(options: CbrtfOptions) -> Self {
        Self::new()
            .domain(options.domain)
            .precision(options.precision)
            .portability(options.portability)
            .validation(options.validation)
            .execution(options.execution)
    }

    pub(crate) const fn from_bits_const(bits: u32) -> Self {
        let Some(opts) = CbrtfOptions::from_bits(bits) else {
            panic!("invalid cbrtf const options");
        };
        Self::from_options(opts)
    }

    /// Accept inputs in `[lo, hi]`. Both must be finite `f32`s strictly below
    /// `f32::MAX / 3` in magnitude with `lo <= hi`.
    pub const fn input_range(mut self, lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi && abs(lo) < MAX_BOUNDED && abs(hi) < MAX_BOUNDED,
            "cbrtf input range must satisfy -f32::MAX/3 < lo <= hi < f32::MAX/3"
        );
        self.input = Some((lo, hi));
        self
    }

    /// Select the domain (`Bounded` or `Full`).
    pub const fn domain(mut self, value: CbrtfDomain) -> Self {
        self.full = matches!(value, CbrtfDomain::Full);
        self
    }

    /// Accept every `f32` input (rescaling subnormals and large magnitudes).
    pub const fn full_domain(mut self) -> Self {
        self.full = true;
        self
    }

    /// Select the approximation family (`Midp` or `Lowp`).
    pub const fn precision(mut self, value: CbrtfPrecision) -> Self {
        self.precision = value;
        self
    }

    /// Select cross-backend agreement.
    pub const fn portability(mut self, value: Portability) -> Self {
        self.portability = value;
        self
    }

    /// Select numerical precondition enforcement.
    pub const fn validation(mut self, value: Validation) -> Self {
        self.validation = value;
        self
    }

    /// Select whole-operation backend dispatch.
    pub const fn execution(mut self, value: Execution) -> Self {
        self.execution = value;
        self
    }

    /// Validate that this profile is self-consistent.
    pub const fn validate(&self) {
        assert!(
            !(self.full && self.input.is_some()),
            "full_domain cannot be combined with input_range"
        );
    }

    pub(crate) const fn input_bounds(&self) -> Option<(f32, f32)> {
        self.input
    }
    pub(crate) const fn denorm_possible(&self) -> bool {
        match self.input {
            Some((lo, hi)) => lo < f32::MIN_POSITIVE && hi > -f32::MIN_POSITIVE,
            None => !self.full,
        }
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
    pub(crate) const fn assume_valid(&self) -> bool {
        matches!(self.validation, Validation::AssumeValid)
    }
    pub(crate) const fn is_caller(&self) -> bool {
        matches!(self.execution, Execution::Caller)
    }
}

impl Default for CbrtfSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A named cube-root configuration evaluated at compile time.
pub trait CbrtfProfile: 'static {
    /// The configuration specification.
    const SPEC: CbrtfSpec;
}

/// Profile adapter constructed from legacy [`CbrtfOptions::bits()`] integer bits.
#[derive(Clone, Copy, Debug)]
pub struct Options<const BITS: u32>;

impl<const BITS: u32> CbrtfProfile for Options<BITS> {
    const SPEC: CbrtfSpec = CbrtfSpec::from_bits_const(BITS);
}

impl<P: CbrtfProfile> CbrtfProfile for crate::Caller<P> {
    const SPEC: CbrtfSpec = P::SPEC.execution(Execution::Caller);
}

impl<P: CbrtfProfile> CbrtfProfile for crate::Unchecked<P> {
    const SPEC: CbrtfSpec = P::SPEC.validation(Validation::AssumeValid);
}

impl<P: CbrtfProfile> CbrtfProfile for crate::Backend<P> {
    const SPEC: CbrtfSpec = P::SPEC.portability(Portability::BackendDependent);
}

/// Built-in cube-root profiles mirroring the standard presets.
pub mod presets {
    use super::{CbrtfOptions, CbrtfProfile, CbrtfSpec};

    /// BitExact MIDP, bounded normal domain, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct Midp;
    impl CbrtfProfile for Midp {
        const SPEC: CbrtfSpec = CbrtfSpec::from_options(CbrtfOptions::PORTABLE_MIDP);
    }

    /// BitExact LOWP, bounded normal domain, checked, automatic execution.
    #[derive(Clone, Copy, Debug)]
    pub struct Lowp;
    impl CbrtfProfile for Lowp {
        const SPEC: CbrtfSpec = CbrtfSpec::from_options(CbrtfOptions::PORTABLE_LOWP);
    }

    /// BitExact MIDP accepting every `f32` input (subnormals, large normals, specials).
    #[derive(Clone, Copy, Debug)]
    pub struct MidpFull;
    impl CbrtfProfile for MidpFull {
        const SPEC: CbrtfSpec = CbrtfSpec::from_options(CbrtfOptions::PORTABLE_MIDP_FULL);
    }

    /// BitExact LOWP accepting every `f32` input.
    #[derive(Clone, Copy, Debug)]
    pub struct LowpFull;
    impl CbrtfProfile for LowpFull {
        const SPEC: CbrtfSpec = CbrtfSpec::from_options(CbrtfOptions::PORTABLE_LOWP_FULL);
    }
}
