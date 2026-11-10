//! Configurable f32 exponentials with explicit portability, precision and
//! validation, mirroring [`crate::powf`].
//!
//! MIDP and LOWP evaluate the existing `exp2_midp`/`exp2_lowp` approximations
//! on `exp2(x)` and `exp2(x * log2(e))`; BitExact selects one fused rounding
//! sequence on every backend, BackendDependent keeps the caller's relaxed
//! arithmetic. `Bounded` rejects NaN and infinities under Checked; `Full`
//! propagates them to their natural results (NaN -> NaN, `+inf` -> `+inf`,
//! `-inf` -> `+0`). `AssumeValid` omits checks while retaining memory safety.
//! Existing `exp2_lowp`/`exp2_midp` methods retain their existing behavior.

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;
pub use const_api::{
    try_exp_array_with, try_exp_runtime, try_exp_with, try_exp_with_token, try_exp2_array_with,
    try_exp2_runtime, try_exp2_with, try_exp2_with_token,
};
pub use options::{
    ExpfDomain, ExpfExecution, ExpfOptions, ExpfPortability, ExpfPrecision, ExpfValidation,
};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpfErrorKind {
    /// An input is NaN or infinite in a domain that rejects non-finite inputs.
    InvalidInput,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpfError {
    /// Which part of the contract failed.
    pub kind: ExpfErrorKind,
    /// Lanes failing the reported stage, low bit first.
    pub lane_mask: u16,
}

impl core::fmt::Display for ExpfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "expf {:?}, lane mask {:#06x}", self.kind, self.lane_mask)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ExpfError {}

/// Bounded scalar exp2 using the default checked MIDP domain.
#[inline(always)]
pub fn try_exp2(x: f32) -> Result<f32, ExpfError> {
    try_exp2_with::<{ ExpfOptions::DEFAULT }>(x)
}

/// Bounded scalar exp using the default checked MIDP domain.
#[inline(always)]
pub fn try_exp(x: f32) -> Result<f32, ExpfError> {
    try_exp_with::<{ ExpfOptions::DEFAULT }>(x)
}
