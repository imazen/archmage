//! Configurable f32 logarithms with explicit portability, precision and
//! validation, mirroring [`crate::powf`].
//!
//! MIDP and LOWP evaluate the existing `log2_midp`/`log2_lowp` approximations
//! on `log2(x)`, with `ln`/`log10` as scale-by-constant aliases. `Positive`
//! rejects `x <= 0`, subnormals, NaN and infinities under Checked; `Specials`
//! returns the IEEE specials (`0 -> -inf`, `<0 -> NaN`, NaN -> NaN,
//! `+inf -> +inf`). `AssumeValid` omits checks while retaining memory safety.
//! Existing `log2_lowp`/`log2_midp` methods retain their existing behavior.

pub(crate) mod const_api;
mod dispatch;
mod graph;
mod options;
#[cfg(test)]
mod tests;
pub use const_api::{
    try_log10_array_with, try_log10_runtime, try_log10_with, try_log10_with_token,
    try_log1p_array_runtime, try_log1p_array_with, try_log1p_runtime, try_log1p_with,
    try_log1p_with_token, try_log2_array_with, try_log2_runtime, try_log2_with,
    try_log2_with_token, try_ln_array_with, try_ln_runtime, try_ln_with, try_ln_with_token,
};
pub use options::{
    LogfDomain, LogfExecution, LogfOptions, LogfPortability, LogfPrecision, LogfValidation,
};

/// Reason a complete vector was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogfErrorKind {
    /// An input fails the domain: `x <= 0`, subnormal, NaN or infinite.
    InvalidInput,
    /// The array width is not 1, 4, 8, or enabled width 16.
    UnsupportedWidth,
    /// An unsupported combination of precision, portability, domain or features.
    UnsupportedOptions,
}

/// Whole-vector rejection. No partial result is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogfError {
    /// Which part of the contract failed.
    pub kind: LogfErrorKind,
    /// Lanes failing the reported stage, low bit first.
    pub lane_mask: u16,
}

impl core::fmt::Display for LogfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "logf {:?}, lane mask {:#06x}", self.kind, self.lane_mask)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LogfError {}

/// Positive-domain scalar log2 using the default checked MIDP domain.
#[inline(always)]
pub fn try_log2(x: f32) -> Result<f32, LogfError> {
    try_log2_with::<{ LogfOptions::DEFAULT }>(x)
}

/// Positive-domain scalar ln using the default checked MIDP domain.
#[inline(always)]
pub fn try_ln(x: f32) -> Result<f32, LogfError> {
    try_ln_with::<{ LogfOptions::DEFAULT }>(x)
}

/// Positive-domain scalar log10 using the default checked MIDP domain.
#[inline(always)]
pub fn try_log10(x: f32) -> Result<f32, LogfError> {
    try_log10_with::<{ LogfOptions::DEFAULT }>(x)
}
