//! Architecture-specific code generation.
//!
//! Each architecture module provides intrinsic names, types, and prefixes
//! for generating SIMD code.

pub mod arm;
pub mod x86;

use super::types::{ElementType, SimdWidth};

/// Architecture-specific code generation helpers
pub trait Arch {
    /// Target architecture name for cfg attribute
    fn target_arch() -> &'static str;

    /// Get the intrinsic type name for a given element type and width
    fn intrinsic_type(elem: ElementType, width: SimdWidth) -> &'static str;

    /// Get the intrinsic prefix for a given width (e.g., "_mm256")
    fn prefix(width: SimdWidth) -> &'static str;

    /// Get the type suffix for intrinsics (e.g., "ps" for f32)
    fn suffix(elem: ElementType) -> &'static str;

    /// Get the suffix for min/max operations (handles signed vs unsigned)
    fn minmax_suffix(elem: ElementType) -> &'static str;

    /// Required token for this width
    fn required_token(width: SimdWidth, needs_int_ops: bool) -> &'static str;

    /// Required feature flag (e.g., "avx512")
    fn required_feature(width: SimdWidth) -> Option<&'static str>;

    /// Whether this width is supported on this architecture
    fn supports_width(width: SimdWidth) -> bool;
}
