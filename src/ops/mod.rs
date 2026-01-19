//! Token-gated SIMD operations
//!
//! Operations that require a capability token to call. The token proves
//! the CPU feature is available, making the operation safe.

#[cfg(target_arch = "x86_64")]
pub mod x86;
