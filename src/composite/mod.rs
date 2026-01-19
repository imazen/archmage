//! Composite SIMD operations
//!
//! Higher-level operations built from primitives, all token-gated.

#[cfg(target_arch = "x86_64")]
pub mod transpose;

#[cfg(target_arch = "x86_64")]
pub mod dot_product;

#[cfg(target_arch = "x86_64")]
pub mod horizontal;

// Re-exports
#[cfg(target_arch = "x86_64")]
pub use transpose::*;

#[cfg(target_arch = "x86_64")]
pub use dot_product::*;

#[cfg(target_arch = "x86_64")]
pub use horizontal::*;
