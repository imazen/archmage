//! Composite SIMD operations
//!
//! Higher-level operations built from primitives, all token-gated.

// Operation traits
pub mod scalar_ops;
pub mod simd_ops;

// Implementations (x86_64 only for now)
#[cfg(target_arch = "x86_64")]
pub mod transpose;

#[cfg(target_arch = "x86_64")]
pub mod dot_product;

#[cfg(target_arch = "x86_64")]
pub mod horizontal;

// Token trait implementations
#[cfg(target_arch = "x86_64")]
mod x86_impls;

// Re-export traits
pub use scalar_ops::*;
pub use simd_ops::*;

// Re-export implementations
#[cfg(target_arch = "x86_64")]
pub use transpose::transpose_8x8;

#[cfg(target_arch = "x86_64")]
pub use dot_product::dot_product_f32;

#[cfg(target_arch = "x86_64")]
pub use horizontal::*;
