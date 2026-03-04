//! Combined intrinsics: `core::arch` + `safe_unaligned_simd` in one namespace.
//!
//! Each sub-module glob-imports `core::arch::{arch}::*` for types and value
//! intrinsics, then explicitly re-exports `safe_unaligned_simd` functions.
//! Rust's name resolution makes explicit imports shadow glob imports, so safe
//! reference-based memory ops (e.g., `_mm256_loadu_ps(&data)`) win over the
//! unsafe pointer-based versions automatically.
//!
//! **Auto-generated** by `cargo xtask generate` — do not edit the `generated/`
//! subdirectory manually.
//!
//! # Usage
//!
//! ```rust,ignore
//! use archmage::intrinsics::x86_64::*;
//!
//! // Types, value intrinsics, AND safe memory ops — all in one namespace.
//! let v = _mm256_loadu_ps(data);  // safe_unaligned_simd version (takes reference)
//! let doubled = _mm256_add_ps(v, v);  // core::arch version (value op)
//! ```

mod generated {
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    pub mod aarch64;
    #[cfg(target_arch = "wasm32")]
    pub mod wasm32;
    #[cfg(target_arch = "x86")]
    pub mod x86;
    #[cfg(target_arch = "x86_64")]
    pub mod x86_64;
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub use generated::aarch64;
#[cfg(target_arch = "wasm32")]
pub use generated::wasm32;
#[cfg(target_arch = "x86")]
pub use generated::x86;
#[cfg(target_arch = "x86_64")]
pub use generated::x86_64;
