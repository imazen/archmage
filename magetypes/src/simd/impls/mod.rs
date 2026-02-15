//! Backend trait implementations for each token type.
//!
//! Each file implements the backend traits (e.g., `F32x8Backend`) for one
//! token, using that platform's native intrinsics.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "x86_64")]
mod x86_v3;

#[cfg(target_arch = "aarch64")]
mod arm_neon;

#[cfg(target_arch = "wasm32")]
mod wasm128;

mod scalar;
