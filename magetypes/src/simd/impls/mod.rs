//! Backend trait implementations for each token type.
//!
//! Each file implements the backend traits (e.g., `F32x8Backend`) for one
//! token, using that platform's native intrinsics.

#[cfg(target_arch = "x86_64")]
mod x86_v3;
