//! Backend trait implementations for each token type.
//!
//! Each file implements the backend traits (e.g., `F32x8Backend`) for one
//! token, using that platform's native intrinsics.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "x86_64")]
mod x86_v3;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod x86_v4;

// The AVX-512 tokens' W128 / W256 f32 backends, forwarded to V3's
// (generated from the trait definitions; V4 ⊃ V3, so delegating is
// sound), plus the hand-written AVX-512VL pixel-packing overrides.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod x86_v4_f32_delegated;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod x86_v4_f32_overrides;

#[cfg(target_arch = "aarch64")]
mod arm_neon;

#[cfg(target_arch = "wasm32")]
mod wasm128;

mod scalar;
