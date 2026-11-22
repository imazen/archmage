//! Backend trait implementations for each token type.
//!
//! Each file implements the backend traits (e.g., `F32x8Backend`) for one
//! token, using that platform's native intrinsics.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod x86_v3;

// Pure-V3 W128 / W256 backend delegation: all 34 narrow traits for
// `X64V3VnniToken` (independent of `feature = "avx512"`) plus the 30
// non-f32 narrow traits for the AVX-512 tokens under `feature = "avx512"`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod x86_v3_delegated;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
mod x86_v4;

// `X64V4xToken`'s base W512 backends, forwarded to `X64V4Token` via
// `self.v4()` (V4x ⊃ V4, zero `unsafe`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    feature = "avx512",
    feature = "w512"
))]
mod x86_v4_w512_delegated;

// The AVX-512 tokens' W128 / W256 f32 backends, forwarded to V3's
// (generated from the trait definitions; V4 ⊃ V3, so delegating is
// sound), plus the hand-written AVX-512VL pixel-packing overrides.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
mod x86_v4_f32_delegated;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
mod x86_v4_f32_overrides;

#[cfg(target_arch = "aarch64")]
mod arm_neon;

#[cfg(target_arch = "wasm32")]
mod wasm128;

mod scalar;
