//! Backend trait implementations for each token type.
//!
//! Each file implements the backend traits (e.g., `F32x8Backend`) for one
//! token, using that platform's native intrinsics.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_v3;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod x86_v4;

// Hand-written AVX-512VL pixel-packing overrides called by `Sealed` hooks.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod x86_v4_f32_overrides;

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm_neon;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm128;
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
pub(crate) mod wasm128_fma_overrides;

pub(crate) mod scalar;

// Single blanket impls over `SimdToken + Sealed` for all 45 unified
// `*Backend` / `*Convert` / `*Bitcast` traits (plus `F64x8Convert`).
mod delegated;
