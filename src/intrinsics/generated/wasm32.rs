//! Combined `core::arch` + `safe_unaligned_simd` intrinsics for `wasm32`.
//!
//! **Auto-generated** by `cargo xtask generate` — do not edit manually.
//!
//! This module glob-imports all of `core::arch::wasm32` (types, value intrinsics,
//! and unsafe memory ops), then explicitly re-exports the safe reference-based
//! memory operations from `safe_unaligned_simd`. Rust's name resolution rules
//! make explicit imports shadow glob imports, so `_mm256_loadu_ps` etc. resolve
//! to the safe versions automatically.

#[allow(unused_imports)]
pub use core::arch::wasm32::*;

#[allow(unused_imports)]
pub use safe_unaligned_simd::wasm32::{
    i16x8_load_extend_i8x8, i16x8_load_extend_u8x8, i32x4_load_extend_i16x4,
    i32x4_load_extend_u16x4, i64x2_load_extend_i32x2, i64x2_load_extend_u32x2,
    u16x8_load_extend_u8x8, u32x4_load_extend_u16x4, u64x2_load_extend_u32x2, v128_load,
    v128_load8_splat, v128_load16_splat, v128_load32_splat, v128_load32_zero, v128_load64_splat,
    v128_load64_zero, v128_store,
};
