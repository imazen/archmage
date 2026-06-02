# Wasm128RelaxedToken — WASM Relaxed SIMD

Proof that WASM Relaxed SIMD is available.

**Architecture:** wasm32 | **Features:** simd128, relaxed-simd
**Total intrinsics:** 28 (28 safe, 0 unsafe, 28 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Wasm128Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Wasm128Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Wasm128Token, chunk: &mut [f32; 4]) {
    let v = v128_load(chunk);  // safe!
    let doubled = f32x4_add(v, v);  // value intrinsic (safe inside #[rite])
    v128_store(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (28 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `f32x4_relaxed_madd` | Computes `a * b + c` with either one rounding or two roundin... |  | — |
| `f32x4_relaxed_max` | A relaxed version of `f32x4_max` which has implementation-sp... |  | — |
| `f32x4_relaxed_min` | A relaxed version of `f32x4_min` which has implementation-sp... |  | — |
| `f32x4_relaxed_nmadd` | Computes `-a * b + c` with either one rounding or two roundi... |  | — |
| `f64x2_relaxed_madd` | Computes `a * b + c` with either one rounding or two roundin... |  | — |
| `f64x2_relaxed_max` | A relaxed version of `f64x2_max` which has implementation-sp... |  | — |
| `f64x2_relaxed_min` | A relaxed version of `f64x2_min` which has implementation-sp... |  | — |
| `f64x2_relaxed_nmadd` | Computes `-a * b + c` with either one rounding or two roundi... |  | — |
| `i16x8_relaxed_dot_i8x16_i7x16` | A relaxed dot-product instruction.  This instruction will pe... |  | — |
| `i16x8_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `i16x8_relaxed_q15mulr` | A relaxed version of `i16x8_relaxed_q15mulr` where if both l... |  | — |
| `i32x4_relaxed_dot_i8x16_i7x16_add` | Similar to |  | — |
| `i32x4_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `i32x4_relaxed_trunc_f32x4` | A relaxed version of `i32x4_trunc_sat_f32x4(a)` converts the... |  | — |
| `i32x4_relaxed_trunc_f64x2_zero` | A relaxed version of `i32x4_trunc_sat_f64x2_zero(a)` convert... |  | — |
| `i64x2_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `i8x16_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `i8x16_relaxed_swizzle` | A relaxed version of `i8x16_swizzle(a, s)` which selects lan... |  | — |
| `u16x8_relaxed_dot_i8x16_i7x16` | A relaxed dot-product instruction.  This instruction will pe... |  | — |
| `u16x8_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `u16x8_relaxed_q15mulr` | A relaxed version of `i16x8_relaxed_q15mulr` where if both l... |  | — |
| `u32x4_relaxed_dot_i8x16_i7x16_add` | Similar to |  | — |
| `u32x4_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `u32x4_relaxed_trunc_f32x4` | A relaxed version of `u32x4_trunc_sat_f32x4(a)` converts the... |  | — |
| `u32x4_relaxed_trunc_f64x2_zero` | A relaxed version of `u32x4_trunc_sat_f64x2_zero(a)` convert... |  | — |
| `u64x2_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `u8x16_relaxed_laneselect` | A relaxed version of `v128_bitselect` where this either beha... |  | — |
| `u8x16_relaxed_swizzle` | A relaxed version of `i8x16_swizzle(a, s)` which selects lan... |  | — |


