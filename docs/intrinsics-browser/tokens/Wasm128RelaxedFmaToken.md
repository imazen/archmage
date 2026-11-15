# Wasm128RelaxedFmaToken — WASM Relaxed FMA

Proof that WASM Relaxed SIMD is available AND both `f32x4.relaxed_madd` and `f64x2.relaxed_madd` fuse to single-rounding FMA on the executing engine.

**Architecture:** wasm32 | **Features:** simd128, relaxed-simd
**Total intrinsics:** 0 (0 safe, 0 unsafe, 0 stable, 0 unstable/unknown)

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

*No intrinsics mapped to this token.*

