# Arm64Sve2Token — Arm64 SVE2

**Experimental.** Proof that NEON + SVE + SVE2 is available.

**Architecture:** aarch64 | **Features:** neon, sve, sve2
**Total intrinsics:** 0 (0 safe, 0 unsafe, 0 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Arm64Sve2Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Arm64Sve2Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Arm64Sve2Token, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

*No intrinsics mapped to this token.*

