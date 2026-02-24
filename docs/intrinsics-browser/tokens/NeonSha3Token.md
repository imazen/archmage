# NeonSha3Token — NEON+SHA3

Proof that NEON + SHA3 is available.

**Architecture:** aarch64 | **Features:** neon, sha3
**Total intrinsics:** 22 (22 safe, 0 unsafe, 22 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = NeonSha3Token::summon() {
    process(token, &mut data);
}

#[arcane]  // Entry point only
fn process(token: NeonSha3Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite]  // All inner helpers
fn process_chunk(_: NeonSha3Token, chunk: &mut [f32; 4]) {
    let v = safe_unaligned_simd::aarch64::vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    safe_unaligned_simd::aarch64::vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (22 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `vbcaxq_s16` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_s32` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_s64` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_s8` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_u16` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_u32` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_u64` | Bit clear and exclusive OR | bcax | — |
| `vbcaxq_u8` | Bit clear and exclusive OR | bcax | — |
| `veor3q_s16` | Three-way exclusive OR | eor3 | — |
| `veor3q_s32` | Three-way exclusive OR | eor3 | — |
| `veor3q_s64` | Three-way exclusive OR | eor3 | — |
| `veor3q_s8` | Three-way exclusive OR | eor3 | — |
| `veor3q_u16` | Three-way exclusive OR | eor3 | — |
| `veor3q_u32` | Three-way exclusive OR | eor3 | — |
| `veor3q_u64` | Three-way exclusive OR | eor3 | — |
| `veor3q_u8` | Three-way exclusive OR | eor3 | — |
| `vrax1q_u64` | Rotate and exclusive OR | rax1 | — |
| `vsha512h2q_u64` | SHA512 hash update part 2 | sha512h2 | — |
| `vsha512hq_u64` | SHA512 hash update part 1 | sha512h | — |
| `vsha512su0q_u64` | SHA512 schedule update 0 | sha512su0 | — |
| `vsha512su1q_u64` | SHA512 schedule update 1 | sha512su1 | — |
| `vxarq_u64` | Exclusive OR and rotate | xar | — |


