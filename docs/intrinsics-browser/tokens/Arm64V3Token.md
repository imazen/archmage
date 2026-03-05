# Arm64V3Token — Arm64-v3

Proof that the full modern ARM SIMD feature set is available (Arm64-v3).

**Architecture:** aarch64 | **Features:** neon, crc, rdm, dotprod, fp16, aes, sha2, fhm, fcma, sha3, i8mm, bf16
**Total intrinsics:** 71 (71 safe, 0 unsafe, 0 stable, 71 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Arm64V3Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Arm64V3Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Arm64V3Token, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Unstable/Nightly (71 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `vcadd_rot270_f32` | Floating-point complex add | fcadd |
| `vcadd_rot90_f32` | Floating-point complex add | fcadd |
| `vcaddq_rot270_f32` | Floating-point complex add | fcadd |
| `vcaddq_rot270_f64` | Floating-point complex add | fcadd |
| `vcaddq_rot90_f32` | Floating-point complex add | fcadd |
| `vcaddq_rot90_f64` | Floating-point complex add | fcadd |
| `vcmla_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot180_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot180_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot180_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot180_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot180_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot180_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot270_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot270_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot270_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot270_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot270_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot270_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot90_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot90_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot90_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot90_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot90_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmla_rot90_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_f64` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_f64` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot180_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_f64` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot270_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_f64` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_lane_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_lane_f32` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_laneq_f16` | Floating-point complex multiply accumulate | fcmla |
| `vcmlaq_rot90_laneq_f32` | Floating-point complex multiply accumulate | fcmla |
| `vmmlaq_s32` | 8-bit integer matrix multiply-accumulate | nop |
| `vmmlaq_u32` | 8-bit integer matrix multiply-accumulate | nop |
| `vsudot_lane_s32` | Dot product index form with signed and unsigned integers | vsudot |
| `vsudot_laneq_s32` | Dot product index form with signed and unsigned integers | sudot |
| `vsudotq_lane_s32` | Dot product index form with signed and unsigned integers | vsudot |
| `vsudotq_laneq_s32` | Dot product index form with signed and unsigned integers | sudot |
| `vusdot_lane_s32` | Dot product index form with unsigned and signed integers | vusdot |
| `vusdot_laneq_s32` | Dot product index form with unsigned and signed integers | usdot |
| `vusdot_s32` | Dot product vector form with unsigned and signed integers | vusdot |
| `vusdotq_lane_s32` | Dot product index form with unsigned and signed integers | vusdot |
| `vusdotq_laneq_s32` | Dot product index form with unsigned and signed integers | usdot |
| `vusdotq_s32` | Dot product vector form with unsigned and signed integers | vusdot |
| `vusmmlaq_s32` | Unsigned and signed 8-bit integer matrix multiply-accumulate | nop |


