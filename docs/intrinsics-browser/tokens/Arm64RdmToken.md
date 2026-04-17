# Arm64RdmToken — NEON+RDM

Proof that NEON + FEAT_RDM (Rounding Doubling Multiply, ARMv8.1) is available.

**Architecture:** aarch64 | **Features:** neon, rdm
**Total intrinsics:** 36 (36 safe, 0 unsafe, 36 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Arm64RdmToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Arm64RdmToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Arm64RdmToken, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (36 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `vqrdmlah_lane_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlah_lane_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlah_laneq_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlah_laneq_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlah_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlah_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahh_lane_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahh_laneq_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahh_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahq_lane_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahq_lane_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahq_laneq_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahq_laneq_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahq_s16` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahq_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahs_lane_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahs_laneq_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlahs_s32` | Signed saturating rounding doubling multiply accumulate retu... | sqrdmlah | — |
| `vqrdmlsh_lane_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlsh_lane_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlsh_laneq_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlsh_laneq_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlsh_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlsh_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshh_lane_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshh_laneq_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshh_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshq_lane_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshq_lane_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshq_laneq_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshq_laneq_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshq_s16` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshq_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshs_lane_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshs_laneq_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |
| `vqrdmlshs_s32` | Signed saturating rounding doubling multiply subtract return... | sqrdmlsh | — |


