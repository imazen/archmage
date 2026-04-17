# Arm64V2Token — Arm64-v2

Proof that the Arm64-v2 feature set is available.

**Architecture:** aarch64 | **Features:** neon, crc, rdm, dotprod, fp16, aes, sha2
**Total intrinsics:** 445 (401 safe, 44 unsafe, 10 stable, 435 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Arm64V2Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Arm64V2Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Arm64V2Token, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (10 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `vsha1cq_u32` | SHA1 hash update accelerator, choose | sha1c | — |
| `vsha1h_u32` | SHA1 fixed rotate | sha1h | — |
| `vsha1mq_u32` | SHA1 hash update accelerator, majority | sha1m | — |
| `vsha1pq_u32` | SHA1 hash update accelerator, parity | sha1p | — |
| `vsha1su0q_u32` | SHA1 schedule update accelerator, first part | sha1su0 | — |
| `vsha1su1q_u32` | SHA1 schedule update accelerator, second part | sha1su1 | — |
| `vsha256h2q_u32` | SHA1 schedule update accelerator, upper part | sha256h2 | — |
| `vsha256hq_u32` | SHA1 schedule update accelerator, first part | sha256h | — |
| `vsha256su0q_u32` | SHA256 schedule update accelerator, first part | sha256su0 | — |
| `vsha256su1q_u32` | SHA256 schedule update accelerator, second part | sha256su1 | — |

### Unstable/Nightly (435 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `vabd_f16` |  |  |
| `vabdh_f16` | Floating-point absolute difference | fabd |
| `vabdq_f16` |  |  |
| `vabs_f16` |  |  |
| `vabsh_f16` |  |  |
| `vabsq_f16` |  |  |
| `vadd_f16` |  |  |
| `vaddh_f16` |  |  |
| `vaddq_f16` |  |  |
| `vbsl_f16` | Bitwise Select | vbsl |
| `vbslq_f16` | Bitwise Select | vbsl |
| `vcadd_rot270_f16` | Floating-point complex add | fcadd |
| `vcadd_rot90_f16` | Floating-point complex add | fcadd |
| `vcaddq_rot270_f16` | Floating-point complex add | fcadd |
| `vcaddq_rot90_f16` | Floating-point complex add | fcadd |
| `vcage_f16` |  |  |
| `vcageh_f16` | Floating-point absolute compare greater than or equal |  |
| `vcageq_f16` |  |  |
| `vcagt_f16` |  |  |
| `vcagth_f16` | Floating-point absolute compare greater than |  |
| `vcagtq_f16` |  |  |
| `vcale_f16` |  |  |
| `vcaleh_f16` | Floating-point absolute compare less than or equal |  |
| `vcaleq_f16` |  |  |
| `vcalt_f16` |  |  |
| `vcalth_f16` | Floating-point absolute compare less than |  |
| `vcaltq_f16` |  |  |
| `vceq_f16` |  |  |
| `vceqh_f16` | Floating-point compare equal |  |
| `vceqq_f16` |  |  |
| `vceqz_f16` | Floating-point compare bitwise equal to zero |  |
| `vceqzh_f16` | Floating-point compare bitwise equal to zero |  |
| `vceqzq_f16` | Floating-point compare bitwise equal to zero |  |
| `vcge_f16` |  |  |
| `vcgeh_f16` | Floating-point compare greater than or equal |  |
| `vcgeq_f16` |  |  |
| `vcgez_f16` |  |  |
| `vcgezh_f16` | Floating-point compare greater than or equal to zero |  |
| `vcgezq_f16` |  |  |
| `vcgt_f16` |  |  |
| `vcgth_f16` | Floating-point compare greater than |  |
| `vcgtq_f16` |  |  |
| `vcgtz_f16` |  |  |
| `vcgtzh_f16` | Floating-point compare greater than zero |  |
| `vcgtzq_f16` |  |  |
| `vcle_f16` |  |  |
| `vcleh_f16` | Floating-point compare less than or equal |  |
| `vcleq_f16` |  |  |
| `vclez_f16` |  |  |
| `vclezh_f16` | Floating-point compare less than or equal to zero |  |
| `vclezq_f16` |  |  |
| `vclt_f16` |  |  |
| `vclth_f16` | Floating-point compare less than |  |
| `vcltq_f16` |  |  |
| `vcltz_f16` |  |  |
| `vcltzh_f16` | Floating-point compare less than zero |  |
| `vcltzq_f16` |  |  |
| `vcombine_f16` | Join two smaller vectors into a single larger vector | nop |
| `vcreate_f16` |  |  |
| `vcvt_f16_f32` |  |  |
| `vcvt_f16_s16` |  |  |
| `vcvt_f16_u16` |  |  |
| `vcvt_f32_f16` |  |  |
| `vcvt_high_f16_f32` | Floating-point convert to lower precision |  |
| `vcvt_high_f32_f16` | Floating-point convert to higher precision |  |
| `vcvt_n_f16_s16` |  |  |
| `vcvt_n_f16_u16` |  |  |
| `vcvt_n_s16_f16` |  |  |
| `vcvt_n_u16_f16` |  |  |
| `vcvt_s16_f16` |  |  |
| `vcvt_u16_f16` |  |  |
| `vcvta_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  |
| `vcvta_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtah_s16_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_s32_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_s64_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_u16_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_u32_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_u64_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtaq_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  |
| `vcvtaq_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvth_f16_s16` | Fixed-point convert to floating-point |  |
| `vcvth_f16_s32` | Fixed-point convert to floating-point |  |
| `vcvth_f16_s64` | Fixed-point convert to floating-point |  |
| `vcvth_f16_u16` | Unsigned fixed-point convert to floating-point |  |
| `vcvth_f16_u32` | Unsigned fixed-point convert to floating-point |  |
| `vcvth_f16_u64` | Unsigned fixed-point convert to floating-point |  |
| `vcvth_n_f16_s16` | Fixed-point convert to floating-point |  |
| `vcvth_n_f16_s32` | Fixed-point convert to floating-point |  |
| `vcvth_n_f16_s64` | Fixed-point convert to floating-point |  |
| `vcvth_n_f16_u16` | Fixed-point convert to floating-point |  |
| `vcvth_n_f16_u32` | Fixed-point convert to floating-point |  |
| `vcvth_n_f16_u64` | Fixed-point convert to floating-point |  |
| `vcvth_n_s16_f16` | Floating-point convert to fixed-point, rounding toward zero |  |
| `vcvth_n_s32_f16` | Floating-point convert to fixed-point, rounding toward zero |  |
| `vcvth_n_s64_f16` | Floating-point convert to fixed-point, rounding toward zero |  |
| `vcvth_n_u16_f16` | Floating-point convert to fixed-point, rounding toward zero |  |
| `vcvth_n_u32_f16` | Floating-point convert to fixed-point, rounding toward zero |  |
| `vcvth_n_u64_f16` | Floating-point convert to fixed-point, rounding toward zero |  |
| `vcvth_s16_f16` | Floating-point convert to signed fixed-point |  |
| `vcvth_s32_f16` | Floating-point convert to signed fixed-point |  |
| `vcvth_s64_f16` | Floating-point convert to signed fixed-point |  |
| `vcvth_u16_f16` | Floating-point convert to unsigned fixed-point |  |
| `vcvth_u32_f16` | Floating-point convert to unsigned fixed-point |  |
| `vcvth_u64_f16` | Floating-point convert to unsigned fixed-point |  |
| `vcvtm_s16_f16` | Floating-point convert to signed integer, rounding toward mi... |  |
| `vcvtm_u16_f16` | Floating-point convert to unsigned integer, rounding toward ... |  |
| `vcvtmh_s16_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_s32_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_s64_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_u16_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_u32_f16` | Floating-point convert to unsigned integer, rounding towards... |  |
| `vcvtmh_u64_f16` | Floating-point convert to unsigned integer, rounding towards... |  |
| `vcvtmq_s16_f16` | Floating-point convert to signed integer, rounding toward mi... |  |
| `vcvtmq_u16_f16` | Floating-point convert to unsigned integer, rounding toward ... |  |
| `vcvtn_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  |
| `vcvtn_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtnh_s16_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtnh_s32_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtnh_s64_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtnh_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtnh_u32_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtnh_u64_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtnq_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  |
| `vcvtnq_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtp_s16_f16` | Floating-point convert to signed integer, rounding to plus i... |  |
| `vcvtp_u16_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtph_s16_f16` | Floating-point convert to integer, rounding to plus infinity |  |
| `vcvtph_s32_f16` | Floating-point convert to integer, rounding to plus infinity |  |
| `vcvtph_s64_f16` | Floating-point convert to integer, rounding to plus infinity |  |
| `vcvtph_u16_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtph_u32_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtph_u64_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtpq_s16_f16` | Floating-point convert to signed integer, rounding to plus i... |  |
| `vcvtpq_u16_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtq_f16_s16` |  |  |
| `vcvtq_f16_u16` |  |  |
| `vcvtq_n_f16_s16` |  |  |
| `vcvtq_n_f16_u16` |  |  |
| `vcvtq_n_s16_f16` |  |  |
| `vcvtq_n_u16_f16` |  |  |
| `vcvtq_s16_f16` |  |  |
| `vcvtq_u16_f16` |  |  |
| `vdiv_f16` | Divide | fdiv |
| `vdivh_f16` | Divide | nop |
| `vdivq_f16` | Divide | fdiv |
| `vdot_lane_s32` | Dot product arithmetic (indexed) | vsdot |
| `vdot_lane_u32` | Dot product arithmetic (indexed) | vudot |
| `vdot_laneq_s32` | Dot product arithmetic (indexed) | sdot |
| `vdot_laneq_u32` | Dot product arithmetic (indexed) | udot |
| `vdot_s32` | Dot product arithmetic (vector) | vsdot |
| `vdot_u32` | Dot product arithmetic (vector) | vudot |
| `vdotq_lane_s32` | Dot product arithmetic (indexed) | vsdot |
| `vdotq_lane_u32` | Dot product arithmetic (indexed) | vudot |
| `vdotq_laneq_s32` | Dot product arithmetic (indexed) | sdot |
| `vdotq_laneq_u32` | Dot product arithmetic (indexed) | udot |
| `vdotq_s32` | Dot product arithmetic (vector) | vsdot |
| `vdotq_u32` | Dot product arithmetic (vector) | vudot |
| `vdup_lane_f16` |  |  |
| `vdup_laneq_f16` |  |  |
| `vdup_n_f16` |  |  |
| `vduph_lane_f16` | Set all vector lanes to the same value |  |
| `vduph_laneq_f16` | Extract an element from a vector |  |
| `vdupq_lane_f16` |  |  |
| `vdupq_laneq_f16` |  |  |
| `vdupq_n_f16` |  |  |
| `vext_f16` |  |  |
| `vextq_f16` |  |  |
| `vfma_f16` |  |  |
| `vfma_lane_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfma_laneq_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfma_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmla |
| `vfmah_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmah_lane_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmah_laneq_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmaq_f16` |  |  |
| `vfmaq_lane_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmaq_laneq_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmaq_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmla |
| `vfmlal_high_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal2 |
| `vfmlal_lane_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlal_lane_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlal_laneq_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlal_laneq_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlal_low_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal |
| `vfmlalq_high_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal2 |
| `vfmlalq_lane_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlalq_lane_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlalq_laneq_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlalq_laneq_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  |
| `vfmlalq_low_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal |
| `vfmlsl_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl2 |
| `vfmlsl_lane_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlsl_lane_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlsl_laneq_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlsl_laneq_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlsl_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl |
| `vfmlslq_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl2 |
| `vfmlslq_lane_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlslq_lane_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlslq_laneq_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlslq_laneq_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  |
| `vfmlslq_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl |
| `vfms_f16` |  |  |
| `vfms_lane_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfms_laneq_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfms_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmls |
| `vfmsh_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsh_lane_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsh_laneq_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsq_f16` |  |  |
| `vfmsq_lane_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsq_laneq_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsq_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmls |
| `vget_high_f16` | Duplicate vector element to vector | nop |
| `vget_lane_f16` | Duplicate vector element to scalar | nop |
| `vget_low_f16` | Duplicate vector element to vector | nop |
| `vgetq_lane_f16` | Duplicate vector element to scalar | nop |
| `vld1_dup_f16` |  |  |
| `vld1_f16_x2` |  |  |
| `vld1_f16_x3` |  |  |
| `vld1_f16_x4` |  |  |
| `vld1_lane_f16` |  |  |
| `vld1q_dup_f16` |  |  |
| `vld1q_f16_x2` |  |  |
| `vld1q_f16_x3` |  |  |
| `vld1q_f16_x4` |  |  |
| `vld1q_lane_f16` |  |  |
| `vld2_dup_f16` | Load single 2-element structure and replicate to all lanes o... | vld2 |
| `vld2_f16` | Load single 2-element structure and replicate to all lanes o... |  |
| `vld2q_dup_f16` | Load single 2-element structure and replicate to all lanes o... | vld2 |
| `vld2q_f16` | Load single 2-element structure and replicate to all lanes o... |  |
| `vld3_dup_f16` | Load single 3-element structure and replicate to all lanes o... |  |
| `vld3_f16` | Load single 3-element structure and replicate to all lanes o... |  |
| `vld3q_dup_f16` | Load single 3-element structure and replicate to all lanes o... |  |
| `vld3q_f16` | Load single 3-element structure and replicate to all lanes o... |  |
| `vld4_dup_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vld4_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vld4q_dup_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vld4q_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vmax_f16` |  |  |
| `vmaxh_f16` | Maximum (vector) | fmax |
| `vmaxnm_f16` |  |  |
| `vmaxnmh_f16` | Floating-point Maximum Number | fmaxnm |
| `vmaxnmq_f16` |  |  |
| `vmaxnmv_f16` | Floating-point maximum number across vector | fmaxnmv |
| `vmaxnmvq_f16` | Floating-point maximum number across vector | fmaxnmv |
| `vmaxq_f16` |  |  |
| `vmaxv_f16` | Floating-point maximum number across vector | fmaxv |
| `vmaxvq_f16` | Floating-point maximum number across vector | fmaxv |
| `vmin_f16` |  |  |
| `vminh_f16` | Minimum (vector) | fmin |
| `vminnm_f16` |  |  |
| `vminnmh_f16` | Floating-point Minimum Number | fminnm |
| `vminnmq_f16` |  |  |
| `vminnmv_f16` | Floating-point minimum number across vector | fminnmv |
| `vminnmvq_f16` | Floating-point minimum number across vector | fminnmv |
| `vminq_f16` |  |  |
| `vminv_f16` | Floating-point minimum number across vector | fminv |
| `vminvq_f16` | Floating-point minimum number across vector | fminv |
| `vmov_n_f16` |  |  |
| `vmovq_n_f16` |  |  |
| `vmul_f16` |  |  |
| `vmul_lane_f16` |  |  |
| `vmul_laneq_f16` | Floating-point multiply |  |
| `vmul_n_f16` |  |  |
| `vmulh_f16` | Add | nop |
| `vmulh_lane_f16` | Floating-point multiply |  |
| `vmulh_laneq_f16` | Floating-point multiply |  |
| `vmulq_f16` |  |  |
| `vmulq_lane_f16` |  |  |
| `vmulq_laneq_f16` | Floating-point multiply |  |
| `vmulq_n_f16` |  |  |
| `vmulx_f16` | Floating-point multiply extended | fmulx |
| `vmulx_lane_f16` | Floating-point multiply extended |  |
| `vmulx_laneq_f16` | Floating-point multiply extended |  |
| `vmulx_n_f16` | Vector multiply by scalar |  |
| `vmulxh_f16` | Floating-point multiply extended | fmulx |
| `vmulxh_lane_f16` | Floating-point multiply extended |  |
| `vmulxh_laneq_f16` | Floating-point multiply extended |  |
| `vmulxq_f16` | Floating-point multiply extended | fmulx |
| `vmulxq_lane_f16` | Floating-point multiply extended |  |
| `vmulxq_laneq_f16` | Floating-point multiply extended |  |
| `vmulxq_n_f16` | Vector multiply by scalar |  |
| `vneg_f16` |  |  |
| `vnegh_f16` | Negate | fneg |
| `vnegq_f16` |  |  |
| `vpadd_f16` |  |  |
| `vpaddq_f16` | Floating-point add pairwise | faddp |
| `vpmax_f16` | Floating-point add pairwise | fmaxp |
| `vpmaxnm_f16` | Floating-point add pairwise | fmaxnmp |
| `vpmaxnmq_f16` | Floating-point add pairwise | fmaxnmp |
| `vpmaxq_f16` | Floating-point add pairwise | fmaxp |
| `vpmin_f16` | Floating-point add pairwise | fminp |
| `vpminnm_f16` | Floating-point add pairwise | fminnmp |
| `vpminnmq_f16` | Floating-point add pairwise | fminnmp |
| `vpminq_f16` | Floating-point add pairwise | fminp |
| `vrecpe_f16` |  |  |
| `vrecpeh_f16` | Reciprocal estimate |  |
| `vrecpeq_f16` |  |  |
| `vrecps_f16` |  |  |
| `vrecpsh_f16` | Floating-point reciprocal step |  |
| `vrecpsq_f16` |  |  |
| `vrecpxh_f16` | Floating-point reciprocal exponent |  |
| `vreinterpret_f16_f32` |  |  |
| `vreinterpret_f16_f64` | Vector reinterpret cast operation | nop |
| `vreinterpret_f16_p16` |  |  |
| `vreinterpret_f16_p64` |  |  |
| `vreinterpret_f16_p8` |  |  |
| `vreinterpret_f16_s16` |  |  |
| `vreinterpret_f16_s32` |  |  |
| `vreinterpret_f16_s64` |  |  |
| `vreinterpret_f16_s8` |  |  |
| `vreinterpret_f16_u16` |  |  |
| `vreinterpret_f16_u32` |  |  |
| `vreinterpret_f16_u64` |  |  |
| `vreinterpret_f16_u8` |  |  |
| `vreinterpret_f32_f16` |  |  |
| `vreinterpret_f64_f16` | Vector reinterpret cast operation | nop |
| `vreinterpret_p16_f16` |  |  |
| `vreinterpret_p64_f16` |  |  |
| `vreinterpret_p8_f16` |  |  |
| `vreinterpret_s16_f16` |  |  |
| `vreinterpret_s32_f16` |  |  |
| `vreinterpret_s64_f16` |  |  |
| `vreinterpret_s8_f16` |  |  |
| `vreinterpret_u16_f16` |  |  |
| `vreinterpret_u32_f16` |  |  |
| `vreinterpret_u64_f16` |  |  |
| `vreinterpret_u8_f16` |  |  |
| `vreinterpretq_f16_f32` |  |  |
| `vreinterpretq_f16_f64` | Vector reinterpret cast operation | nop |
| `vreinterpretq_f16_p128` |  |  |
| `vreinterpretq_f16_p16` |  |  |
| `vreinterpretq_f16_p64` |  |  |
| `vreinterpretq_f16_p8` |  |  |
| `vreinterpretq_f16_s16` |  |  |
| `vreinterpretq_f16_s32` |  |  |
| `vreinterpretq_f16_s64` |  |  |
| `vreinterpretq_f16_s8` |  |  |
| `vreinterpretq_f16_u16` |  |  |
| `vreinterpretq_f16_u32` |  |  |
| `vreinterpretq_f16_u64` |  |  |
| `vreinterpretq_f16_u8` |  |  |
| `vreinterpretq_f32_f16` |  |  |
| `vreinterpretq_f64_f16` | Vector reinterpret cast operation | nop |
| `vreinterpretq_p128_f16` |  |  |
| `vreinterpretq_p16_f16` |  |  |
| `vreinterpretq_p64_f16` |  |  |
| `vreinterpretq_p8_f16` |  |  |
| `vreinterpretq_s16_f16` |  |  |
| `vreinterpretq_s32_f16` |  |  |
| `vreinterpretq_s64_f16` |  |  |
| `vreinterpretq_s8_f16` |  |  |
| `vreinterpretq_u16_f16` |  |  |
| `vreinterpretq_u32_f16` |  |  |
| `vreinterpretq_u64_f16` |  |  |
| `vreinterpretq_u8_f16` |  |  |
| `vrev64_f16` |  |  |
| `vrev64q_f16` |  |  |
| `vrnd_f16` | Floating-point round to integral, toward zero | frintz |
| `vrnda_f16` | Floating-point round to integral, to nearest with ties to aw... | frinta |
| `vrndah_f16` | Floating-point round to integral, to nearest with ties to aw... | frinta |
| `vrndaq_f16` | Floating-point round to integral, to nearest with ties to aw... | frinta |
| `vrndh_f16` | Floating-point round to integral, to nearest with ties to aw... | frintz |
| `vrndi_f16` | Floating-point round to integral, using current rounding mod... | frinti |
| `vrndih_f16` | Floating-point round to integral, using current rounding mod... | frinti |
| `vrndiq_f16` | Floating-point round to integral, using current rounding mod... | frinti |
| `vrndm_f16` | Floating-point round to integral, toward minus infinity | frintm |
| `vrndmh_f16` | Floating-point round to integral, toward minus infinity | frintm |
| `vrndmq_f16` | Floating-point round to integral, toward minus infinity | frintm |
| `vrndn_f16` |  |  |
| `vrndnh_f16` | Floating-point round to integral, toward minus infinity | frintn |
| `vrndnq_f16` |  |  |
| `vrndp_f16` | Floating-point round to integral, toward plus infinity | frintp |
| `vrndph_f16` | Floating-point round to integral, toward plus infinity | frintp |
| `vrndpq_f16` | Floating-point round to integral, toward plus infinity | frintp |
| `vrndq_f16` | Floating-point round to integral, toward zero | frintz |
| `vrndx_f16` | Floating-point round to integral exact, using current roundi... | frintx |
| `vrndxh_f16` | Floating-point round to integral, using current rounding mod... | frintx |
| `vrndxq_f16` | Floating-point round to integral exact, using current roundi... | frintx |
| `vrsqrte_f16` | Reciprocal square-root estimate | vrsqrte |
| `vrsqrteh_f16` | Reciprocal square-root estimate |  |
| `vrsqrteq_f16` | Reciprocal square-root estimate | vrsqrte |
| `vrsqrts_f16` | Floating-point reciprocal square root step | vrsqrts |
| `vrsqrtsh_f16` | Floating-point reciprocal square root step | frsqrts |
| `vrsqrtsq_f16` | Floating-point reciprocal square root step | vrsqrts |
| `vset_lane_f16` |  |  |
| `vsetq_lane_f16` |  |  |
| `vsqrt_f16` | Calculates the square root of each lane |  |
| `vsqrth_f16` | Floating-point round to integral, using current rounding mod... | fsqrt |
| `vsqrtq_f16` | Calculates the square root of each lane |  |
| `vst1_f16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1_f16_x2` | Store multiple single-element structures to one, two, three,... |  |
| `vst1_f16_x3` | Store multiple single-element structures to one, two, three,... |  |
| `vst1_f16_x4` | Store multiple single-element structures to one, two, three,... | vst1 |
| `vst1_lane_f16` |  |  |
| `vst1q_f16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1q_f16_x2` | Store multiple single-element structures to one, two, three,... |  |
| `vst1q_f16_x3` | Store multiple single-element structures to one, two, three,... |  |
| `vst1q_f16_x4` | Store multiple single-element structures to one, two, three,... | vst1 |
| `vst1q_lane_f16` |  |  |
| `vst2_f16` | Store multiple 2-element structures from two registers | st2 |
| `vst2_lane_f16` | Store multiple 2-element structures from two registers |  |
| `vst2q_f16` | Store multiple 2-element structures from two registers | st2 |
| `vst2q_lane_f16` | Store multiple 2-element structures from two registers |  |
| `vst3_f16` | Store multiple 3-element structures from three registers | vst3 |
| `vst3_lane_f16` | Store multiple 3-element structures from three registers |  |
| `vst3q_f16` | Store multiple 3-element structures from three registers | vst3 |
| `vst3q_lane_f16` | Store multiple 3-element structures from three registers |  |
| `vst4_f16` | Store multiple 4-element structures from four registers | vst4 |
| `vst4_lane_f16` | Store multiple 4-element structures from four registers |  |
| `vst4q_f16` | Store multiple 4-element structures from four registers | vst4 |
| `vst4q_lane_f16` | Store multiple 4-element structures from four registers |  |
| `vsub_f16` |  |  |
| `vsubh_f16` | Subtract | nop |
| `vsubq_f16` |  |  |
| `vtrn1_f16` | Transpose vectors | trn1 |
| `vtrn1q_f16` | Transpose vectors | trn1 |
| `vtrn2_f16` | Transpose vectors | trn2 |
| `vtrn2q_f16` | Transpose vectors | trn2 |
| `vtrn_f16` |  |  |
| `vtrnq_f16` |  |  |
| `vuzp1_f16` | Unzip vectors | uzp1 |
| `vuzp1q_f16` | Unzip vectors | uzp1 |
| `vuzp2_f16` | Unzip vectors | uzp2 |
| `vuzp2q_f16` | Unzip vectors | uzp2 |
| `vuzp_f16` |  |  |
| `vuzpq_f16` |  |  |
| `vzip1_f16` | Zip vectors | zip1 |
| `vzip1q_f16` | Zip vectors | zip1 |
| `vzip2_f16` | Zip vectors | zip2 |
| `vzip2q_f16` | Zip vectors | zip2 |
| `vzip_f16` |  |  |
| `vzipq_f16` |  |  |


