# Arm64V2Token — Arm64-v2

Proof that the Arm64-v2 feature set is available.

**Architecture:** aarch64 | **Features:** neon, crc, rdm, dotprod, fp16, aes, sha2
**Total intrinsics:** 375 (365 safe, 10 unsafe, 232 stable, 143 unstable/unknown)

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

### Stable, Safe (232 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `vabd_f16` |  |  | — |
| `vabdq_f16` |  |  | — |
| `vabs_f16` |  |  | — |
| `vabsq_f16` |  |  | — |
| `vadd_f16` |  |  | — |
| `vaddq_f16` |  |  | — |
| `vbsl_f16` | Bitwise Select | vbsl | — |
| `vbslq_f16` | Bitwise Select | vbsl | — |
| `vcage_f16` |  |  | — |
| `vcageq_f16` |  |  | — |
| `vcagt_f16` |  |  | — |
| `vcagtq_f16` |  |  | — |
| `vcale_f16` |  |  | — |
| `vcaleq_f16` |  |  | — |
| `vcalt_f16` |  |  | — |
| `vcaltq_f16` |  |  | — |
| `vceq_f16` |  |  | — |
| `vceqq_f16` |  |  | — |
| `vceqz_f16` | Floating-point compare bitwise equal to zero |  | — |
| `vceqzq_f16` | Floating-point compare bitwise equal to zero |  | — |
| `vcge_f16` |  |  | — |
| `vcgeq_f16` |  |  | — |
| `vcgez_f16` |  |  | — |
| `vcgezq_f16` |  |  | — |
| `vcgt_f16` |  |  | — |
| `vcgtq_f16` |  |  | — |
| `vcgtz_f16` |  |  | — |
| `vcgtzq_f16` |  |  | — |
| `vcle_f16` |  |  | — |
| `vcleq_f16` |  |  | — |
| `vclez_f16` |  |  | — |
| `vclezq_f16` |  |  | — |
| `vclt_f16` |  |  | — |
| `vcltq_f16` |  |  | — |
| `vcltz_f16` |  |  | — |
| `vcltzq_f16` |  |  | — |
| `vcvt_f16_s16` |  |  | — |
| `vcvt_f16_u16` |  |  | — |
| `vcvt_n_f16_s16` |  |  | — |
| `vcvt_n_f16_u16` |  |  | — |
| `vcvt_n_s16_f16` |  |  | — |
| `vcvt_n_u16_f16` |  |  | — |
| `vcvt_s16_f16` |  |  | — |
| `vcvt_u16_f16` |  |  | — |
| `vcvta_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  | — |
| `vcvta_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  | — |
| `vcvtaq_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  | — |
| `vcvtaq_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  | — |
| `vcvtm_s16_f16` | Floating-point convert to signed integer, rounding toward mi... |  | — |
| `vcvtm_u16_f16` | Floating-point convert to unsigned integer, rounding toward ... |  | — |
| `vcvtmq_s16_f16` | Floating-point convert to signed integer, rounding toward mi... |  | — |
| `vcvtmq_u16_f16` | Floating-point convert to unsigned integer, rounding toward ... |  | — |
| `vcvtn_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  | — |
| `vcvtn_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  | — |
| `vcvtnq_s16_f16` | Floating-point convert to signed integer, rounding to neares... |  | — |
| `vcvtnq_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  | — |
| `vcvtp_s16_f16` | Floating-point convert to signed integer, rounding to plus i... |  | — |
| `vcvtp_u16_f16` | Floating-point convert to unsigned integer, rounding to plus... |  | — |
| `vcvtpq_s16_f16` | Floating-point convert to signed integer, rounding to plus i... |  | — |
| `vcvtpq_u16_f16` | Floating-point convert to unsigned integer, rounding to plus... |  | — |
| `vcvtq_f16_s16` |  |  | — |
| `vcvtq_f16_u16` |  |  | — |
| `vcvtq_n_f16_s16` |  |  | — |
| `vcvtq_n_f16_u16` |  |  | — |
| `vcvtq_n_s16_f16` |  |  | — |
| `vcvtq_n_u16_f16` |  |  | — |
| `vcvtq_s16_f16` |  |  | — |
| `vcvtq_u16_f16` |  |  | — |
| `vdiv_f16` | Divide | fdiv | — |
| `vdivq_f16` | Divide | fdiv | — |
| `vext_f16` |  |  | — |
| `vextq_f16` |  |  | — |
| `vfma_f16` |  |  | — |
| `vfma_lane_f16` | Floating-point fused multiply-add to accumulator |  | — |
| `vfma_laneq_f16` | Floating-point fused multiply-add to accumulator |  | — |
| `vfmaq_f16` |  |  | — |
| `vfmaq_lane_f16` | Floating-point fused multiply-add to accumulator |  | — |
| `vfmaq_laneq_f16` | Floating-point fused multiply-add to accumulator |  | — |
| `vfmlal_high_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal2 | — |
| `vfmlal_lane_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlal_lane_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlal_laneq_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlal_laneq_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlal_low_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal | — |
| `vfmlalq_high_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal2 | — |
| `vfmlalq_lane_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlalq_lane_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlalq_laneq_high_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlalq_laneq_low_f16` | Floating-point fused Multiply-Add Long to accumulator (by el... |  | — |
| `vfmlalq_low_f16` | Floating-point fused Multiply-Add Long to accumulator (vecto... | fmlal | — |
| `vfmlsl_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl2 | — |
| `vfmlsl_lane_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlsl_lane_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlsl_laneq_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlsl_laneq_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlsl_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl | — |
| `vfmlslq_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl2 | — |
| `vfmlslq_lane_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlslq_lane_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlslq_laneq_high_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlslq_laneq_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... |  | — |
| `vfmlslq_low_f16` | Floating-point fused Multiply-Subtract Long from accumulator... | fmlsl | — |
| `vfms_f16` |  |  | — |
| `vfms_lane_f16` | Floating-point fused multiply-subtract from accumulator |  | — |
| `vfms_laneq_f16` | Floating-point fused multiply-subtract from accumulator |  | — |
| `vfmsq_f16` |  |  | — |
| `vfmsq_lane_f16` | Floating-point fused multiply-subtract from accumulator |  | — |
| `vfmsq_laneq_f16` | Floating-point fused multiply-subtract from accumulator |  | — |
| `vmax_f16` |  |  | — |
| `vmaxnm_f16` |  |  | — |
| `vmaxnmq_f16` |  |  | — |
| `vmaxq_f16` |  |  | — |
| `vmin_f16` |  |  | — |
| `vminnm_f16` |  |  | — |
| `vminnmq_f16` |  |  | — |
| `vminq_f16` |  |  | — |
| `vmul_f16` |  |  | — |
| `vmul_lane_f16` |  |  | — |
| `vmul_laneq_f16` | Floating-point multiply |  | — |
| `vmulq_f16` |  |  | — |
| `vmulq_lane_f16` |  |  | — |
| `vmulq_laneq_f16` | Floating-point multiply |  | — |
| `vmulx_f16` | Floating-point multiply extended | fmulx | — |
| `vmulx_lane_f16` | Floating-point multiply extended |  | — |
| `vmulx_laneq_f16` | Floating-point multiply extended |  | — |
| `vmulxq_f16` | Floating-point multiply extended | fmulx | — |
| `vmulxq_lane_f16` | Floating-point multiply extended |  | — |
| `vmulxq_laneq_f16` | Floating-point multiply extended |  | — |
| `vneg_f16` |  |  | — |
| `vnegq_f16` |  |  | — |
| `vpadd_f16` |  |  | — |
| `vpaddq_f16` | Floating-point add pairwise | faddp | — |
| `vpmax_f16` | Floating-point add pairwise | fmaxp | — |
| `vpmaxnm_f16` | Floating-point add pairwise | fmaxnmp | — |
| `vpmaxnmq_f16` | Floating-point add pairwise | fmaxnmp | — |
| `vpmaxq_f16` | Floating-point add pairwise | fmaxp | — |
| `vpmin_f16` | Floating-point add pairwise | fminp | — |
| `vpminnm_f16` | Floating-point add pairwise | fminnmp | — |
| `vpminnmq_f16` | Floating-point add pairwise | fminnmp | — |
| `vpminq_f16` | Floating-point add pairwise | fminp | — |
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
| `vrecpe_f16` |  |  | — |
| `vrecpeq_f16` |  |  | — |
| `vrecps_f16` |  |  | — |
| `vrecpsq_f16` |  |  | — |
| `vrev64_f16` |  |  | — |
| `vrev64q_f16` |  |  | — |
| `vrnd_f16` | Floating-point round to integral, toward zero | frintz | — |
| `vrnda_f16` | Floating-point round to integral, to nearest with ties to aw... | frinta | — |
| `vrndaq_f16` | Floating-point round to integral, to nearest with ties to aw... | frinta | — |
| `vrndi_f16` | Floating-point round to integral, using current rounding mod... | frinti | — |
| `vrndiq_f16` | Floating-point round to integral, using current rounding mod... | frinti | — |
| `vrndm_f16` | Floating-point round to integral, toward minus infinity | frintm | — |
| `vrndmq_f16` | Floating-point round to integral, toward minus infinity | frintm | — |
| `vrndn_f16` |  |  | — |
| `vrndnq_f16` |  |  | — |
| `vrndp_f16` | Floating-point round to integral, toward plus infinity | frintp | — |
| `vrndpq_f16` | Floating-point round to integral, toward plus infinity | frintp | — |
| `vrndq_f16` | Floating-point round to integral, toward zero | frintz | — |
| `vrndx_f16` | Floating-point round to integral exact, using current roundi... | frintx | — |
| `vrndxq_f16` | Floating-point round to integral exact, using current roundi... | frintx | — |
| `vrsqrte_f16` | Reciprocal square-root estimate | vrsqrte | — |
| `vrsqrteq_f16` | Reciprocal square-root estimate | vrsqrte | — |
| `vrsqrts_f16` | Floating-point reciprocal square root step | vrsqrts | — |
| `vrsqrtsq_f16` | Floating-point reciprocal square root step | vrsqrts | — |
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
| `vsqrt_f16` | Calculates the square root of each lane |  | — |
| `vsqrtq_f16` | Calculates the square root of each lane |  | — |
| `vsub_f16` |  |  | — |
| `vsubq_f16` |  |  | — |
| `vtrn1_f16` | Transpose vectors | trn1 | — |
| `vtrn1q_f16` | Transpose vectors | trn1 | — |
| `vtrn2_f16` | Transpose vectors | trn2 | — |
| `vtrn2q_f16` | Transpose vectors | trn2 | — |
| `vtrn_f16` |  |  | — |
| `vtrnq_f16` |  |  | — |
| `vuzp1_f16` | Unzip vectors | uzp1 | — |
| `vuzp1q_f16` | Unzip vectors | uzp1 | — |
| `vuzp2_f16` | Unzip vectors | uzp2 | — |
| `vuzp2q_f16` | Unzip vectors | uzp2 | — |
| `vuzp_f16` |  |  | — |
| `vuzpq_f16` |  |  | — |
| `vzip1_f16` | Zip vectors | zip1 | — |
| `vzip1q_f16` | Zip vectors | zip1 | — |
| `vzip2_f16` | Zip vectors | zip2 | — |
| `vzip2q_f16` | Zip vectors | zip2 | — |
| `vzip_f16` |  |  | — |
| `vzipq_f16` |  |  | — |

### Unstable/Nightly (143 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `vabdh_f16` | Floating-point absolute difference | fabd |
| `vabsh_f16` |  |  |
| `vaddh_f16` |  |  |
| `vcadd_rot270_f16` | Floating-point complex add | fcadd |
| `vcadd_rot90_f16` | Floating-point complex add | fcadd |
| `vcaddq_rot270_f16` | Floating-point complex add | fcadd |
| `vcaddq_rot90_f16` | Floating-point complex add | fcadd |
| `vcageh_f16` | Floating-point absolute compare greater than or equal |  |
| `vcagth_f16` | Floating-point absolute compare greater than |  |
| `vcaleh_f16` | Floating-point absolute compare less than or equal |  |
| `vcalth_f16` | Floating-point absolute compare less than |  |
| `vceqh_f16` | Floating-point compare equal |  |
| `vceqzh_f16` | Floating-point compare bitwise equal to zero |  |
| `vcgeh_f16` | Floating-point compare greater than or equal |  |
| `vcgezh_f16` | Floating-point compare greater than or equal to zero |  |
| `vcgth_f16` | Floating-point compare greater than |  |
| `vcgtzh_f16` | Floating-point compare greater than zero |  |
| `vcleh_f16` | Floating-point compare less than or equal |  |
| `vclezh_f16` | Floating-point compare less than or equal to zero |  |
| `vclth_f16` | Floating-point compare less than |  |
| `vcltzh_f16` | Floating-point compare less than zero |  |
| `vcvtah_s16_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_s32_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_s64_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_u16_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_u32_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtah_u64_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
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
| `vcvtmh_s16_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_s32_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_s64_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_u16_f16` | Floating-point convert to integer, rounding towards minus in... |  |
| `vcvtmh_u32_f16` | Floating-point convert to unsigned integer, rounding towards... |  |
| `vcvtmh_u64_f16` | Floating-point convert to unsigned integer, rounding towards... |  |
| `vcvtnh_s16_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtnh_s32_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtnh_s64_f16` | Floating-point convert to integer, rounding to nearest with ... |  |
| `vcvtnh_u16_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtnh_u32_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtnh_u64_f16` | Floating-point convert to unsigned integer, rounding to near... |  |
| `vcvtph_s16_f16` | Floating-point convert to integer, rounding to plus infinity |  |
| `vcvtph_s32_f16` | Floating-point convert to integer, rounding to plus infinity |  |
| `vcvtph_s64_f16` | Floating-point convert to integer, rounding to plus infinity |  |
| `vcvtph_u16_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtph_u32_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vcvtph_u64_f16` | Floating-point convert to unsigned integer, rounding to plus... |  |
| `vdivh_f16` | Divide | fdiv |
| `vdot_lane_s32` | Dot product arithmetic (indexed) | vsdot |
| `vdot_lane_u32` | Dot product arithmetic (indexed) | vudot |
| `vdot_laneq_s32` | Dot product arithmetic (indexed) | vsdot |
| `vdot_laneq_u32` | Dot product arithmetic (indexed) | vudot |
| `vdot_s32` | Dot product arithmetic (vector) | vsdot |
| `vdot_u32` | Dot product arithmetic (vector) | vudot |
| `vdotq_lane_s32` | Dot product arithmetic (indexed) | vsdot |
| `vdotq_lane_u32` | Dot product arithmetic (indexed) | vudot |
| `vdotq_laneq_s32` | Dot product arithmetic (indexed) | vsdot |
| `vdotq_laneq_u32` | Dot product arithmetic (indexed) | vudot |
| `vdotq_s32` | Dot product arithmetic (vector) | vsdot |
| `vdotq_u32` | Dot product arithmetic (vector) | vudot |
| `vduph_lane_f16` | Set all vector lanes to the same value |  |
| `vduph_laneq_f16` | Extract an element from a vector |  |
| `vfma_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmla |
| `vfmah_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmah_lane_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmah_laneq_f16` | Floating-point fused multiply-add to accumulator |  |
| `vfmaq_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmla |
| `vfms_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmls |
| `vfmsh_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsh_lane_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsh_laneq_f16` | Floating-point fused multiply-subtract from accumulator |  |
| `vfmsq_n_f16` | Floating-point fused Multiply-Subtract from accumulator | fmls |
| `vld4_dup_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vld4_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vld4q_dup_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vld4q_f16` | Load single 4-element structure and replicate to all lanes o... |  |
| `vmaxh_f16` | Maximum (vector) | fmax |
| `vmaxnmh_f16` | Floating-point Maximum Number | fmaxnm |
| `vmaxnmv_f16` | Floating-point maximum number across vector | fmaxnmv |
| `vmaxnmvq_f16` | Floating-point maximum number across vector | fmaxnmv |
| `vmaxv_f16` | Floating-point maximum number across vector | fmaxv |
| `vmaxvq_f16` | Floating-point maximum number across vector | fmaxv |
| `vminh_f16` | Minimum (vector) | fmin |
| `vminnmh_f16` | Floating-point Minimum Number | fminnm |
| `vminnmv_f16` | Floating-point minimum number across vector | fminnmv |
| `vminnmvq_f16` | Floating-point minimum number across vector | fminnmv |
| `vminv_f16` | Floating-point minimum number across vector | fminv |
| `vminvq_f16` | Floating-point minimum number across vector | fminv |
| `vmul_n_f16` |  |  |
| `vmulh_f16` | Add | fmul |
| `vmulh_lane_f16` | Floating-point multiply |  |
| `vmulh_laneq_f16` | Floating-point multiply |  |
| `vmulq_n_f16` |  |  |
| `vmulx_n_f16` | Vector multiply by scalar |  |
| `vmulxh_f16` | Floating-point multiply extended | fmulx |
| `vmulxh_lane_f16` | Floating-point multiply extended |  |
| `vmulxh_laneq_f16` | Floating-point multiply extended |  |
| `vmulxq_n_f16` | Vector multiply by scalar |  |
| `vnegh_f16` | Negate | fneg |
| `vrecpeh_f16` | Reciprocal estimate |  |
| `vrecpsh_f16` | Floating-point reciprocal step |  |
| `vrecpxh_f16` | Floating-point reciprocal exponent |  |
| `vrndah_f16` | Floating-point round to integral, to nearest with ties to aw... | frinta |
| `vrndh_f16` | Floating-point round to integral, to nearest with ties to aw... | frintz |
| `vrndih_f16` | Floating-point round to integral, using current rounding mod... | frinti |
| `vrndmh_f16` | Floating-point round to integral, toward minus infinity | frintm |
| `vrndnh_f16` | Floating-point round to integral, toward minus infinity | frintn |
| `vrndph_f16` | Floating-point round to integral, toward plus infinity | frintp |
| `vrndxh_f16` | Floating-point round to integral, using current rounding mod... | frintx |
| `vrsqrteh_f16` | Reciprocal square-root estimate |  |
| `vrsqrtsh_f16` | Floating-point reciprocal square root step | frsqrts |
| `vset_lane_f16` |  |  |
| `vsetq_lane_f16` |  |  |
| `vsqrth_f16` | Floating-point round to integral, using current rounding mod... | fsqrt |
| `vst1_f16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1_f16_x2` | Store multiple single-element structures to one, two, three,... |  |
| `vst1_f16_x3` | Store multiple single-element structures to one, two, three,... |  |
| `vst1q_f16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1q_f16_x2` | Store multiple single-element structures to one, two, three,... |  |
| `vst1q_f16_x3` | Store multiple single-element structures to one, two, three,... |  |
| `vsubh_f16` | Subtract | fsub |


