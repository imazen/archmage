# NeonAesToken — NEON+AES

Proof that NEON + AES is available.

**Architecture:** aarch64 | **Features:** neon, aes
**Total intrinsics:** 117 (67 safe, 50 unsafe, 108 stable, 9 unstable/unknown)

## Usage

```rust
use archmage::{NeonAesToken, SimdToken};

if let Some(token) = NeonAesToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: NeonAesToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: NeonAesToken, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (67 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `vaesdq_u8` | AES single round encryption | aesd | — |
| `vaeseq_u8` | AES single round encryption | aese | — |
| `vaesimcq_u8` | AES inverse mix columns | aesimc | — |
| `vaesmcq_u8` | AES mix columns | aesmc | — |
| `vcreate_p64` | Insert vector element from another vector element | nop | — |
| `vmull_high_p64` | Polynomial multiply long | pmull2 | — |
| `vmull_p64` | Polynomial multiply long | pmull | — |
| `vreinterpret_p16_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_p64` | Vector reinterpret cast operation | nop | — |
| `vset_lane_p64` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_p64` | Insert vector element from another vector element | nop | — |
| `vsli_n_p64` | Shift Left and Insert (immediate) | sli | — |
| `vsliq_n_p64` | Shift Left and Insert (immediate) | sli | — |
| `vsri_n_p64` | Shift Right and Insert (immediate) | sri | — |
| `vsriq_n_p64` | Shift Right and Insert (immediate) | sri | — |

### Stable, Unsafe (41 intrinsics) — use import_intrinsics for safe versions

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `vld1_dup_p64` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_lane_p64` | Load one single-element structure to one lane of one registe... | — |
| `vld1_p64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_dup_p64` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_lane_p64` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_p64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld2_dup_p64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_lane_p64` | Load multiple 2-element structures to two registers | — |
| `vld2_p64` | Load multiple 2-element structures to two registers | — |
| `vld2q_dup_p64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_lane_p64` | Load multiple 2-element structures to two registers | — |
| `vld2q_p64` | Load multiple 2-element structures to two registers | — |
| `vld3_dup_p64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_lane_p64` | Load multiple 3-element structures to three registers | — |
| `vld3_p64` | Load multiple 3-element structures to three registers | — |
| `vld3q_dup_p64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_lane_p64` | Load multiple 3-element structures to three registers | — |
| `vld3q_p64` | Load multiple 3-element structures to three registers | — |
| `vld4_dup_p64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_lane_p64` | Load multiple 4-element structures to four registers | — |
| `vld4_p64` | Load multiple 4-element structures to four registers | — |
| `vld4q_dup_p64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_lane_p64` | Load multiple 4-element structures to four registers | — |
| `vst1_lane_p64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_p64_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p64_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p64_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_lane_p64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_p64_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p64_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p64_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst2_lane_p64` | Store multiple 2-element structures from two registers | — |
| `vst2_p64` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_p64` | Store multiple 2-element structures from two registers | — |
| `vst2q_p64` | Store multiple 2-element structures from two registers | — |
| `vst3_p64` | Store multiple 3-element structures from three registers | — |
| `vst4_p64` | Store multiple 4-element structures from four registers | — |

### Unstable/Nightly (9 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `vld1_p64` | Load multiple single-element structures to one, two, three, ... | vldr |
| `vld1q_p64` | Load multiple single-element structures to one, two, three, ... | "vld1.64" |
| `vld4q_p64` | Load multiple 4-element structures to four registers | ld4 |
| `vst3_lane_p64` | Store multiple 3-element structures from three registers | st3 |
| `vst3q_lane_p64` | Store multiple 3-element structures from three registers | st3 |
| `vst3q_p64` | Store multiple 3-element structures from three registers | st3 |
| `vst4_lane_p64` | Store multiple 4-element structures from four registers | st4 |
| `vst4q_lane_p64` | Store multiple 4-element structures from four registers | st4 |
| `vst4q_p64` | Store multiple 4-element structures from four registers | st4 |


