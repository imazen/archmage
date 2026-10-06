# What the AVX-512 context changes for 256-bit kernels

Assembly inspection, 2026-10-06. Instruction selection only: no timings and no
speed claims.

`#[magetypes(define(f32x8), v4(cfg(avx512)), v3, …)]` compiles one body for `v3`
(AVX2 + FMA) and for `v4` (AVX-512 F/BW/CD/DQ/VL). An `f32x8` is eight lanes in
both. This records which kernels come out different in the `v4` copy.

## Result

- **A plain multiply, FMA, rounding or reduction loop over `f32x8` is the same
  code in `v3` and `v4`.** The hot loops match instruction for instruction after
  renaming registers.
- **Two or four `f32x8` per iteration do not become `zmm` operations.** LLVM
  already unrolls the one-vector loop to eight `ymm` multiplies (64 floats) per
  iteration. The paired and quadrupled sources compile to that same loop.
- **Select, bitwise and integer idioms pick AVX-512VL instructions at 256
  bits.** A blend goes through a mask register, which adds an instruction per
  select here. Three-input bitwise selects, rotates, 64-bit absolute value and
  unsigned compares each collapse to one instruction.
- **A loop with 20 live vectors uses `ymm16`–`ymm27` in `v4`.** The `v3` copy
  reloads six vectors from the stack every iteration; the `v4` copy reloads
  none.
- **Code LLVM vectorizes itself goes to `zmm`:** a plain loop under
  `#[autoversion]`, and the scalar tail loops of the `f32x8` kernels.
- **`f32x16` is the type that widens.** Its `v4` loop runs eight `zmm`
  multiplies (128 floats) per iteration; its `v3` loop is the eight-`ymm` loop.
- **`f32x8<X64V4Token>` has no transcendentals and no integer conversions.**
  They need `F32x8Convert`, which the V4 tokens do not implement. A `v4` copy
  that calls `exp2_midp`, `ln_midp` or `to_i32_saturating` does not compile
  (E0599). Those three kernels were compiled for `v4` through `token.v3()`
  instead.

## Setup

- archmage `68dd8096` (`main`); no library code changed.
- rustc 1.99.0 (b940084d7 2026-09-28), LLVM 23.1.1, `x86_64-unknown-linux-gnu`,
  release profile, no `-C target-cpu`.
- cargo-show-asm 0.2.62:
  `cargo asm --lib --features avx512 --intel --simplify <symbol>`.
- Probe crate, agreement tests and dump script: `experiments/v4-context-asm/` at
  `76cc32c7`, branch `draft/v4-context-asm-2026-10-06` (not on `main`).
  `./dump.sh` writes the 50 dumps. `cargo test --features avx512` checks that
  each kernel's `v3`, `v4` and scalar results agree: 24 passed.
- Every kernel has one body, compiled for both tiers. "Hot loop" is the loop
  that runs the vector body; the counts below are instructions in that loop.

## Kernels

| Kernel | Hot loop, v3 / v4 | Widest register, v3 / v4 | Change in `v4` |
|---|---|---|---|
| Gain, one `f32x8` per iteration | 19 / 19 | ymm / ymm | None |
| Gain, two `f32x8` per iteration | 19 / 19 | ymm / ymm | None |
| Gain, four `f32x8` per iteration | 19 / 19 | ymm / ymm | None |
| Compare and `blend` | 23 / 27 | ymm / ymm | `vblendvps` becomes `vpmovd2m` into `k1` and a masked move |
| `abs`, negate, three-input bitwise select | 11 / 8 | ymm / ymm | `vandnps` + `vorps` become one `vpternlogd` |
| Degree-8 Horner, nine constants | 25 / 25 | ymm / ymm | None in the loop; the peeled first iteration uses `{1to8}` operands |
| 20-tap FIR, 20 live coefficient vectors | 34 / 28 | ymm / ymm | `ymm16`–`ymm27`; stack reloads in the loop 6 → 0; frame 776 → 304 bytes |
| `round`, `floor` | 23 / 23 | ymm / ymm | None |
| `to_u8` | 16 / 14 | ymm / ymm | The hand-written V4 body: `vpmovusdb` |
| `reduce_add`; `reduce_max` | 11 / 11 each | ymm / ymm | None |
| `recip`; `rsqrt` | 19 / 21; 23 / 25 | ymm / ymm | The fixup blend goes through `k1` |
| Gain with `f32x16` | 19 / 19 | ymm / zmm | 64 → 128 floats per iteration |
| Plain gain loop, `#[autoversion]` | 11 / 11 | ymm / zmm | 32 → 64 floats per iteration |
| Plain float sum of squares | 27 / 27 | xmm / xmm | None; not vectorized in either tier |

Hand-written AVX2 intrinsics, compiled in a V3 and a V4 function:

| Body | Hot loop, v3 / v4 | Change in `v4` |
|---|---|---|
| `_mm256_blendv_ps` on a `_mm256_cmp_ps` mask | 8 / 9 | The blend goes through `k1` |
| `or(and(a, m), andnot(m, b))` | 8 / 7 | One `vpternlogq` |
| 64-bit min as `cmpgt_epi64` + `blendv_epi8` | 14 / 16 | Mask-register blend; not recognized as `vpminsq` |
| 32-bit rotate as `or(slli, srli)` | 14 / 8 | `vprold` |
| 64-bit absolute value, AVX2 form | 12 / 8 | `vpabsq` |
| Unsigned compare by sign bias, then `movemask` | 16 / 10 | `vpcmpnleud` into a mask register |

`f32x8` methods the V4 tokens lack, compiled for `v4` by running
`f32x8::<X64V3Token>` inside a V4 function:

| Kernel | Hot loop, v3 / v4 | Change in `v4` |
|---|---|---|
| `to_i32_saturating` | 18 / 20 | The fixup blend goes through `k1` |
| `exp2_midp` | 27 / 26 | Mask compare replaces `vpcmpgtd` + `vpandn` + `vblendvps` |
| `ln_midp` | 31 / 30 | Mask-register blend; uses `ymm16`+ |

No hot loop in either tier contains a `call`: the V3 implementations that
`f32x8<X64V4Token>` delegates to inline into the `v4` function and are compiled
with its features.

## Not covered

- Timing. A different instruction is not a faster one; the mask-register
  selects are one instruction longer than `vblendvps` here.
- Bodies other than multiply for the two- and four-vector question.
- Whether the `zmm` tail loops can execute. They sit behind a length guard, and
  an `f32x8` tail holds at most seven elements.
- Other compilers. LLVM's idiom recognition changes between releases.
