+++
title = "ISA Quirks and Fixups"
description = "Concrete edge-case results, portable contracts, and the cost of fixing ISA differences"
weight = 9
+++

A portable vector type guarantees a lane count; it does not guarantee one machine
instruction, identical floating-point results, or availability under every token.
This page distinguishes **uniform contracts we enforce** from **current differences
callers must account for**. It covers the generic `magetypes::simd::generic` API,
primarily `f32x4`, `f32x8`, and `f32x16`; do not transfer these tables to legacy
architecture-specific wrappers or other conversion methods without checking them.

The floating-point tables assume the normal floating-point environment. Changing
rounding modes or enabling flush-to-zero can change results. `NaN` means a NaN
result, without a guarantee about its payload or sign.

## Results we make portable

| Operation and input | Result on every backend | What we fix up |
|---|---|---|
| `shl_uniform(16)` / `shr_logical_uniform(16)` on 16-bit lanes; likewise count 32 on 32-bit lanes | All zero | x86 count behavior already agrees. NEON clamps before broadcasting the count; WASM masks the shifted result. |
| `shr_arithmetic_uniform(16)` on signed 16-bit lanes `[1, -1, MIN, MAX]` | `[0, -1, -1, 0]` | NEON/WASM clamp the count to 15. Counts as large as `u32::MAX` obey the same contract. |
| `*_const::<0>()` | Input unchanged | NEON supports the zero-count case through the chosen lowering. |
| `*_const::<lane_bits>()` or a negative constant count | Compile error | A front-end const assertion prevents ISA-dependent masking, zeroing, or rejection. No runtime check. |
| `u8` saturating addition `250 + 10` | `255` | Native saturation on hardware backends. Scalar implements the same contract. |
| `u8` saturating subtraction `3 - 10` | `0` | Same as above. Saturating add/sub are exposed for 8/16-bit lanes. |
| Widen unsigned byte `255` / signed byte `-1` | `255u16` / `-1i16` | Signedness selects zero/sign extension. Low/high select the corresponding half of the source lanes. |
| Narrow signed i16 values `[-1, 0, 255, 256]` to u8 | `[0, 0, 255, 255]` | Unsigned destinations still have **signed sources**. Native x86/WASM packs interpret sources that way. |
| Narrow two vectors `a`, `b` | All clamped lanes of `a`, followed by all clamped lanes of `b` | AVX2 packs interleave 128-bit groups; a permutation restores order. Native AVX-512 uses conversions and insertion, plus a zero clamp for unsigned destinations. |
| Signed i16 `MIN.abs_diff(MAX)` | `65535u16` | NEON has native absolute difference. x86/WASM use signed max/min and wrapping subtraction; the unsigned result preserves the full range. |
| Adjacent i16 dot product with both pairs equal to `(MIN, MIN)` | `i32::MIN` | `madd_adjacent` wraps modulo 2³². NEON widens products and pair-adds; x86/WASM dot instructions already wrap this exceptional sum. |
| `pairwise_widen_add` on unsigned pairs `(255u8, 255)` / `(65535u16, 65535)` | `510u16` / `131070u32` per pair | Both inputs widen before addition. NEON/WASM have unsigned pairwise widening instructions; x86 masks and shifts each wider lane before adding. |
| Sum 64 bytes all equal to 255 | `16320u32` | `reduce_add_u32` widens rather than using wrapping byte addition. `sum_abs_diff` has the same maximum for 64 lanes. |
| `round()` on `[2.5, 3.5, -2.5, -3.5]` | `[2, 4, -2, -4]` | Ties-to-even, including the scalar fallback. This differs from Rust's ties-away-from-zero `f32::round()`. |

Runtime byte shifts are deferred; constant byte shifts remain available. x86
implements constant byte shifts with wider shifts and masks because it has no
native byte-shift instruction. No new runtime fixups are introduced by the trait
consolidation.

## Float-to-integer conversion: choose the contract

`to_i32` truncates finite, in-range values. Its name does **not** promise Rust's
saturating `as i32` behavior for exceptional inputs. `to_i32_saturating` does.

| f32 input | `to_i32`: x86 | `to_i32`: NEON / WASM / scalar | `to_i32_saturating`: all |
|---|---|---|---|
| `1.9` / `-1.9` | `1` / `-1` | `1` / `-1` | `1` / `-1` |
| `3_000_000_000.0` | `i32::MIN` | `i32::MAX` | `i32::MAX` |
| `+inf` | `i32::MIN` | `i32::MAX` | `i32::MAX` |
| `-inf` | `i32::MIN` | `i32::MIN` | `i32::MIN` |
| `NaN` | `i32::MIN` | `0` | `0` |

The saturating variant adds compare/select fixups on x86; NEON/WASM/scalar already
have the requested conversion behavior. A caller that has proved the input range
may prefer `to_i32`. This table does not describe `to_i32_round`, f64 conversions,
or pixel-packing helpers.

## Current floating-point differences we do not fix up

These are observable implementation differences, including differences between
widths on the same backend. They are compatibility constraints to review, not
recommendations to depend on unusual NaN behavior.

| Expression | x86 V3, all supported widths | Native V4/V4x f32x16 | NEON / WASM | Scalar f32x4/f32x8 | Scalar f32x16 |
|---|---|---|---|---|---|
| `min(NaN, 1)` and `max(NaN, 1)` | `1` | `1` | `NaN` | `1` | `1` |
| `min(1, NaN)` and `max(1, NaN)` | `NaN` | `NaN` | `NaN` | `1` | `NaN` |
| `simd_ne(NaN, 1)` and `simd_ne(NaN, NaN)` | false mask (ordered comparison) | true mask (unordered comparison) | true mask | true mask | true mask |
| Negate `+0.0` | `+0.0` | `+0.0` | `-0.0` | `-0.0` | `-0.0` |
| Negate `-0.0` | `+0.0` | `+0.0` | `+0.0` | `+0.0` | `+0.0` |

x86 min/max select the second operand when a NaN is involved. NEON uses
`vminq`/`vmaxq`, and WASM uses `min`/`max`, which propagate NaNs. Smaller scalar
vectors use Rust's numeric min/max; scalar f32x16 currently uses comparisons and
selects the second operand when the comparison is false. Rust's numeric min/max
ignore a single NaN and do not promise which equal signed zero is returned.
[The scalar contract is documented by Rust](https://doc.rust-lang.org/std/primitive.f32.html#method.min).

For an unordered not-equal mask, invert `simd_eq`; do not assume `simd_ne` currently
has identical NaN behavior on every tier. If NaNs or signed zeros matter to min/max
or negation, handle them explicitly. Follow-up candidates are an unordered x86
comparison predicate and sign-bit XOR negation; both need their own semantic and
codegen review rather than silently changing a published operation here.

## Fused arithmetic and reductions

| Example | x86 / NEON | WASM / scalar | What we fix up |
|---|---|---|---|
| `a.mul_add(b, -1)` where `a = 1 + 2^-23`, `b = 1 - 2^-23` | `-2^-46` (fused) | `0` (separate multiply/add) | Nothing: the existing method uses native FMA where available. |
| Floating-point `reduce_add` | Association depends on backend and vector shape | Association depends on backend and vector shape | Nothing: no universal cross-backend ULP or relative-error bound. |

Fused versus unfused arithmetic can disagree substantially near cancellation or
intermediate overflow. “Within 1 ULP” is not a general cross-backend guarantee.
Choose the arithmetic formulation according to the application's error budget.

## Reciprocal and reciprocal square root

| Input | `recip()` | `rsqrt()` | Contract and fixup |
|---|---|---|---|
| `+0` / `-0` | `+inf` / `-inf` | `+inf` / `-inf` | Working tier preserves IEEE special cases. |
| `+inf` | `+0` | `+0` | x86 repairs invalid Newton intermediates; NEON uses fused refinement instructions with suitable special-case behavior. |
| `-inf` | `-0` | `NaN` | Same policy. |
| `NaN` | `NaN` | `NaN` | Payload/sign unspecified. |
| Normal inputs in the real domain | Working precision, at most 4 ULP | Working precision, at most 4 ULP | Estimates plus refinement where appropriate; division on other backends. |
| Subnormal inputs | Unspecified at the working tier | Unspecified at the working tier | Use the `_portable` variants when subnormals matter. |

On x86 128/256-bit vectors, special-case repair uses comparison and blending;
native AVX-512 uses fixup instructions. NEON arranges fused refinement to avoid
forming an invalid intermediate multiply. The `_portable` variants use division
(and square root for rsqrt). Raw `_approx` variants have weaker guarantees;
do not infer their special-case behavior from the working-tier table.
`sigmoid_midp` and `silu_midp` use division internally rather than inheriting a raw
reciprocal approximation's exceptional-value behavior.

## Portability of cost and availability

| Abstraction | What can differ | How to use it |
|---|---|---|
| Fixed lane count | A 512-bit vector is one native AVX-512 register, two AVX2 registers, or four NEON/WASM vectors. | Treat width as a logical shape; measure the whole kernel. |
| Token feature superset | V4 does not implement every smaller vector backend just because the CPU supports their instructions. | Use the supported common shape, or explicitly downcast to V3 for a smaller-width kernel. |
| Widening/narrowing | Extraction, concatenation, lane-order repair, and load folding vary by ISA and shape. | Do not assume one instruction per public method. |
| `pairwise_widen_add` followed by vector addition | NEON can fuse the pairwise sum and accumulation into `UADALP`; other ISAs use their own lowering. | Each pair sum is exact, but later accumulation wraps at the destination width. Bound the accumulated range or periodically widen/drain the accumulator. |
| `sum_abs_diff` | x86 SAD produces partial sums natively; the public method returns a completed scalar reduction. | For long loops, vector accumulation followed by one reduction can beat reducing every chunk. |
| Checked slices and indexing | Unknown dynamic indices retain bounds checks; slice views may require length/alignment checks. | Use fixed arrays or loops whose bounds are visible to the compiler. Bounds checks are not unconditionally elided. |
| Unsigned-source narrowing, wider saturation, per-lane shifts | Some ISAs need emulation. Their absence from this API is a scope/cost decision, not mathematical impossibility. | Add an operation only with a concrete consumer and measured lowering. |

## Executable evidence

The concrete floating-point examples are asserted in the `isa_quirks` module of
`magetypes/tests/doc_examples.rs`, across scalar, x86, NEON, and WASM runners.
Integer contracts are covered by `int_widen_narrow.rs`,
`int_uniform_shift_saturating.rs`, and the constant-shift boundary tests.
`convert_saturating.rs`, `precise_reciprocals.rs`, and `sigmoid_silu.rs` cover their
respective fixups. `xtask/codegen.py` checks per-ISA consumer assembly, direct
storage access, and integer formulations. These gates check the tested shapes
and contexts; they do not establish optimal codegen for every possible caller.
