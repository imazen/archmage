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

## Runtime fixups: exact calls and extra work

These are **source-level lowerings**, not universal cycle counts. Constants can
fold away, invariant count preparation can move out of a loop, and wider
polyfills repeat the native sequence. The measured table below identifies its
CPU, width, compiler, baseline, and workload.

| Exact generic method call | ISA / shape | Baseline and runtime fixup | Where the extra work occurs |
|---|---|---|---|
| `v.to_i32_saturating()` on `f32x4`, `f32x8`, `f32x16` | V3 native 128/256; W512 uses two halves | `to_i32()` uses truncating conversion. Add an overflow comparison, select `i32::MAX`, unordered comparison, and AND-NOT to zero NaNs. | Four additional vector operations per native half, plus constant materialization if needed. No data-dependent branch in this lowering. |
| Same calls | Native V4/V4x `f32x16` | Truncating conversion plus two mask comparisons and two masked moves. | Per vector; predicate-register form rather than V3's full-vector masks. |
| Same calls | NEON / WASM / scalar | The base conversion already clamps and maps NaN to zero. | No additional semantic repair; benchmark these as identity controls. |
| `v.recip()` / `v.rsqrt()` on `f32x4`, `f32x8`; V3 `f32x16` | V3 native half | Same estimate and Newton arithmetic, then unordered compare and blend back to the estimate on invalid intermediates. | Two semantic repair operations per native half; the inspected lowering also needs a register copy to retain the estimate. Constants, loads and scheduling affect time. |
| `v.recip()` / `v.rsqrt()` on native `f32x16` | V4/V4x | Estimate + Newton + `VFIXUPIMM` with a constant rail table. | One fixup instruction per vector plus table materialization; retaining values can also require register copies. |
| `v.recip()` / `v.rsqrt()` | NEON f32 | Two fused refinement steps. `rsqrt` places `a` and `y*y` inside `FRSQRTS` instead of forming the invalid `a*y` first. | Operand arrangement preserves special cases without an added compare/select. Current probe uses an identity control here, not the historical broken arrangement. |
| `v.shl_uniform(count)` / `v.shr_logical_uniform(count)` on `i16x8/16/32`, `u16x8/16/32`, `i32x4/8/16`, `u32x4/8/16` | NEON | Clamp `count` to lane bits before conversion/broadcast; negate for right shift. This prevents large counts aliasing through NEON's signed low-byte interpretation. | Scalar min/select and broadcast per distinct count; usually hoistable when all loop iterations share the count. |
| `v.shr_arithmetic_uniform(count)` on signed shapes above | NEON / WASM | Clamp to lane bits minus one, preserving sign-fill for excessive counts. | Scalar count preparation; NEON also broadcasts and negates. |
| `v.shl_uniform(count)` / `v.shr_logical_uniform(count)` on the shapes above | WASM | Native shifts mask counts modulo lane bits. Compare the original count and AND the result with all-ones or zero. | Count comparison/splat plus a vector AND; invariant preparation may hoist. |
| All uniform-shift calls above | x86 | Native excessive-count zero/sign-fill behavior already matches. | No semantic repair. Moving a runtime count into a register is still required. |
| `a.narrow_saturating_i8(b)` / `a.narrow_saturating_u8(b)` on `i16x16`; `a.narrow_saturating_i16(b)` / `a.narrow_saturating_u16(b)` on `i32x8` | AVX2 | Native packs interleave 128-bit groups. `VPERMQ` with `0xD8` restores all of `a` followed by all of `b`. | One lane-order permutation per 256-bit pack; the V3 W512 path repeats it twice. |
| Same narrowing methods on `i16x32` / `i32x16` | Native AVX-512 | Narrow each input to a half and insert/concatenate. Unsigned destinations clamp signed inputs to zero before unsigned conversion. | Two zero clamps for unsigned output, plus the two conversions and concatenation needed for the operation. |
| Same narrowing families at 128 bits; NEON / WASM widths | Native halves | Native signed-source saturating narrows already have the desired lane order. | No AVX2-style lane-order repair; polyfills still require composition. |
| `v.shl_const::<N>()`, `v.shr_logical_const::<N>()`, `v.shr_arithmetic_const::<N>()` | Every backend | Const assertions reject invalid counts before execution. | **No runtime assertion.** Constant byte shifts can still require shift/mask emulation. |

`abs_diff`, `madd_adjacent`, `pairwise_widen_add`, saturating arithmetic, and
`reduce_add_u32` sometimes need several instructions because an ISA lacks that
operation. That is emulation of the operation, not an extra check imposed by a
safe wrapper. `round()` enforces ties-to-even; the scalar fallback may need
rounding code while hardware backends have native rounding instructions.

## Measured overhead and recorded values

[Open the interactive ISA explorer](../../isa-explorer/) to filter by CPU, ISA,
width, and exact call; inspect decimal values and raw bits; or download the
JSON with every paired timing round. It is an offline record viewer and does
not claim to execute NEON or AVX-512 in your browser.

{{ isa_explorer() }}

The first measurements use 2,048 ordinary positive f32 values, release builds,
no `target-cpu` override, and nine alternating AB/BA rounds. Timings include
loads, stores, and the loop. The percentage is the median of paired ratios;
MAD describes within-process variation, not a confidence interval. These are
**throughput costs in this kernel**, not instruction latencies or predicted
application slowdowns. Different widths cannot be compared as equal work per
vector. Exceptional inputs are sampled separately, not included in timing.

| CPU / compiler | Exact call / shape | Baseline | Median overhead | Ratio MAD |
|---|---|---|---:|---:|
| AMD Ryzen 9 7950X 16-Core Processor / 1.98.1 | `to_i32_saturating()` / x86-v3/f32x4 | `to_i32()` | +84.2% | 0.6 pp |
| AMD Ryzen 9 7950X 16-Core Processor / 1.98.1 | `recip()` / x86-v3/f32x4 | same Newton arithmetic, no rail repair | +31.8% | 1.6 pp |
| AMD Ryzen 9 7950X 16-Core Processor / 1.98.1 | `rsqrt()` / x86-v3/f32x4 | same Newton arithmetic, no rail repair | +26.7% | 0.5 pp |
| AMD Ryzen 9 7950X 16-Core Processor / 1.98.1 | `to_i32_saturating()` / x86-v4/f32x16 | `to_i32()` | +44.7% | 0.7 pp |
| AMD Ryzen 9 7950X 16-Core Processor / 1.98.1 | `recip()` / x86-v4/f32x16 | same Newton arithmetic, no rail repair | +17.2% | 0.5 pp |
| AMD Ryzen 9 7950X 16-Core Processor / 1.98.1 | `rsqrt()` / x86-v4/f32x16 | same Newton arithmetic, no rail repair | +13.7% | 1.1 pp |
| Apple M4 Pro / 1.98.0 | `to_i32_saturating()` / neon/f32x4 | identity control | -0.0% | 1.2 pp |
| Apple M4 Pro / 1.98.0 | `recip()` / neon/f32x4 | identity control | -0.0% | 0.1 pp |
| Apple M4 Pro / 1.98.0 | `rsqrt()` / neon/f32x4 | identity control | -0.1% | 0.1 pp |

The M4 identity controls show measurement variation, not a benefit from adding
or removing code. Integer clamp/permutation costs above are currently supported
by source/codegen inspection; **isolated baseline-versus-fixup timings have not
yet been recorded for them**. Existing `int_uniform_shift` benches compare
runtime versus constant counts, which also changes instruction selection and
is not an isolated fixup measurement. Do not label that ratio a clamp cost.

Reproduce the sample/timing run with `cargo run --release -p magetypes --example
isa_fixups --features avx512`. Use `-- --samples-only` for Wasmtime/QEMU. Capture
`rustc -Vv` and CPU identification before stdout, then collect runs with
`python3 xtask/isa_evidence.py CPU=run.txt --output docs/site/static/isa-explorer/data.json`.
The collector validates array shapes, bit encodings, and saturating-conversion
outputs against an independent scalar oracle. The dataset records its library
revision and probe source hash. Store actual outputs per CPU; do not synthesize
ISA results from a table or report emulator timings as hardware performance.

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
| Ordinary normal inputs in the tested range | Working precision target: at most 4 ULP | Working precision target: at most 4 ULP | Estimates plus refinement where appropriate; see the documented gap below. |
| Subnormal inputs | Unspecified at the working tier | Unspecified at the working tier | Use the `_portable` variants when subnormals matter. |

**Observed contract gap:** on Ryzen 7950X, V3 `f32x4::recip()` maps
`f32::MAX` (`0x7f7fffff`) to `+0` and `-f32::MAX` to `-0`. Their correctly rounded
reciprocals are nonzero subnormals (`0x00200000` / `0x80200000`). Thus the current
rustdoc promise that *only subnormal inputs* are unspecified is too broad:
normal inputs with subnormal reciprocal outputs can also lose the ≤4 ULP
claim. This is recorded as a gap, not silently treated as a conforming result.
`recip_portable()` supplies the division result. A repair to the working tier
needs a separate accuracy/codegen/performance review; this documentation change
does not add runtime branches or alter the implementation.

On x86 128/256-bit vectors, special-case repair uses comparison and blending;
native AVX-512 uses fixup instructions. NEON arranges fused refinement to avoid
forming an invalid intermediate multiply. The `_portable` variants use division
(and square root for rsqrt). Raw `_approx` variants have weaker guarantees;
do not infer their special-case behavior from the working-tier table.
Transcendental approximation domains and fixups are documented separately in
[Transcendentals](@/magetypes/math/transcendentals.md).

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
