+++
title = "Polyfills"
weight = 1
+++

A logical vector can use several hardware vectors. For example, [`f32x8<NeonToken>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html)
uses two 128-bit halves. The public shape and lane order remain eight f32 values.
No algorithm-level runtime dispatch is needed merely to split those halves.

`w512` enables logical 512-bit types, including polyfills, and is a default
feature. `avx512` additionally enables native AVX-512 implementations. A wider
logical shape is not a promise that the CPU executes it as one instruction.

Use the same complete [generated kernel](@/archmage/getting-started/first-simd.md)
for supported backends. Do not call a bare generic SIMD loop from baseline code
and assume the polyfill will establish the feature context.

## What polyfills cost

In a gain kernel and a sum kernel written at 4, 8 and 16 lanes, each polyfilled
operation compiled to its native operations and nothing else. No AVX2 or NEON
loop touches the stack or moves a vector between registers
([assembly results](https://github.com/imazen/archmage/blob/main/benchmarks/polyfill_asm_2026-10-06.md), rustc 1.99.0; instruction counts, no timings):

| Target | Kernel | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|---|
| AVX2 | gain | 19 / 32 | 19 / 64 | 19 / 64 |
| AVX2 | sum | 11 / 32 | 11 / 64 | 11 / 64 |
| NEON | gain | 5 / 4 | 6 / 8 | 11 / 16 |
| NEON | sum | 4 / 4 | 5 / 8 | 9 / 16 |

Each cell is instructions in the main loop / floats per iteration. Instruction
counts are not times. [Timed](https://github.com/imazen/archmage/blob/main/benchmarks/polyfill_timing_2026-10-06.md) on a Neoverse-N1 and on Zen 5:

- **A sum ran faster with the wider type**, because each part keeps its own
  accumulator. Summing 8,192 floats on the Neoverse-N1 took 1.45 µs with
  `f32x4`, 734 ns with `f32x8` and 488 ns with `f32x16`. On Zen 5 the `f32x16`
  sum ran 1.4 to 1.9 times faster than `f32x8` from 1,024 floats up.
- **An elementwise multiply did not.** On the Neoverse-N1 the three widths ran
  within 1% of each other from 1,024 floats up. On Zen 5 `f32x16` took 2 to 9%
  longer than `f32x8`.
- **At 64 floats the widest type lost:** up to 20% longer than the native
  width.

So a wider polyfilled type pays off for reductions over long inputs. For
elementwise work it made little difference on these two CPUs.

Three costs remain:

- **Registers.** Each `f32x16` value occupies two registers on AVX2 and four on
  NEON, so a kernel with many live vectors runs out sooner.
- **Reductions.** `reduce_add` reduces each part and then adds the results: 13
  instructions for `f32x16` on AVX2 against 6 for `f32x8`, once per call. That
  is what the short inputs above pay for.
- **Tails.** A wider chunk leaves a longer scalar tail: up to 15 elements for
  `f32x16`.

Some operations split cheaply; others require cross-half shuffles, scalar
fallbacks, or numerical fixups. Neither constant overhead nor a universal
speedup over scalar code is guaranteed. Benchmark the operation in its actual
loop. [ISA quirks](@/magetypes/isa-quirks.md) is the semantic reference, including
cases where identically named operations retain different backend behavior.
