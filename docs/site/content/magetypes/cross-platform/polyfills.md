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

Each cell is instructions in the main loop / floats per iteration. On AVX2 an
`f32x16` loop costs the same per float as an `f32x8` loop. On NEON the wider
types do more per iteration, because LLVM does not unroll the `f32x4` loop.

Three costs remain:

- **Registers.** Each `f32x16` value occupies two registers on AVX2 and four on
  NEON, so a kernel with many live vectors runs out sooner.
- **Reductions.** `reduce_add` reduces each part and then adds the results: 13
  instructions for `f32x16` on AVX2 against 6 for `f32x8`, once per call.
- **Tails.** A wider chunk leaves a longer scalar tail: up to 15 elements for
  `f32x16`.

Some operations split cheaply; others require cross-half shuffles, scalar
fallbacks, or numerical fixups. Neither constant overhead nor a universal
speedup over scalar code is guaranteed. Benchmark the operation in its actual
loop. [ISA quirks](@/magetypes/isa-quirks.md) is the semantic reference, including
cases where identically named operations retain different backend behavior.
