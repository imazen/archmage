# What a polyfilled vector compiles to

Assembly inspection, 2026-10-06. Instruction counts only: no timings and no
speed claims.

A logical vector wider than the hardware's registers is a polyfill. An `f32x16`
is two 256-bit halves on AVX2 and four 128-bit parts on NEON and WASM. An
`f32x8` is two 128-bit parts on NEON and WASM. This records what a gain kernel
and a sum kernel compile to at 4, 8 and 16 lanes.

## Result

- **In these kernels a polyfilled operation is its native operations and
  nothing else.** No AVX2 or NEON loop below touches the stack or moves a
  vector between registers. The WASM main loops contain no scalar float
  operation.
- **On AVX2, `f32x16` loops cost the same per float as `f32x8` loops.** Both run
  eight `ymm` operations (64 floats) per iteration, in the same number of
  instructions.
- **On NEON, the wider types do more per iteration.** LLVM does not unroll the
  `f32x4` loops there. The gain loop takes 1.25 instructions per float at
  `f32x4`, 0.75 at `f32x8` and 0.69 at `f32x16`. Fewer instructions did not
  mean less time for that loop: see the timings below.
- **On WASM, the three widths compile to the same work per iteration:** 16
  floats in the gain loop and 32 in the sum loop.
- **A polyfilled reduction costs more.** `reduce_add` reduces each part on its
  own and then adds the results: 13 instructions for `f32x16` on AVX2 against 6
  for `f32x8`. It runs once per call, after the loop.

## Loops

Instructions in the main loop / floats per iteration. WASM counts are
WebAssembly operations.

| Target | Kernel | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|---|
| AVX2 (`v3`) | gain | 19 / 32 | 19 / 64 | 19 / 64 |
| AVX2 (`v3`) | sum | 11 / 32 | 11 / 64 | 11 / 64 |
| NEON | gain | 5 / 4 | 6 / 8 | 11 / 16 |
| NEON | sum | 4 / 4 | 5 / 8 | 9 / 16 |
| WASM SIMD128 | gain | 42 / 16 | 42 / 16 | 42 / 16 |
| WASM SIMD128 | sum | 49 / 32 | 51 / 32 | 55 / 32 |

Instructions in `reduce_add`, after the sum loop:

| Target | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|
| AVX2 (`v3`) | 4 | 6 | 13 |
| NEON | 2 | 3 | 15 |

## Setup

- archmage `68dd8096` (`main`); no library code changed.
- rustc 1.99.0 (b940084d7 2026-09-28), LLVM 23.1.1, release profile, no
  `-C target-cpu`. Targets: `x86_64-unknown-linux-gnu`,
  `aarch64-unknown-linux-gnu` and `wasm32-unknown-unknown`. The ARM and WASM
  code was compiled and read, not run.
- cargo-show-asm 0.2.62 (`--intel` on x86, `--wasm` for WebAssembly).
- Kernels and dump script: `experiments/v4-context-asm/src/group_p.rs` and
  `dump_polyfill.sh` at `b7bcfdde`, branch `draft/v4-context-asm-2026-10-06`
  (not on `main`). Each kernel is one `#[magetypes]` body compiled for `v3`,
  `neon`, `wasm128` and `scalar`. On x86-64, a test checks that the `v3` and
  scalar results agree at each width.
- The gain kernel is the README quick start's. The sum kernel adds each chunk
  into one accumulator vector and calls `reduce_add` at the end.

## Related timing

`benchmarks/arm_codegen_2026-09-05/README.md` timed a byte kernel on an Apple
M4 Pro at 16 bytes per vector (native) and 32 (polyfilled). On 65,536-byte rows
the 32-byte form took 1.20 µs against 2.15 µs. On 16-byte rows it took 11.2 ns
against 10.2 ns, because the whole row fell into its scalar tail.

## Timing

[polyfill_timing_2026-10-06.md](polyfill_timing_2026-10-06.md) times these
kernels on a Neoverse-N1, an Apple M4 Pro and Zen 5. The sum ran 1.4 to 3
times faster with the wider types. The gain kernel did not speed up on the
Neoverse-N1.

## Not covered

- Register pressure. Each `f32x16` value occupies two registers on AVX2 and
  four on NEON, so a kernel with many live vectors runs out sooner. Neither
  kernel here keeps more than a few.
- Operations that move data between the parts (interleaves, transposes,
  narrowing), conversions and transcendentals.
