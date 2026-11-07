# magemath::portable cost and verification, 2026-11-05

What `portable` costs against magetypes' relaxed methods, per tier, and the evidence
that it returns the same bits everywhere. Kernels at jj change `wrluwyzp`, commit
`289baa32` (draft; the kernels did not change after this run).

| run | host | toolchain | conditions |
|---|---|---|---|
| x86 timing, WASM, compile time | Ryzen 9 5900XT (Zen 3, AVX2), 32 threads | rustc 1.99.1, wasmtime 40.0.1 | otherwise idle, load about 1 |
| NEON timing | Neoverse-N1, 16 vCPU | rustc 1.97.1 | a niced fuzz job on 6 of 16 vCPUs. High per-round noise (CV 20-47%), so paired CIs are the numbers to trust |
| x86 exhaustive | Ryzen 9 7900X, 24 threads | rustc 1.99.1 | shared, run at nice 19 |
| ARM exhaustive | Neoverse-N1 | rustc 1.97.1 | as NEON timing |

Not measured: timing on Zen 5 or any AVX-512 tier, any WASM engine but wasmtime.

Reproduce: `just magemath-bench`, `just magemath-bench-wasm`, `compile_time.sh` here,
`just magemath-exhaustive`, and `asm_probe/` here. Raw zenbench output:
[`raw_ryzen5900xt.txt`](raw_ryzen5900xt.txt), [`raw_neoverse_n1.txt`](raw_neoverse_n1.txt).

## Bits

| check | inputs | result |
|---|---|---|
| Ryzen 9 7900X: scalar, v3, v4x, against the mirrors | all 2^32, 26 ops, 78 runs | all match; 251 s |
| Neoverse-N1: scalar, NEON, against the mirrors | all 2^32, 26 ops, 52 runs | all match; 470 s |
| `just test-magemath-wasm`: strict, relaxed on an engine that fuses, relaxed on one that rounds twice, no SIMD | about 230,000 per op | all match |
| `just test-magemath-no-fma`: qemu Nehalem, dispatcher to scalar | about 230,000 per op | all match |
| `just test-magemath-i686`: qemu-i386, scalar only | about 230,000 per op | all match |
| `just test-magemath-v3-build`: `-Ctarget-cpu=x86-64-v3`, scalar uses hardware FMA | about 230,000 per op | all match |
| debug and release on Ryzen 9 9950X3D (with v4x) and Neoverse-N1 | about 230,000 per op | all match |

The 26 ops are the 21 in the crate with `pow` pinned at several exponents. Each run also
matched one set of golden digests. "Match" means identical bits, or both NaN.

## Against the relaxed methods, tiers that fuse in hardware

Elements per second over 8,192 elements in 1,024 `[f32; 8]` values, L1-resident. "vs
relaxed" is `portable`'s time, paired 95% CI.

| Ryzen 9 5900XT, V3 | portable | unchecked | relaxed | vs relaxed | unchecked saves |
|---|---:|---:|---:|---:|---:|
| `exp2_midp` | 4.10G | 5.88G | 3.84G | -7.5% to -3.9% | 30% |
| `log2_midp` | 3.35G | 5.41G | 3.24G | -5.4% to -0.8% | 38% |
| `pow_midp(2.4)` | 1.23G | 1.69G | 1.17G | -7.3% to -3.2% | 27% |
| `exp2_lowp` | 6.33G | 9.04G | 7.70G | +19.2% to +20.6% | 30% |
| `log2_lowp` | 4.21G | 6.35G | 6.54G | +54.4% to +55.4% | 34% |

| Neoverse-N1, NEON | portable | unchecked | relaxed | vs relaxed | unchecked saves |
|---|---:|---:|---:|---:|---:|
| `exp2_midp` | 660M | 854M | 594M | -9.4% to -8.6% | 23% |
| `log2_midp` | 712M | 956M | 720M | -0.5% to +0.5% | 26% |
| `pow_midp(2.4)` | 271M | 331M | 256M | -2.7% to -1.8% | 18% |
| `exp2_lowp` | 1.00G | 1.34G | 1.11G | +12.3% to +13.3% | 25% |
| `log2_lowp` | 798M | 1.04G | 1.09G | +33.8% to +34.9% | 23% |

`portable` midp is at least as fast because `exp2_midp` drops the clamp. The relaxed
lowp methods have no NaN handling, and `log2_lowp` no edge handling at all, which is
what `portable` lowp pays for. Its `_unchecked` form matches relaxed `log2_lowp` within
2% on both hosts, and takes 16-18% less time than relaxed `exp2_lowp`, which still clamps.

## Tiers that fuse in software

Scalar tier, x86-64 without FMA at compile time, Ryzen 9 5900XT. On real CPUs this runs
only without V3.

| | portable | unchecked | relaxed | relaxed is faster by | `zenmath` mirror, no `std` | `std` |
|---|---:|---:|---:|---:|---:|---:|
| `exp2_midp` | 140M | 146M | 472M | 3.4x | 111M | 453M |
| `log2_midp` | 306M | 343M | 1.33G | 4.3x | 143M | 458M |
| `pow_midp(2.4)` | 96.8M | 109M | 362M | 3.7x | 49.7M | 268M |
| `exp2_lowp` | 185M | 184M | 367M | 2.0x | 175M | 439M |
| `log2_lowp` | 402M | 437M | 2.00G | 5.0x | 229M | 454M |

The aarch64 scalar tier fuses in hardware, one lane at a time, so `portable` and
relaxed are within 3% for midp there (`raw_neoverse_n1.txt`). The dispatcher never picks
that tier on aarch64.

WASM, wasmtime 40.0.1 on Ryzen 9 5900XT. ns per element, best of 200 passes
(`examples/wasm_cost.rs`), not zenbench's paired statistics.

| | portable, strict | relaxed method, strict | ratio | portable, relaxed on a fusing engine | relaxed method there | ratio | portable, relaxed on a non-fusing engine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exp2_midp` | 20.52 | 1.52 | 13.5x | 1.78 | 1.22 | 1.5x | 13.71 |
| `log2_midp` | 12.50 | 0.69 | 18x | 1.38 | 0.61 | 2.3x | 9.61 |
| `pow_midp(2.4)` | 35.64 | 3.14 | 11x | 4.61 | 2.56 | 1.8x | 25.44 |
| `exp2_lowp` | 8.14 | 0.80 | 10x | 0.78 | 0.69 | 1.1x | 6.21 |
| `log2_lowp` | 8.62 | 0.34 | 25x | 0.99 | 0.27 | 3.7x | 7.65 |

## Slice tails

`exp2_midp` per call, ns. The shipped shape is the first column.

| length | Ryzen 9 5900XT: one loop, pad | loop then pad | loop then mirror | Neoverse-N1: one loop, pad | loop then pad | loop then mirror |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.2 | 20.3 | 16.1 | 33.2 | 34.6 | 40.5 |
| 7 | 23.4 | 23.5 | 68.5 | 34.5 | 33.4 | 150.6 |
| 8 | 8.67 | 7.66 | 8.08 | 21.0 | 18.7 | 20.2 |
| 9 | 18.4 | 30.5 | 20.5 | 39.9 | 48.7 | 57.5 |
| 15 | 18.8 | 30.6 | 73.9 | 41.1 | 49.5 | 163.7 |
| 1031 | 271 | 288 | 336 | 1,800 | 1,700 | 1,900 |

These are layout-sensitive: an earlier run, before an unrelated kernel edit, measured
34.3 ns for length 7 in the first column.

## Compile time

`compile_time.sh`, Ryzen 9 5900XT, medians of three cold builds including dependencies,
on the committed script.

| | seconds |
|---|---:|
| `zenmath` check | 0.13 |
| `zenmath` build --release | 0.14 |
| `magetypes` check, no `w512` | 3.49 |
| `magetypes` check, default features | 3.83 |
| `magemath` check | 3.62 |
| `magemath` build --release | 3.84 |
| `magemath` build --release, non-inline dispatchers | 4.89 |
| `magemath` check --features avx512 | 4.57 |
| `magemath` build --release --features avx512 | 5.03 |

A downstream crate, rebuilt alone with the dependencies already built:

| calls | rebuild, s | stripped binary, bytes |
|---|---:|---:|
| nothing | 0.14 | 363,128 |
| one slice op | 0.22 | 376,544 |
| all 21 slice ops | 1.13 | 493,560 |

## Register residency

`asm_probe/` with rustc 1.99, `cargo rustc --release --lib -- --emit=asm -C debuginfo=0 -C
codegen-units=1` for x86-64 and aarch64, then `analyze.py`. The tables are in
[`magemath/docs/DESIGN.md`](../../magemath/docs/DESIGN.md#register-residency).
