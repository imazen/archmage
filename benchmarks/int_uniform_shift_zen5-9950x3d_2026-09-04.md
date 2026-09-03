# Uniform (runtime-count) shift vs shift-by-const — x86 (Zen 5)

x86 counterpart to `int_uniform_shift_apple-m4-pro_2026-09-03.md`, extended
with the 32-bit family added as the PR #71 follow-up. Same bench
(`magetypes/benches/int_uniform_shift.rs`), which as of this run also carries
`const_u32`/`uniform_u32` kernels on `u32x4`.

## Setup

- AMD Ryzen 9 9950X3D (Zen 5) under WSL2; rustc 1.98.1 stable; no
  `-Ctarget-cpu=native`. Figures are means from 2 independent process runs
  (zenbench criterion_compat single-pass protocol; the "0 rounds" warning is
  cosmetic — see the raw companion).
- `u16x8` and `u32x4` are both 128-bit types, so the generic kernels inline at
  the SSE2 baseline. (A first draft used `u32x8`, whose AVX2 ops do NOT inline
  outside a `#[target_feature]` region and measured the call boundary instead
  of the shift — 15x slower than scalar; kept out of the record on purpose.)
- Command: `cargo bench -p magetypes --bench int_uniform_shift --features std`

## u16x8, `v3` label (128-bit lanes, SSE2-inlined), ns/µs per pass

| size | const | uniform | delta |
|---|---|---|---|
| 16 | 1.09–1.11 ns | 1.15–1.18 ns | +5–6% |
| 256 | 6.9–7.1 ns | 6.7–7.0 ns | parity |
| 4096 | 57–59 ns | 59–68 ns | parity–noise (one run +18%, not reproduced) |
| 65536 | 1.1 µs | 1.1–1.2 µs | ≤ +9% |
| 1M | 30.8–30.9 µs | 31.8–32.0 µs | ~+3% |

## u32x4 (the new 32-bit family), ns/µs per pass

| size | const | uniform | delta |
|---|---|---|---|
| 16 | 1.45–1.50 ns | 1.49–1.51 ns | parity |
| 256 | 10.4 ns | 10.2–10.3 ns | parity |
| 4096 | 105–106 ns | 141–149 ns | **+34–40%, reproducible** |
| 65536 | 2.1 µs | 2.3 µs | +9% |
| 1M | 59.9–69.7 µs | 66.5–67.1 µs | noise-level (runs disagree on sign) |

The 4096-element u32 gap is real across runs but NOT explained by the obvious
suspect: disassembly shows the count already hoisted to a register
(`psrld %xmm1, %xmmN`, no in-loop `movd`, loop 2x unrolled), so the
per-iteration difference is only the register-count vs immediate-count PSRLD
form plus whatever loop-shaping LLVM chose differently. Cause not isolated —
recorded rather than hidden. At the sizes the motivating kernels run
(dequant rows, thousands of lanes and up, usually memory-adjacent), the
worst measured cost of one generic body over per-N monomorphisation is the
+9% at 64K lanes; parity below 1K lanes and at streaming sizes.

## cdef-shaped composite (u16, from the same runs)

`cdef_uniform` (saturating_sub + uniform shift) measured FASTER than
`cdef_const` at every size ≥ 256 on this machine (e.g. 105 vs 198 ns at
4096, 1.6 vs 3.1 µs at 64K) — the const composite's codegen is the slower
of the two here, the opposite of the naive expectation. Not investigated
further; noted because it means the "monomorphised upper bound" label from
the bench's doc comment does not hold universally on Zen 5.
