# Uniform (runtime-count) shift vs shift-by-constant — Apple M4 Pro

What the new `shr_logical_uniform(count)` costs against the
`shr_logical_const::<N>` it is meant to replace, and what `saturating_sub`
costs against a wrapping `-`.

The question this answers is the consumer's: a CDEF-shaped kernel whose shift
amount is `max(damping - msb(strength), 0)` has to be monomorphised over every
reachable `N` if it can only use the const form — 81 instantiations in the
SVT-AV1 port that prompted this work. Collapsing that to one generic body is
only worth doing if the runtime-count form is not measurably slower.

## Provenance

| | |
|---|---|
| Box | Apple **M4 Pro**, 12 cores, macOS 26.5.2 (Darwin 25.5.0) |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18), stable, `aarch64-apple-darwin` |
| Commit | `b37802da` |
| Bench | `magetypes/benches/int_uniform_shift.rs` |
| Command | `cargo bench -p magetypes --bench int_uniform_shift` |
| Harness | zenbench 0.1.7 (criterion-compat) |
| Workload | `u16x8` over a slice of N u16, `chunks_exact(8)`, load → op → store. Runtime counts are `black_box`ed so they cannot fold to an immediate. |
| Raw | `int_uniform_shift_apple-m4-pro_2026-09-03.raw.txt` |

Only the `scalar` and `neon` backends run here: this is an aarch64 box, and
neither Rosetta 2 nor Docker Desktop's amd64 emulation reports AVX
(`avx=false`), so the v3/v4 arms cannot be executed on it at all. Those tiers
are measured by CI, not by this file.

## Results — NEON (`NeonToken`, `u16x8` = `uint16x8_t`)

N is u16 lanes; 8 lanes per vector.

| N | `const/…::<3>` | `uniform(3)` | Δ | `cdef_const` | `cdef_uniform` | Δ |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 1.08 ns | 1.24 ns | +15 % | 1.31 ns | 1.48 ns | +13 % |
| 256 | 6.98 ns | 7.71 ns | +10 % | 9.23 ns | 9.19 ns | −0.4 % |
| 4096 | 70.5 ns | 71.8 ns | +1.8 % | 138 ns | 140 ns | +1.4 % |
| 65536 | 2.4 µs | 2.4 µs | ~0 % | 2.1 µs | 2.1 µs | ~0 % |
| 1048576 | 41.1 µs | 41.3 µs | +0.5 % | 32.6 µs | 32.6 µs | 0 % |

`cdef_*` is the whole clamp — `saturating_sub(threshold, x >> 3)` — in both
forms.

**Reading it.** The uniform form's extra work is one scalar `min` and one
`vdupq_n_s16`, both loop-invariant. At N = 16 there is barely a loop to amortise
them over and they show up as a 13–15 % overhead on a ~1 ns kernel; by N = 4096
they are under 2 %, and from N = 65536 they are inside the noise. Nothing in
between suggests a per-element cost, which is what the lowering predicts: the
splat is hoisted and `USHL` runs at the same rate as the immediate `USHR`.

For the motivating kernel this is the answer: one generic body instead of 81
monomorphs costs nothing measurable at any size a CDEF filter actually runs on.

## Results — saturating vs wrapping (NEON)

| N | `wrapping_sub` | `saturating_sub` | Δ |
|--:|--:|--:|--:|
| 16 | 1.17 ns | 1.17 ns | 0 % |
| 256 | 7.47 ns | 7.72 ns | +3.3 % |
| 4096 | 71.9 ns | 72.3 ns | +0.6 % |
| 65536 | 2.4 µs | 2.5 µs | +2 % |
| 1048576 | 41.1 µs | 41.2 µs | +0.2 % |

`VQSUB` and `VSUB` are the same cost on this core, as expected — the saturating
form is a different opcode, not extra instructions.

## Results — scalar backend (`ScalarToken`, `[u16; 8]` array math)

| N | `const::<3>` | `uniform(3)` | Δ |
|--:|--:|--:|--:|
| 16 | 3.75 ns | 2.47 ns | **−34 %** |
| 256 | 6.82 ns | 9.19 ns | +35 % |
| 4096 | 72.7 ns | 138 ns | +90 % |
| 65536 | 1.2 µs | 2.1 µs | +75 % |
| 1048576 | 34.6 µs | 34.5 µs | −0.3 % |

The scalar backend is the one place the runtime count is *not* free, and the
sign of the effect flips with size. The const form is a fully unrolled
per-lane expression that LLVM constant-folds and re-vectorises; the uniform
form carries a runtime `count >= 16` branch into each lane's expression, which
blocks that. At N = 16 the uniform form is nonetheless *faster*, because the
const path's unrolled setup dominates when there are only two vectors of work.

The gap closes again at N = 1048576, where both forms are memory-bound (34.6 vs
34.5 µs) and the extra ALU work hides behind the loads and stores.

This matters less than it looks: the scalar backend is the no-SIMD fallback,
and a caller on a machine with any vector unit never takes it. It is recorded
here because "the fallback got slower" is exactly the sort of thing that should
not be discovered later.

## Caveat on the harness output

zenbench printed `0 rounds ⚠ only 0 rounds` for every entry in this run; the
per-entry mean, MAD and 95 % CI are still reported. The table's numbers come
from the first of two back-to-back runs; the second reproduced every NEON entry
to within 2 % except `cdef_const/neon/1048576` (32.6 → 34.6 µs, ~6 %), so treat
the 1 M row as ±6 % rather than as the ±0.3 % the CI suggests. The warning is a harness reporting
artifact, not missing measurement, but it means the round-to-round regression
gate did not engage — treat these as point measurements, not as a gate.
