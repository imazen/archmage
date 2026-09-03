# Integer widening / saturating narrowing vs the array round trip — Apple M4 Pro

What `widen_low`/`widen_high` and `narrow_saturating_*` cost against the only
portable alternative a `#[magetypes]` body has without them: a round trip
through `to_array()` / `from_array()` with the casts written out by hand.

The question this answers is the consumer's. A byte-exact SVT-AV1 port has
`me_sad`, the directional predictors, `residual_i16`/`residual_i32`,
`variance::sse` and the SATD accumulate hand-written per ISA today, because the
`u8 -> u16` and `i16 -> u8` steps had no portable spelling. Collapsing those to
one generic body per kernel is only worth doing if the portable spelling is not
measurably slower than what a hand-written body would do.

## Provenance

| | |
|---|---|
| Box | Apple **M4 Pro**, 12 cores, macOS 26.5.2 (Darwin 25.5.0) |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18), stable, `aarch64-apple-darwin` |
| Commit | `7c4a88f6` (measured at `dfff95e0`; rebased twice since, content identical) |
| Bench | `magetypes/benches/int_widen_narrow.rs` |
| Command | `cargo bench -p magetypes --bench int_widen_narrow` |
| Harness | zenbench 0.1.7 (criterion-compat) |
| Workload | `u8x16` / `i16x8` over a slice of N elements, `chunks_exact`, load → op → store |
| Raw | `int_widen_narrow_apple-m4-pro_2026-09-03.raw.txt` (two back-to-back runs) |

Only the `scalar` and `neon` backends run here: this is an aarch64 box, and
neither Rosetta 2 nor Docker Desktop's amd64 emulation reports AVX
(`avx=false`), so the v3/v4 arms cannot be executed on it at all. **The AVX2
`permute4x64` fixup's cost is therefore not measured by this file** — the
lane-order correctness of that arm is covered by CI, and its cost is one
3-cycle-latency lane-crossing shuffle per narrowed vector by construction.

## Cases

| Case | What it does |
|---|---|
| `widen` | `widen_low()` + `widen_high()`, both halves stored — one instruction per half |
| `widen_via_array` | the same result via `to_array()`, per-lane `as`, `from_array()` |
| `widen_chain` | `u8 -> u16 -> u32` accumulate: both widening pairs in one dependency chain (the `variance::sse` shape) |
| `widen_chain_via_array` | the same chain, every step through arrays |
| `narrow` | `narrow_saturating_u8(hi)`, one call per 16 output bytes |
| `narrow_via_array` | the same result via arrays and an explicit `clamp(0, 255)` |

## Results

Run 1 is tabulated; run 2 is in the raw log and is quoted wherever the two
disagree. Sizes 65 536 and 1 048 576 are at or past the memory-bandwidth knee
(calibration reported 105 GiB/s; `widen` at 1 M moves 3 MB in ~21–28 µs, i.e.
110–145 GB/s) and their run-to-run spread — up to 20 % — is larger than any
difference they show, so **no conclusion is drawn from those two rows.**

### NEON — widening

| N | `widen` | `widen_via_array` | Δ | `widen_chain` | `..._via_array` | Δ |
|---|---|---|---|---|---|---|
| 16 | 0.91 ns | 1.05 ns | **−13 %** | 0.56 ns | 1.67 ns | **−66 %** |
| 256 | 6.25 ns | 6.66 ns | **−6 %** | 8.98 ns | 22.7 ns | **−60 %** |
| 4 096 | 93.1 ns | 86.3 ns | +8 % *(run 2: 80.5 vs 84.6, −5 %)* | 442 ns | 505 ns | **−12 %** |
| 65 536 | 1.2 µs | 1.2 µs | bandwidth-bound | 8.3 µs | 8.3 µs | bandwidth-bound |
| 1 048 576 | 27.7 µs | 27.6 µs | bandwidth-bound | 133.4 µs | 133.9 µs | bandwidth-bound |

### NEON — narrowing

| N | `narrow` | `narrow_via_array` | Δ |
|---|---|---|---|
| 16 | 0.94 ns | 0.81 ns | +16 % *(run 2: 0.78 vs 0.81, −4 %)* |
| 256 | 4.98 ns | 5.20 ns | **−4 %** *(run 2: 4.97 vs 5.39, −8 %)* |
| 4 096 | 75.7 ns | 76.2 ns | −1 % |
| 65 536 | 1.1 µs | 1.1 µs | bandwidth-bound |
| 1 048 576 | 26.7 µs | 21.4 µs | bandwidth-bound *(run 2: 21.1 vs 20.0)* |

### Scalar fallback

| N | `widen` | `via_array` | Δ | `widen_chain` | `via_array` | Δ | `narrow` | `via_array` | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.28 ns | 1.28 ns | 0 % | 1.40 ns | 0.72 ns | **+94 %** | 0.78 ns | 0.79 ns | −1 % |
| 256 | 10.3 ns | 6.80 ns | **+51 %** | 17.4 ns | 11.9 ns | **+46 %** | 5.32 ns | 5.28 ns | +1 % |
| 4 096 | 165 ns | 93.2 ns | **+77 %** | 365 ns | 467 ns | **−22 %** | 76.1 ns | 76.0 ns | 0 % |
| 65 536 | 2.6 µs | 1.1 µs | **+136 %** | 5.9 µs | 8.3 µs | **−29 %** | 1.1 µs | 1.1 µs | 0 % |
| 1 048 576 | 42.0 µs | 21.2 µs | **+98 %** | 95.9 µs | 133.1 µs | **−28 %** | 27.1 µs | 27.3 µs | −1 % |

The scalar `widen` column is the one large-N row where a difference survives the
bandwidth caveat, because it is a *rate* difference (see the fit below), not a
level difference: `widen` at 1 M runs at 71 GB/s, well under the 113 GB/s
ceiling `via_array` is sitting on.

## Fixed overhead vs per-element cost

Least-squares fit of `total = α + β·N` over the three sizes below the bandwidth
knee (N = 16, 256, 4096), run 1:

| Case | α (per call) | β (per element) |
|---|---|---|
| `widen` / neon | 0.51 ns | 0.02261 ns |
| `widen_via_array` / neon | 1.01 ns | 0.02083 ns |
| `narrow` / neon | 0.47 ns | 0.01836 ns |
| `narrow_via_array` / neon | 0.49 ns | 0.01848 ns |
| `widen` / scalar | 0.33 ns | **0.04020 ns** |
| `widen_via_array` / scalar | 0.98 ns | **0.02252 ns** |

Two things fall out of this that the ratio columns hide:

* **On NEON the difference is in the intercept, not the rate.** The primitive
  saves 0.5 ns of per-call setup on widening and 0.02 ns on narrowing, with
  slopes within 8 % and 1 % respectively. That is why the win is large at N = 16
  and 256 and gone by N = 4096 — it is a fixed cost, not a rate.
* **On scalar the difference is in the rate** (0.0402 vs 0.0225 ns/element, 1.79×)
  with a *lower* intercept. That is what makes the scalar regression below grow
  with N instead of amortising away.

The `widen_chain` rows do **not** admit this model: the same fit gives a negative
intercept (−9.9 ns for `widen_chain`/neon, −4.6 ns for its baseline), i.e. the
cost is superlinear over this range and a two-parameter affine fit is the wrong
shape. No α/β is reported for them — only the measured points above.

## What to take from this

1. **On NEON the primitive is free or better, and much better in the shape the
   consumer actually has.** The copy-shaped `widen`/`narrow` cases are within a
   few percent of the array round trip once N is large enough for the intercept
   to amortise, and win by 4–13 % below that. The compute-shaped `widen_chain` —
   where the widened values are *used* rather than immediately stored — is
   **2.5–3.0× faster** at 16 and 256 elements and 12 % faster at 4096. That is
   the SVT-AV1 kernel shape.

2. **REGRESSION: the scalar backend's `widen_low`/`widen_high` is ~1.8× slower
   per element than the identical casts written inline in the caller**,
   reproducibly at every size from 256 up, in both runs. This is a codegen
   effect, not an algorithmic one: the scalar impl is literally
   `core::array::from_fn(|i| a[i] as u16)` — the same expression the baseline
   uses — and it is `#[inline(always)]`. Rewriting it as an explicit sub-array
   plus `.map()` made it **worse** (18.3 ns vs 10.3 ns at N = 256), so the
   obvious fix is not the fix. Reported rather than blocked on: the scalar
   backend is the no-SIMD fallback, the affected shape is a bandwidth-bound copy
   loop, and the compute-shaped `widen_chain` still favours the primitive by
   22–29 % on scalar from N = 4096 up.

3. **The scalar `widen_chain` at N ≤ 256 (+46–94 %) is the other place the
   primitive loses**, and it is the same class as (2): the scalar backend's
   array-returning trait methods cost more than the equivalent expression
   inlined at the call site. It inverts by N = 4096. On the SIMD backends
   neither effect appears.

4. **The AVX2 `permute4x64` fixup's cost is not measured here** — no AVX2 on
   this box. By construction it is one lane-crossing shuffle (3-cycle latency)
   per narrowed 256-bit vector, on top of one `vpackuswb`. The correctness of
   that arm is covered by CI, and by the `narrow_lane_order` assertion in
   `magetypes/tests/int_widen_narrow.rs`, which fails without it.
