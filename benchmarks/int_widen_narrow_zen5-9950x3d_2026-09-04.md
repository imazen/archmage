# Widening / saturating narrowing vs array round-trip — x86 (Zen 5)

x86 counterpart to `int_widen_narrow_apple-m4-pro_2026-09-03.md`, from the
AVX2 arms that PR could not execute. Adds two things to the bench: a
`v3_arcane` label (the six generic kernels called inside one `#[arcane]`
AVX2 region, plus a 256-bit narrow pair) and this record.

## The bare-generic `v3` label measures a call boundary, not the ops

`widen_low` lowers to `_mm_cvtepu8_epi16` — SSE4.1, **not** in the x86-64
baseline — so outside a `#[target_feature]` region every widen is an
outlined call while the `via_array` casts inline and auto-vectorize at
SSE2: `widen/v3/4096` = 395 ns vs 54.7 ns for the array form, a 7x
artifact that vanishes inside the region (`widen/v3_arcane/4096` = 54.3 ns).
The M4 record never hit this because NEON is the aarch64 baseline. x86
conclusions below are from the `v3_arcane` label — the shape real
`#[magetypes]`/`#[arcane]` consumers have.

- Machine: AMD Ryzen 9 9950X3D (Zen 5), WSL2, rustc 1.98.1, no
  `-Ctarget-cpu=native`. Two process runs; zenbench criterion_compat
  single-pass ("0 rounds" warning is cosmetic).

## Widening (u8x16 → 2×u16x8), `v3_arcane`

| size | widen | via_array | verdict |
|---|---|---|---|
| 16 | 1.13 ns | 0.94 ns | call-overhead zone |
| 256 | 5.60 ns | 5.75 ns | parity |
| 4096 | 54.3 ns | 54.9 ns | parity |
| 1M | 22.2 µs | 22.1 µs | parity (memory-bound) |

Copy-shaped widening is parity on AVX2 — LLVM auto-vectorizes the cast
loop to the same `pmovzx`. The win is compute-shaped:

## Widening chain (u8 → u16 → u32 accumulate), `v3_arcane`

| size | chain | via_array | speedup |
|---|---|---|---|
| 16 | 1.13 ns | 2.44 ns | **2.2x** |
| 256 | 8.42 ns | 33.3 ns | **4.0x** |
| 4096 | 258 ns | 538 ns | **2.1x** |
| 65536 | 4.6 µs | 8.5 µs | **1.8x** |
| 1M | 73.6 µs | 137.4 µs | **1.9x** |

Unlike the M4 (tie above 4096), Zen 5 keeps the win at every size.

## Narrowing (i16 pairs → u8, saturating), `v3_arcane`, both runs

| size | 128-bit primitive | 128-bit via_array | 256-bit primitive (permute fixup) | 256-bit via_array |
|---|---|---|---|---|
| 16 | 1.07–1.13 ns | 0.99–1.01 ns | 0.98 ns | 0.94 ns |
| 256 | 6.17–6.47 ns | 5.57–5.81 ns | **3.58 ns** | 6.85 ns |
| 4096 | 63.1–63.5 ns | 54.0–55.5 ns | **41.4 ns** | 67.3 ns |
| 65536 | ~1.1 µs | ~880 ns | **787 ns** | 989 ns |
| 1M | 23.3–23.5 µs | 22.5–23.1 µs | 23.4 µs | 22.8 µs |

Two honest findings:

1. **At 128-bit the primitive is parity to ~15% SLOWER** than the scalar
   clamp loop, because LLVM already auto-vectorizes that loop to `packus` —
   when the pattern is clean enough for it to recognize, which is a
   heuristic, not a guarantee.
2. **At 256-bit — the width that carries the `permute4x64::<0xD8>` fixup —
   the primitive is 1.3–1.9x FASTER** at every compute-relevant size, and
   faster than the 128-bit primitive as well. The lane-order fixup costs
   less than double-pumping 128-bit packs; the width where the silent-wrong
   trap lives is also the width where the primitive wins outright.

## Verdict for the perf gate

Widening: earns its keep on AVX2 (2–4x compute-shaped, parity otherwise).
Narrowing: at the trap-carrying 256-bit width it is both the only *correct*
portable spelling and the fastest one; at 128-bit its case is guarantee and
portability rather than speed. Scalar-backend regression from the M4 record
reproduces here amplified (`widen/scalar/4096` 353 ns vs 64.6 ns) — same
codegen effect, same conclusion: documented, not blocking, scalar is the
no-SIMD fallback.
