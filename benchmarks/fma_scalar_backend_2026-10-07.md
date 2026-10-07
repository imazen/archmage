# Scalar-backend `mul_add_portable` with the FMA instruction (2026-10-07)

`nostd_math::fmaf` / `fma` now use the CPU's FMA instruction when it has one
(NEON on AArch64; FMA via `X64V3Token::summon()` on x86-64), instead of fusing
in software on every CPU. Before: main `f05d478b`; after: `ce639685`
(`perf(magetypes): fmaf/fma use the FMA instruction where the CPU has one`).

Bench: `magetypes/benches/mul_add_cost.rs`, groups `scalar backend`; 1,024
L1-resident vectors (stream) and 1,024 dependent steps (chain), zenbench means.
`mul_add_portable` is shown as time over the same run's `a * b + c`.

```
cargo bench -p magetypes --bench mul_add_cost --features avx512 -- --group="scalar backend" --format=md   # dev
RUSTFLAGS="-Ctarget-cpu=x86-64-v3" CARGO_TARGET_DIR=target/v3 cargo bench ... (same)                         # dev, v3 build
cargo bench -p magetypes --bench mul_add_cost -- --group="scalar backend" --format=md                       # zen-arm-xl
```

## AArch64, Neoverse-N1 (zen-arm-xl, 16 vCPU)

NEON is in the baseline: `NeonToken::summon()` folds to a constant and the
`#[arcane]` helper inlines, so `mul_add_portable` is one `fmadd` per lane.

| Group | `a * b + c` | before | after |
|---|---|---|---|
| `f32x4` stream | 707 / 687 ns | 14.37 µs (20.3×) | 708 ns (1.03×) |
| `f32x4` chain | 1.73 / 1.79 µs | 11.40 µs (6.6×) | 1.40 µs (0.78×) |
| `f64x2` stream | 772 / 780 ns | 22.79 µs (29.5×) | 719 ns (0.92×) |
| `f64x2` chain | 1.74 / 1.80 µs | 21.30 µs (12.2×) | 1.41 µs (0.78×) |

## x86-64, Ryzen 9 9950X3D (dev), `-Ctarget-cpu=x86-64-v3` build

`X64V3Token::summon()` is a constant and the helper inlines: one `vfmadd` per
lane.

| Group | `a * b + c` | before | after |
|---|---|---|---|
| `f32x8` stream | 584 / 630 ns | 6.95 µs (11.9×) | 614 ns (0.97×) |
| `f32x8` chain | 1.12 / 1.10 µs | 6.82 µs (6.1×) | 714 ns (0.65×) |
| `f64x4` stream | 632 / 630 ns | 15.15 µs (24.0×) | 600 ns (0.95×) |
| `f64x4` chain | 1.15 / 1.09 µs | 12.90 µs (11.2×) | 709 ns (0.65×) |

## x86-64, Ryzen 9 9950X3D (dev), default build

Without `-Ctarget-cpu`, `summon()` is a cached check and the FMA helper cannot
inline into a caller compiled without `fma`, so every lane pays a call. This
path runs only when the scalar backend is forced on an FMA CPU (dispatch picks
the V3 backend there) or through the `f32x1`/`f64x1` scalar types.

| Group | `a * b + c` | before | after |
|---|---|---|---|
| `f32x8` stream | 2.43 / 2.42 µs | 8.27 µs (3.4×) | 7.07 µs (2.9×) |
| `f32x8` chain | 1.12 / 1.13 µs | 6.99 µs (6.2×) | 6.41 µs (5.7×) |
| `f64x4` stream | 1.32 / 1.35 µs | 16.05 µs (12.2×) | 3.65 µs (2.7×) |
| `f64x4` chain | 1.11 / 1.11 µs | 13.71 µs (12.4×) | 3.35 µs (3.0×) |

An array-wise variant (one `#[arcane]` entry per vector with 256-bit FMA inside)
was measured on this build and dropped: it cut streams further (f32x8 6.0 µs,
f64x4 2.4 µs) but a block entry costs about 5.5 ns of latency in a chain
(arrays cross the boundary through memory) against about 0.8 ns per scalar
entry, so f64x4 chains regressed to 5.5 µs, and the only path it serves is the
forced-fallback case above.

## Software path, unchanged

`fmaf_soft` keeps the software algorithm (exact f64 product, 2Sum, round to
odd); `fma` without hardware is `libm::fma`. CPUs without FMA, strict WASM and
relaxed WASM take these, at the costs in
`mul_add_portable_zen5-m4pro_2026-10-05.md` and
`mul_add_wasm_wasmtime_zen5-9950x3d_2026-10-05.md`.
