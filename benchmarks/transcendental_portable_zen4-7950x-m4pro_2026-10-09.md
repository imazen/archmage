# `*_midp_portable` against the plain `*_midp` transcendentals (2026-10-09)

What one rounding per multiply-add on every backend costs. The portable forms
run the midp algorithms with `mul_add_portable` and pass NaN through; the
plain forms use `mul_add`, which fuses on x86-64-v3/v4 and NEON and rounds
twice on the scalar backend and strict WASM.

Bench: `magetypes/benches/transcendental_portable.rs`, one stream of 1,024
L1-resident vectors (`f32x8`, `f32x16` on AVX-512) per pass, zenbench 0.1.10;
before timing, every portable kernel gave the scalar backend's portable bits.
WASM: `magetypes/examples/wasm_transcendental_bench.rs` (zenbench does not build
for wasm32), `f32x4`, 2,048 elements, median of 9 rotating rounds. rustc 1.99.0;
release builds without `-Ctarget-cpu`.

```
cargo bench -p magetypes --bench transcendental_portable --features avx512 -- --format=md
TRANSCENDENTAL_PORTABLE_NO_FMA=1 cargo bench -p magetypes --bench transcendental_portable --features avx512 -- --format=md
cargo bench -p magetypes --bench transcendental_portable -- --format=md          # aarch64
CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime RUSTFLAGS="-C target-feature=+simd128" \
  cargo run --release -p magetypes --example wasm_transcendental_bench --target wasm32-wasip1 --features std
```

## x86-64: AMD Ryzen 9 7950X (Zen 4), WSL2

zenbench means per pass (µs), load average about 2.

| Group | `log2` | `log2` portable | `exp2` | `exp2` portable | `pow(x, 2.4)` | `pow` portable |
|---|---:|---:|---:|---:|---:|---:|
| `f32x8`, AVX2 (V3) | 1.87 | 2.16 (+15%) | 1.85 | 1.98 (+7%) | 5.94 | 6.23 (+5%) |
| `f32x16`, AVX-512 (V4) | 2.84 | 3.06 (+8%) | 2.09 | 2.33 (+11%) | 6.38 | 7.41 (+16%) |
| `f32x8`, scalar backend, FMA instruction | 5.36 | 33.33 (6.2×) | 14.97 | 65.06 (4.3×) | 20.80 | 93.20 (4.5×) |
| `f32x8`, scalar backend, software FMA (`TRANSCENDENTAL_PORTABLE_NO_FMA=1`) | 5.34 | 79.67 (14.9×) | 14.92 | 78.94 (5.3×) | 19.78 | 264.93 (13.4×) |

On the SIMD tiers both forms fuse with the same instructions, and the portable
forms also pass NaN through (a compare and a blend); they cost 5–16% more. On
the scalar backend the plain forms compute `a * b + c`, while the portable forms
call `nostd_math::fmaf` for each lane, which in a baseline build checks
`X64V3Token::summon()` at run time before using the FMA instruction (4.3–6.2×
the plain time), or fuses in software when the CPU has no FMA (5.3–14.9×).

## AArch64: Apple M4 Pro (mac)

zenbench minima per pass (µs). The means mix performance- and efficiency-core
runs (2.6 against 9.9 µs for `log2`), so the minima compare like with like.

| Group | `log2` | `log2` portable | `exp2` | `exp2` portable | `pow(x, 2.4)` | `pow` portable |
|---|---:|---:|---:|---:|---:|---:|
| `f32x8`, NEON | 2.56 | 2.78 (+9%) | 2.76 | 2.99 (+8%) | 7.14 | 8.94 (+25%) |
| `f32x8`, scalar backend (FMA instruction) | 18.19 | 19.32 (+6%) | 31.54 | 28.99 (−8%) | 55.66 | 56.18 (+1%) |

NEON is in the AArch64 baseline, so `fmaf`'s token check folds away and the
scalar backend's portable forms cost about what its plain forms cost.

## WASM SIMD128: wasmtime 40.0.1 on the Ryzen 9 7950X

ns per element, strict SIMD128 (`-C target-feature=+simd128`).

| Function | midp | portable |
|---|---:|---:|
| `log2` | 1.001 | 11.758 (11.7×) |
| `exp2` | 1.737 | 20.038 (11.5×) |
| `pow(x, 2.4)` | 3.132 | 33.433 (10.7×) |

Every multiply-add fuses in software on WASM (relaxed SIMD may round twice), as
`mul_add_portable`'s own measurements show (8.2× for `f32x4`;
`mul_add_wasm_wasmtime_zen5-9950x3d_2026-10-05.md`).
