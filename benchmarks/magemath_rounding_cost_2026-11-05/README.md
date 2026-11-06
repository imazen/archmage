# Polynomial rounding comparison, 2026-11-05

Draft verification in progress. Portable remains fused; the owner has not changed
its rounding contract. `relaxed` is a real surface with the same special-value
handling and backend-dependent rounding. `unfused` is a prototype.

Native timing and x86 exhaustive source: `71015113`. Verification harness and
surface documentation: `6a8539b6`. No polynomial bodies or golden bits changed.

The earlier [portable cost record](../magemath_portable_cost_2026-11-05/README.md)
called magetypes methods “relaxed”. They are `method` in this comparison;
`relaxed` now means `magemath::relaxed`.

## Native cost

`cargo bench -p magemath --bench portable_cost`, 1,024 f32x8 vectors (8,192
values), no target-cpu override. Streaming groups use independent inputs;
feedback repeatedly applies `exp2_midp(-x)`. Entries are **unfused time minus
fused time**, paired 95% confidence intervals from zenbench. Negative is faster.

| kernel | Ryzen 9 5900XT, V3 | Core Ultra 7 265K, V3 | Neoverse-N1, NEON |
|---|---:|---:|---:|
| exp2_midp | +51.1–51.9% | +22.8–27.9% | +12.5–13.2% |
| log2_midp | +23.2–23.8% | +18.2–22.1% | +20.8–21.7% |
| pow_midp(2.4) | +31.8–32.9% | +31.1–33.2% | +17.1–17.9% |
| exp2_lowp | +18.5–19.1% | +18.6–21.1% | +13.1–13.9% |
| log2_lowp | +16.6–17.2% | +11.5–13.9% | +7.1–7.8% |
| exp2_midp feedback | +29.2–35.5% | +9.4–11.1% | +13.3–14.4% |

The scalar vector backend uses software f32 FMA on baseline x86, but hardware FMA
on this std-enabled AArch64 build. The runtime dispatcher selects V3 or NEON on
these CPUs; these groups deliberately select ScalarToken.

| scalar-tier kernel | Ryzen 9 5900XT | Core Ultra 7 265K | Neoverse-N1 |
|---|---:|---:|---:|
| exp2_midp | −70.5 to −70.3% | −70.0 to −69.8% | −0.3 to +0.2% |
| log2_midp | −76.5 to −76.2% | −79.7 to −79.2% | +3.1–3.7% |
| pow_midp(2.4) | −72.2 to −72.0% | −76.7 to −76.5% | +2.3–2.8% |
| exp2_lowp | −51.0 to −50.6% | −50.9 to −50.5% | −3.9 to −3.4% |
| log2_lowp | −71.5 to −71.1% | −75.6 to −73.6% | −0.0 to +0.7% |
| exp2_midp feedback | −48.0 to −40.9% | −50.6 to −49.0% | +1.9–2.4% |

Conditions: Rust 1.99.1 on Ryzen 5900XT, 1.99.0 on Core Ultra 265K, 1.97.1 on
Neoverse-N1. N1 shared its host with a niced fuzz job. Both N1 and Intel report
noisy rounds; use the paired intervals, not an independent ratio of noisy means.
These intervals characterize these runs, not variation between machines or runs.

Resource summaries from inherited logs:

- Ryzen 5900XT: `done rc=0 1015s | peak-RSS 0.19GiB | min-avail 47697MiB | peak-load 1.40`.
- Core Ultra 265K: `done rc=0 733s | peak-RSS 0.42GiB | min-avail 20690MiB | peak-load 1.40`.
- N1: `rc=0`, 821.5 s reported by zenbench. The inherited run used nice without
  run-heavy; it contains no peak-RSS measurement. No ARM benchmark memory claim.

## Scalar mirror cost

`zenmath` default features (no std): fused mirrors always use software FMA,
including on ARM. Unfused mirrors need none. Mean microseconds per 8,192 inputs,
**fused → unfused**, from the same streaming runs; these are point measurements,
not paired confidence intervals between the two mirrors.

| operation | Ryzen 5900XT | Core Ultra 265K | Neoverse-N1 |
|---|---:|---:|---:|
| exp2_midp | 73.5 → 33.9 | 74.2 → 25.4 | 185.9 → 97.8 |
| log2_midp | 56.3 → 18.9 | 70.1 → 11.3 | 127.8 → 55.8 |
| pow_midp(2.4) | 164.0 → 78.9 | 137.1 → 48.0 | 378.5 → 191.5 |
| exp2_lowp | 46.0 → 26.1 | 59.7 → 25.3 | 108.7 → 61.9 |
| log2_lowp | 34.9 → 16.1 | 43.4 → 13.3 | 92.7 → 38.1 |

The `std` feature delegates fused steps to `f32::mul_add`; its bits pass the mirror
checks. Its speed was not measured here: the benchmark's `std` row times std
transcendentals, not zenmath with std enabled.

## Compile time

Core Ultra 7 265K, Rust 1.99.0, medians of three cold builds including dependencies,
eight build jobs, no incremental compilation. Command:
`just magemath-compile-time <fresh-directory>`. The script removes public surface
module declarations only in isolated copies; all polynomial bodies and dependencies
are identical. The original portable-only numbers on Ryzen 5900XT are not a
same-host baseline for this table.

| compiled surface | cargo check | release build |
|---|---:|---:|
| portable only | 2.49 s | 2.63 s |
| portable + relaxed | 2.56 s | 2.72 s |
| portable + relaxed + unfused | 2.65 s | 2.82 s |
| all three, AVX-512 enabled | 3.23 s | 3.54 s |

Adding relaxed costs 0.07 s check / 0.09 s release in these measurements. Adding
the unfused prototype costs another 0.09 s / 0.10 s. Non-inline dispatchers cost
3.68 s for the all-surface release build. Magetypes alone checks in 2.36 s without
w512 and 2.58 s with it. Zenmath checks in 0.07 s and release-builds in 0.08 s.

| downstream portable calls | rebuild of caller | stripped binary |
|---|---:|---:|
| none | 0.08 s | 363,128 bytes |
| one slice operation | 0.16 s | 376,544 bytes |
| all 21 operations | 0.84 s | 493,528 bytes |

Resource wrapper: `done rc=0 109s | peak-RSS 0.41GiB | min-avail 20520MiB | peak-load 1.98`.
The harness now stops on failed builds, preserves existing scratch directories,
and edits dispatcher attributes in `surface.rs`, where they actually live.

## Assembly

Rust 1.99.0, release, codegen-units=1, no target-cpu override. AArch64 output was
cross-compiled on x86. `asm_probe` and `analyze.py` in the earlier benchmark directory:

```sh
cargo rustc --release --lib -- --emit=asm -C debuginfo=0 -C codegen-units=1
cargo rustc --release --target aarch64-unknown-linux-gnu --lib -- --emit=asm -C debuginfo=0 -C codegen-units=1 -D warnings
```

| probe | V3 instructions, fused → unfused | NEON instructions, fused → unfused |
|---|---:|---:|
| exp2_midp | 35 → 41 | 76 → 81 |
| log2_midp | 38 → 42 | 71 → not probed |
| exp2 then log2 | 66 → 76 | 141 → 152 |
| two powers | 121 → 138 | not probed |

Every listed probe has zero stack references and calls, and one input load plus
one output store (ldp/stp on NEON). Unfused probes have zero FMA instructions;
x86 uses vmulps/vaddps and ARM uses fmul/fadd. Fused exp2 has six V3 FMAs or 12
NEON FMAs for eight lanes. Identical relaxed wrappers are compiler aliases of
fused probes on V3. Resource wrapper: x86 peak RSS 0.41 GiB; AArch64 cross-build
0.36 GiB (warning-free probe-only recompile 0.10 GiB).

## Exhaustive comparison

Ryzen 9 7900X, Rust 1.99.1: all 234 combinations (26 operation/exponent entries,
three surfaces, scalar/V3/V4x) match their mirrors over all 2^32 input patterns.
NaNs compare by NaN-ness; all other results compare by exact bits. Command:

```sh
cargo test -p magemath --features avx512 --release --test exhaustive -- --ignored --nocapture --test-threads=1
```

Both tests passed in 641.46 s. Resource wrapper: `done rc=0 648s | peak-RSS
0.41GiB | min-avail 14199MiB | peak-load 31.36`. Another build shared the host;
this is correctness evidence, not a timing comparison.

The following distance counts representable values between fused and unfused
mirror results over all 2^32 inputs, including inputs outside accuracy domains.
Signed zeros have distance zero in this distance statistic; the parity tests
above compare their bits. NaN-ness agreed for every input.

| operation | inputs with nonzero distance | maximum distance |
|---|---:|---:|
| `sqrt` | 0 | 0 |
| `floor` | 0 | 0 |
| `ceil` | 0 | 0 |
| `roundeven` | 0 | 0 |
| `log2_lowp` | 9,389,585 | 9 |
| `exp2_lowp` | 81,295,474 | 1 |
| `ln_lowp` | 7,955,438 | 6 |
| `exp_lowp` | 81,802,570 | 1 |
| `log10_lowp` | 8,327,609 | 11 |
| `pow_lowp(2.4)` | 152,969,871 | 93,472 |
| `pow_lowp(-1)` | 315,767,324 | 86 |
| `pow_lowp(0)` | 0 | 0 |
| `cbrt_lowp` | 0 | 0 |
| `log2_midp` | 11,104,393 | 2 |
| `exp2_midp` | 18,055,330 | 1 |
| `ln_midp` | 9,316,233 | 2 |
| `exp_midp` | 17,940,407 | 1 |
| `log10_midp` | 9,742,707 | 3 |
| `pow_midp(2.4)` | 94,866,817 | 178 |
| `pow_midp(0.4166667)` | 216,140,586 | 45 |
| `pow_midp(-1)` | 209,571,366 | 88 |
| `pow_midp(0)` | 0 | 0 |
| `cbrt_midp` | 0 | 0 |
| `cbrt_midp_precise` | 0 | 0 |
| `sigmoid_midp` | 7,495,580 | 4 |
| `silu_midp` | 7,017,054 | 6 |

These are differences between approximations, not errors against the true value.
In particular, lowp power can cross a polynomial interval boundary; a large ULP
distance there does not by itself violate its relative-error bound. Accuracy
checks retain the published bounds for all three surfaces.

## Artifacts

Raw logs are preserved on dev at
`/mnt/v/output/archmage/magemath-rounding-2026-11-05/`. Final hashes, mirror,
native/WASM timing tables, assembly and compile-time results are pending.
