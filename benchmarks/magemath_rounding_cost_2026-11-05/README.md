# Polynomial rounding comparison, 2026-11-05

Draft verification in progress. Portable remains fused; the owner has not changed
its rounding contract. `relaxed` is a real surface with the same special-value
handling and backend-dependent rounding. `unfused` is a prototype.

Native timing and x86 exhaustive source: `71015113`. Verification harness and
surface documentation: `6a8539b6`. No polynomial bodies or golden bits changed.

The earlier [portable cost record](../magemath_portable_cost_2026-11-05/README.md)
called magetypes methods “relaxed”. They are `method` in this comparison;
`relaxed` now means `magemath::relaxed`.

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
