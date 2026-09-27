# Adaptive alias cold-build measurements, 2026-09-27

Compared baseline `b79f590d` with implementation `0833022d` on host `i265`
(Intel Core Ultra 7 265K), Rust 1.98.1. Six serial, interleaved builds per case,
release profile, eight jobs, incremental off, fresh target directory each time.
Filesystem caches were not flushed. No target-cpu=native or RUSTFLAGS.

- `before_manual`: old macro crate, manually selected natural-width aliases.
- `after_manual`: new macro crate, identical manual aliases.
- `after_adaptive`: new macro crate, actual `rite(use(f32xN))` selection.

All cases select the same widths: V3 x8, V4 x16, NEON/WASM/scalar x4. The probe
uses the maintained row/tail kernel template, adding 2.0 per element. It executes
scalar, V3, runtime dispatch, and (with AVX-512) V4 under SDE after building each
case. All six case/configuration test binaries passed. The separate integration
suite verifies the other attributes and all ten vector families.

| Features | Case | Total median (s) | Total range (s) | Macro crate median (s) | Consumer median (s) |
|---|---|---:|---:|---:|---:|
| default | before_manual | 2.834 | 2.819–2.850 | 0.310 | 0.040 |
| default | after_manual | 2.848 | 2.841–2.860 | 0.320 | 0.050 |
| default | after_adaptive | 2.839 | 2.817–2.849 | 0.320 | 0.040 |
| avx512 | before_manual | 3.040 | 3.023–3.046 | 0.310 | 0.050 |
| avx512 | after_manual | 3.061 | 3.029–3.065 | 0.320 | 0.055 |
| avx512 | after_adaptive | 3.053 | 3.036–3.063 | 0.320 | 0.060 |

Baseline to adaptive median differences were +0.005 s (default) and +0.013 s
(AVX-512), with overlapping ranges. This experiment does not establish a clear
regression. Cargo's per-crate timings are rounded; do not interpret their small
differences as precise attribution. These results describe this consumer, not
large downstream crates, runtime SIMD performance, or signature rewriting.
No new vector implementations or public traits were added.

Implementation validation: all ten families ran on scalar/V3, SDE V4/V4x, and
QEMU NEON. WASM and i686 integration targets compile; WASM execution was not
measured. Compiler tests reject absent, weaker, and nested ordinary feature
contexts and ambiguous generic backends. Matching/superset contexts and token
alternatives compile. Existing fixed-width use/define tests and website/README
examples passed. Public prelude names remain unchanged in the three-target audit.

Resource guard: `run-heavy: done rc=0 109s | peak-RSS 0.40GiB | min-avail 26414MiB | peak-load 1.66`.
The monitor is an execution guard, not a heap profile.

The [CSV](adaptive_use_compile_2026-09-27.csv),
[log](adaptive_use_compile_2026-09-27.log), and
[metadata](adaptive_use_compile_2026-09-27.meta.json) preserve the measurements,
exact command, commits, host/toolchain, and log checksums. Full Cargo timing HTML,
per-crate JSON, generated consumers, and validation logs remain in
`/home/lilith/data/archmage/adaptive-use/2026-09-27/`.
