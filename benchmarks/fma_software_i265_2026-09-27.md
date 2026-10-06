# Software fused multiply-add cost

Source snapshot: `33b122526c79b879ed9ec1b385b48459917aa766`, a measurement-time
draft that was not kept: it is in neither this repository nor GitHub, so the
exact measured source cannot be checked out. The session that measured it
committed the helpers and the benchmark as `11a35a8b`; that they match the draft
can no longer be verified. The `fma` group of `magetypes/benches/nostd_math_perf.rs`
reruns the comparison.

These helpers are the software fallback that `mul_add_portable` uses on the
scalar backend (66986145). Between 11a35a8b and 66986145 plain `mul_add` used
them too; see `mul_add_portable_zen5-m4pro_2026-10-05.md` for the current
per-backend costs.

Host: i265, Intel Core Ultra 7 265K, Linux x86_64, Rust 1.98.1.

Command: `TMPDIR=/home/lilith/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- cargo bench -p magetypes --bench nostd_math_perf -- --group=fma --format=json`.
No `target-cpu=native` or global FMA target feature was enabled.

200 interleaved rounds over 1,024 L1-resident values, including accumulation and
black-box overhead. Software f32 uses TwoSum and round-to-odd; f64 uses libm 0.2.16.
This is a microbenchmark of the fallback, not a prediction for a consumer kernel.
Native x86/NEON vector FMA instructions are unchanged. WASM performance was not
measured, nor was Zen 5 performance.

| Operation | Mean ns per 1,024 values | MAD ns |
|---|---:|---:|
| fma/f32_unfused_1024 | 400.18 | 20.55 |
| fma/f32_fused_software_1024 | 1135.58 | 35.56 |
| fma/f64_unfused_1024 | 405.58 | 21.44 |
| fma/f64_fused_software_1024 | 5025.68 | 147.14 |

See the adjacent JSON for confidence intervals and calibration. Its source hash
corrects the harness's Git HEAD metadata to the actual jj source snapshot.

Resource guard: peak RSS 0.15 GiB, minimum available memory 26,838 MiB,
peak load 0.58. These guard readings are operational diagnostics, not a memory
benchmark.
