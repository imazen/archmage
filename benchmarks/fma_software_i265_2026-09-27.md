# Software fused multiply-add cost

Source snapshot: `33b122526c79b879ed9ec1b385b48459917aa766` (unlanded FMA work).
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
