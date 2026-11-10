# Draft family build costs

The power study measured production revision `4da80e0a` with 54 fresh target builds on Core Ultra 7 265K with
Rust 1.99.0, four Cargo jobs and an 8 GiB wrapper cap. Toolchain and dependency
downloads were cached. Each configuration has three replicates. The whole
wrapper took 171 seconds and peaked at 0.29 GiB RSS.

| Consumer / enabled families | Backend | Cold seconds (three-run range) | Executable bytes |
|---|---|---:|---:|
| Primitives | Plain | 1.21–1.29 | 416,504 |
| Correct rounding | Plain | 1.54–1.61 | 441,032 |
| Normal | Plain | 1.59–1.60 | 441,016 |
| Configurable | Plain | 1.30–1.32 | 443,336 |
| All families | Plain | 1.62–1.63 | 454,936 |
| All families | Archmage proof | 3.46–3.47 | 455,160 |
| All families | Runtime dispatch | 3.52–3.56 | 471,320 |
| Combined Zenmath/Magemath | Plain | 5.30–5.40 | 501,512 |
| Combined Zenmath/Magemath | Archmage proof | 5.33–5.42 | 501,512 |
| Combined Zenmath/Magemath | Runtime dispatch | 5.36–5.44 | 518,600 |

The primitive workload consumes floor. Correct/normal consume the same
precise exp/log/power inputs; normal shares the precise implementation today.
Configurable consumes all nine power policies on identical positive inputs.
Combined consumers also consume packed power. Family-to-family rows therefore
measure actual available workloads, not identical computation. Backend
comparisons within each family keep the workload identical. Shared output
rows match across every tested feature union/backend. NaNs are canonicalized
only for contract comparison; signed zeros are retained.

In all nine combined builds, Cargo timelines show Zenmath and Magetypes
starting at the same recorded instant, with 0.54–0.56 seconds of overlapping
compilation. Zenmath took 0.54–0.56 seconds and Magetypes 0.85–0.87 seconds
within those timelines. Their shared Archmage prerequisite remains. This is
observed scheduling; no build with an artificial dependency edge was measured,
so these data do not establish a causal parallel-build speedup.

Width sixteen/AVX-512 was not enabled in this cost matrix. Ordinary widths
four/eight leave that support off through dependency defaults. Strict builds
and parity tests separately cover `w512`.

Commands, exact locks, source identities, per-run resource logs, binaries and
Cargo HTML/unit timelines are retained by
[the power optimization study](../../benchmarks/zenmath_pow_opt_2026-11-06/README.md).
The study's `family-build-cost.py` reproduces the 54-build matrix.
