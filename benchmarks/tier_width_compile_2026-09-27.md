# Tier-selected width compile probe, 2026-09-27

This measures the expansion shape of selecting existing vector types, not a new
`use(f32x)` parser. No new backend traits, forwarding layers, or library vector
families were introduced for the experiment.

Both cases use the same `rite` kernels, `arcane` entry points, runtime dispatcher,
and pointwise add/tail loop. Fixed width uses x8 in all variants. Tier-selected
width uses V3 x8, V4 x16, and NEON/WASM/scalar x4. Default builds compile V3 and
scalar on this x86 host; AVX-512 builds also compile and dispatch to V4. Both the
consumer and dependencies enable `avx512`, including the consumer gate used by
`incant!`. Builds produced no warnings.

Six cold Cargo target directories per case, serial interleaved order, release,
8 jobs, incremental disabled, no native target flags or compiler wrapper. OS
filesystem caches were not flushed. Host: i265, Core Ultra 7 265K; rustc 1.98.1.
Library source: `cd6cdc40`. Full commands, source/lockfile hashes, and per-unit
results are in the [metadata](tier_width_compile_2026-09-27.meta.json); raw
samples are in the [CSV](tier_width_compile_2026-09-27.csv).

| Features | Fixed-x8 cold build median (range) | Tier-width cold build median (range) |
|---|---|---|
| Default | 2.831 s (2.814–2.842) | 2.839 s (2.822–2.851) |
| AVX-512 | 3.056 s (3.033–3.062) | 3.057 s (3.032–3.074) |

The ranges overlap. This small probe does not establish a compile-time regression
from selecting widths. Magetypes unit medians were unchanged at 1.25 s default
and 1.46 s AVX-512. Cargo's consumer timing was 0.04–0.05 s default and
0.05–0.06 s AVX-512; that resolution is too coarse for a small parser-cost claim.

The release test binaries assert exact results for every row width 0..=65,
offsets 0..4 (exclusive upper bound), and three padded rows. Both cases execute
scalar, V3, and runtime-dispatch paths. The AVX-512 cases additionally execute V4
under SDE 10.8 `-skx`; native AVX-512 hardware was not measured. ARM/WASM source
variants exist but are not part of these x86 timing measurements. The scalar
case is x4, not a new x1 backend. This pointwise test does not establish reduction
accuracy or runtime speed.

The [full measurement log](tier_width_compile_2026-09-27.log) records the resource
guard: 73 s, peak RSS 0.40 GiB, minimum available RAM 26,341 MiB, peak load 1.19.
Complete compiler/test logs and timings HTML remain at
`/home/lilith/data/archmage/tier-width/2026-09-27-verified/`.

Parser/signature-rewriting changes, scalar-x1 API parity, and supporting all
attributes remain unimplemented and unmeasured. This result cannot be used as
a universal overhead figure for those additions or for larger kernels.
