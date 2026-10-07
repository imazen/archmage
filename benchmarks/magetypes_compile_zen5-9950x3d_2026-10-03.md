# magetypes cold build: v0.9.29 vs main (Zen 5, 2026-10-03)

Harness: `benchmarks/magetypes_compile_perf.py` (`just magetypes-compile-perf ROOT`).
Host: AMD Ryzen 9 9950X3D, x86_64-unknown-linux-gnu, rustc 1.99.0 (b940084d7
2026-09-28). Trees: `git archive v0.9.29` (before) and `git archive 02d388f0b`
(after), each with its own Cargo.lock and target directory. Command:
`~/work/zen/scripts/run-heavy --mem 24G -- python3 benchmarks/magetypes_compile_perf.py ~/tmp/archmage-ct-2026-10-03 6`.
Load: 1-minute average 2.3 at the start, peak 2.9; the box was otherwise idle.
Raw samples are in the `.json` file next to this one.

## What it measures

Every crate above magetypes in a build graph waits for it, so its own compile time
lands on each downstream build's critical path. Each sample removes magetypes'
artifacts with `cargo clean -p magetypes` and rebuilds it with the dependencies
already built, so the timed work is the magetypes unit alone; the harness fails if
anything else recompiles. `CARGO_INCREMENTAL=0` matches how a dependency builds.
Six pairs per configuration, alternating which tree goes first.

## Results

Medians, and the paired change: the geometric mean of the six after/before ratios
with an approximate 95% Student-t interval on the log ratios.

| Configuration | v0.9.29 | main | Paired change |
|---|---:|---:|---:|
| dev, default features | 0.88 s | 0.92 s | +6.4% [+2.4, +10.6] |
| release, default features | 0.97 s | 1.01 s | +3.6% [−0.0, +7.3] |
| release, `avx512` | 1.13 s | 1.19 s | +6.7% [+4.5, +8.9] |

## Reading it

The release adds 7,046 generated lines. 5,945 of them are in the generic wrapper
types, where the `_t` constructor methods and their deprecated forwarders live. A cold
build of magetypes takes about 40–60 ms longer on this machine, and still about a
second.

Only the x86-64 host build was measured. aarch64 and wasm32 compile different backend
files. Consumers' own build times were not measured.
