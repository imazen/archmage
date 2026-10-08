# Real-consumer inline policies — 2026-10-08

These measurements support preserving legacy body hints and proof-wrapper
inlining during migration. They do not establish a universal dispatcher policy.
Requiring an explicit body policy in attune remains an API recommendation; an
explicit spelling and a default emitting the same attributes generate the same
policy. No library defaults or parser behavior changed in this investigation.

## Scope and provenance

- Archmage/magetypes/macros: fixed main `e2dbab66ef5aa08f8e23ed05248e7d1217f58475`, version 0.9.30.
- [zenav1-svt](https://github.com/imazen/zenav1-svt): `224c6bbb9a3dc577c64f6e0531bda5c3f60c6b0b`.
- [rav1d-safe](https://github.com/imazen/rav1d-safe): `5b51b144cf73216fffd1100bfc396bc7dc17b5bd`.
- rustc 1.98.1, LLVM 22.1.8, x86_64 Linux, Intel Core Ultra 7 265K, runtime pinned to CPU 2. No native CPU targeting.
- Two centered crops from one photographic tree-bark image, 8-bit I420: 256×256 QP32/preset8; 512×512 QP20/preset6. Encoder and decoder operate in memory; decoder uses one thread.
- Profile `release`: opt3, LTO off, 16 codegen units. Profile `ship`: opt3, fat LTO, one codegen unit. Both LTO and codegen-unit count change between profiles.

[Methodology and reproduction](../../experiments/inline-real/README.md) describe
the exact attribute edits and timing boundaries. Consumer Rust implementations
are unchanged. Each policy changes one emitted layer from baseline, leaving
explicit consumer attributes and magetypes implementations intact. These are
legacy-emitter experiments, **not complete attune-rewrite benchmarks**.

Baseline inserts body `#[inline]` hints and proof-wrapper `#[inline(always)]`;
autoversion retains its existing dispatcher attributes. `body_none` removes the
implicit body hint, `body_never` replaces it with `inline(never)`, `proof_none`
removes the wrapper attribute, `proof_hint` substitutes a hint, and
`dispatcher_always` forces autoversion dispatcher inlining. Explicit arcane
`inline_always` remains honored in the body variants.

Each cell summarizes five separate process runs, each with 20 timed calls and
100 ms warmup. Policy order is shuffled per pass with recorded fixed seeds.
The reported time is the median of the five process medians. The range is the
minimum/maximum process median, not a confidence interval. Changes compare that
median against baseline. These are not paired zenbench A/B samples; no claim of
statistical significance is made for small differences.

All 120 processes produced identical encoded bytes and decoded pixels for the
same input, across all policies and both profiles. All 240 groups had 20 samples
and passed zenbench's reliability flag. The exporter enforces the complete
matrix, those checks, and parity before writing these results.

## Results

Positive changes mean more elapsed time (slower). Decode and encode are separate
API calls; setup, destruction, process startup and file I/O are outside timing.

### release, 256×256

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 6.304 [5.542, 6.441] | +0.00% | 2.262 [1.525, 2.332] | +0.00% |
| body_none | 7.433 [6.677, 7.639] | +17.91% | 2.334 [1.535, 2.375] | +3.20% |
| body_never | 8.440 [7.653, 8.554] | +33.88% | 2.436 [1.697, 2.502] | +7.68% |
| proof_none | 8.633 [7.768, 8.741] | +36.94% | 2.264 [1.508, 2.369] | +0.09% |
| proof_hint | 6.482 [6.312, 6.529] | +2.82% | 2.291 [2.185, 2.365] | +1.27% |
| dispatcher_always | 6.429 [5.581, 6.557] | +1.98% | 2.232 [1.532, 2.364] | -1.35% |

### release, 512×512

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 122.456 [121.810, 123.323] | +0.00% | 6.864 [6.031, 6.941] | +0.00% |
| body_none | 145.189 [144.368, 145.829] | +18.56% | 6.735 [6.079, 6.905] | -1.88% |
| body_never | 172.391 [171.978, 173.216] | +40.78% | 7.054 [6.338, 7.234] | +2.78% |
| proof_none | 159.819 [159.294, 160.010] | +30.51% | 6.812 [5.964, 6.972] | -0.75% |
| proof_hint | 122.716 [122.500, 123.408] | +0.21% | 6.929 [6.696, 7.054] | +0.95% |
| dispatcher_always | 122.969 [122.058, 123.073] | +0.42% | 6.759 [5.992, 6.886] | -1.53% |

### ship, 256×256

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 4.136 [4.090, 4.185] | +0.00% | 2.079 [2.051, 2.100] | +0.00% |
| body_none | 4.188 [3.351, 4.200] | +1.26% | 2.094 [2.012, 2.213] | +0.73% |
| body_never | 6.008 [5.270, 6.115] | +45.25% | 2.322 [1.550, 2.442] | +11.69% |
| proof_none | 4.146 [3.362, 4.313] | +0.25% | 2.045 [1.331, 2.162] | -1.65% |
| proof_hint | 4.140 [4.038, 4.232] | +0.10% | 2.036 [2.002, 2.139] | -2.08% |
| dispatcher_always | 4.146 [3.324, 4.291] | +0.23% | 2.063 [1.323, 2.199] | -0.79% |

### ship, 512×512

| Policy | Encode ms [range] | Change | Decode ms [range] | Change |
|---|---:|---:|---:|---:|
| baseline | 76.567 [76.344, 76.841] | +0.00% | 6.416 [6.375, 6.631] | +0.00% |
| body_none | 76.771 [76.609, 77.098] | +0.27% | 6.475 [6.450, 6.578] | +0.92% |
| body_never | 116.598 [115.630, 117.333] | +52.28% | 7.141 [6.355, 7.219] | +11.30% |
| proof_none | 76.636 [76.537, 76.691] | +0.09% | 6.514 [6.342, 6.550] | +1.53% |
| proof_hint | 76.485 [76.440, 76.757] | -0.11% | 6.494 [6.368, 6.682] | +1.23% |
| dispatcher_always | 76.746 [75.818, 77.049] | +0.23% | 6.485 [5.669, 6.540] | +1.08% |

## Interpretation and limits

For the non-LTO encoder, removing implicit body hints increased time by 17.91%
and 18.56%; removing proof-wrapper attributes increased it by 36.94% and 30.51%.
Those penalties largely disappeared with the fat-LTO/one-unit profile. This
supports preserving the legacy hint and proof-wrapper policy for users who do
not build with that shipping profile. It does not prove which individual
functions caused the changes.

`inline(never)` is materially different from no attribute. In the 512 encode it
increased time by 40.78% without LTO and 52.28% with the shipping profile. It
should be an explicit opt-out, not the interpretation of legacy `no_inline`.

Proof-wrapper hints were close to always-inline for these cases; this does not
establish equivalence for larger or generic callers. Dispatcher changes were
small, and these workloads do not establish that always-inline dispatchers are
universally preferable. Small differences can include measurement and code-layout
effects. The encoder source inventory contains no autoversion attributes,
so encoder deltas in the dispatcher-only variant are not evidence of dispatcher
speedups or regressions.

Coverage is two still-frame cases from one photo on one CPU, not broad codec
coverage. No video sequences, HDR, film grain, ARM or AVX-512 runtime cases were
measured. Per-callee codegen attribution and the complete attune implementation
remain unmeasured here. This does not close the draft's cold-compile gate.

## Build observations

One fresh build per cell; these are observations, **not a compile-time regression
estimate**. GNU `size` text includes its aggregate read-only contribution and
is not labeled as pure executable `.text`. RSS below comes from `/usr/bin/time -v`.

| Profile | Policy | Build seconds | GNU size text bytes | Maximum RSS KiB |
|---|---|---:|---:|---:|
| release | baseline | 16.390 | 9878974 | 1231388 |
| release | body_none | 15.446 | 9914182 | 1233144 |
| release | body_never | 15.838 | 10007538 | 1207248 |
| release | proof_none | 15.671 | 9904494 | 1222676 |
| release | proof_hint | 15.854 | 9879134 | 1230548 |
| release | dispatcher_always | 15.646 | 9883882 | 1229440 |
| ship | baseline | 54.813 | 9085764 | 1091224 |
| ship | body_none | 54.952 | 9052340 | 1088152 |
| ship | body_never | 53.344 | 9124936 | 1089896 |
| ship | proof_none | 55.383 | 9069608 | 1093212 |
| ship | proof_hint | 55.332 | 9086048 | 1092416 |
| ship | dispatcher_always | 55.152 | 9090984 | 1092932 |

## Commands, resources and artifacts

Run from the repo root, serially through the resource-limiting wrapper, with
`TMPDIR=/home/lilith/tmp`, 16G memory cap and eight build jobs:

```sh
python3 experiments/inline-real/run.py prepare \
  --out /home/lilith/tmp/attune-inline-real-2026-10-08 \
  --sources /home/lilith/tmp/attune-consumer-cold/sources \
  --revision e2dbab66ef5aa08f8e23ed05248e7d1217f58475 \
  --image /home/lilith/tmp/png-corpus-sample/2400-unsplash-textures/2404_textures_tree-bark-texture_by-hwang-sanguk-k0tljwrabxg-unsplash_3024x3024.sdr.png
python3 experiments/inline-real/run.py build --out /home/lilith/tmp/attune-inline-real-2026-10-08
python3 experiments/inline-real/run.py measure --out /home/lilith/tmp/attune-inline-real-2026-10-08 --passes 5
python3 experiments/inline-real/run.py export --out /home/lilith/tmp/attune-inline-real-2026-10-08 --report /absolute/new/report
```

The recorded builds were split into baseline-release, remaining-release and all
ship commands using `--profiles`/`--policies`; every exact Cargo command is in
[builds.json](builds.json). An initial baseline smoke run used a separate output
folder and is excluded from the result matrix.

Resource-wrapper completion lines (observations as reported by the wrapper):

```text
prepare:          rc=0 6s    | peak-RSS 0.11GiB | min-avail 28901MiB | peak-load 0.29
baseline build:   rc=0 17s   | peak-RSS 1.17GiB | min-avail 26877MiB | peak-load 1.96
other non-LTO:    rc=0 79s   | peak-RSS 1.18GiB | min-avail 26878MiB | peak-load 4.86
fat-LTO builds:   rc=0 329s  | peak-RSS 1.04GiB | min-avail 26491MiB | peak-load 4.12
runtime matrix:   rc=0 1040s | peak-RSS 0.02GiB | min-avail 29025MiB | peak-load 1.48
```

[release.csv](release.csv) and [ship.csv](ship.csv) preserve all process-level
measurements; [summary.json](summary.json) contains the computed comparisons.
[plan.json](plan.json) records pins, input hashes, compiler, dependency selection
and equal lockfile hashes. Its consumer-source capture lists older crates.io
archives as provenance of that capture; the actual selected macro stack is the
fixed-main revision above, not those older archives.
[artifacts.json](artifacts.json) records correctness hashes and the raw-output
root. [raw-artifacts.csv](raw-artifacts.csv) indexes full sample/build logs by
SHA-256. Reference binaries are retained outside Cargo `target/`, with hashes in
[builds.json](builds.json). [machine.json](machine.json) records the observed CPU
and commands to resolve it. Images, payloads, full logs and binaries remain in
the raw-output directory; they are not committed.
