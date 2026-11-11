# Reciprocal repeat with patched zenbench — 2026-11-11

**The prior study used unpatched registry zenbench 0.1.11.** Its source was
`983eafaa49ceb20bb41010abc53764ff8accdd5b`; main was 13 commits ahead. This
repeat pins **`73302a1c115836b2646f6a9adc32058c20d1a66f`** in the standalone
benchmark's `[patch.crates-io]` and Cargo.lock. The implementation includes the
observer-affinity changes and macOS identity-bound cumulative CPU-counter fix.
[The source comparison](https://github.com/imazen/zenbench/compare/983eafaa49ceb20bb41010abc53764ff8accdd5b...73302a1c115836b2646f6a9adc32058c20d1a66f)
identifies the exact changes. Checking only the published version missed them.

**This is the current performance assessment. No public graph is promoted.**
Five of 36 strict groups have all 250 retained rounds clean. Every strict suite
fails aggregate confirmation. M6 still has 0/50 clean rounds in every group,
and remains unconfined exploratory timing. The new run finds both a clean
Intel E-core win and a clean Zen 5 stream regression; portability is not a
universal speed win.

## What changed and what stayed fixed

- Archmage study source `9dd48579dd78d8a4d44bccf61eea15ce756c4ef9`.
- Zen 5 executable SHA256 `5569773ad00ff71d4f009273c7af493aaab2ea91a6f9e1184002ccf4ad8f79fe`.
  Intel/M6 compiler and binary identities are retained separately.
- The timed arithmetic, inputs, widths, workload sizes, five-process plan,
  50 interleaved rounds per process, strict thresholds and 2% promotion margin
  are unchanged. M6 retains the explicitly exploratory, one-process plan.
- No timing retry or extension of this patched-source run. All original
  [unpatched measurements](RECIP-QUALIFICATION.md) remain retained. This is a
  new harness-source qualification, not an attempt to select favorable rounds.
- Numerical algorithms and their error bounds did not change. The prior C
  oracle/dense signed-normalized scans, directed exponents and tails remain
  applicable; their coverage and limits are in the linked report.
- Linux trace readback confirms the patched sampler. These suites have no
  measurement-thread pin: observer selection explicitly reports `no_pin`,
  retaining the pre-group inherited mask within the enforced CPU-class cpuset.
  No dedicated spare-observer-CPU isolation is claimed. The patch's narrower
  pin/spare-CPU mode is conditional, not automatically active on every run.
- M6 uses the patched cumulative CPU counters. Its first recorded gate snapshot
  reports CPU load **0.1033357444**, above the unchanged 0.05 limit, with zero
  heavy processes. The correction does not imply a quiet host or enforceable
  macOS core-class placement. `unreliable=false` in exploratory mode does not
  clear the retained per-round failures.

The exact implementations are: existing Magetypes `recip()` including refinement
and rail rescue; portable parallel MIDP (fixed seed, three FMA plus multiply);
current public portable serial MIDP (two fused Newton steps, Normal,
AssumeValid, Caller); HIGHP rounded `1/x`, Caller; and canonical RCP14 (native
AVX-512 at every width, integer-table emulation otherwise). MIDP requires
relative error ≤1e-4 on magnitude 2^-125 through 2^125. The research graphs
spend accuracy relative to the existing working-precision reciprocal. RCP14
has no new public selector.

## Complete matrix, ns

Times are per 64/4096-element stream or 64 vector chain steps, **not ns/lane**.
Intel uses the same enforced **Skymont E-core** class as before. M6 is exploratory.

| CPU / tier | Width / work | Clean | Existing recip | Parallel MIDP | Current serial MIDP | HIGHP division | Canonical RCP14 |
|---|---|---:|---:|---:|---:|---:|---:|
| Zen 5 V3 | f32x4/stream64 | 174/250 | 5.278 | 5.626 | 5.624 | 7.092 | 24.280 |
| Zen 5 V3 | f32x4/stream4096 | 163/250 | 310.017 | 377.671 | 372.786 | 466.189 | 1562.563 |
| Zen 5 V3 | f32x4/chain64 | 190/250 | 158.860 | 162.530 | 188.000 | 76.897 | 451.204 |
| Zen 5 V3 | f32x8/stream64 | 146/250 | 2.586 | 2.835 | 2.855 | 3.735 | 14.324 |
| Zen 5 V3 | f32x8/stream4096 | 250/250 | 154.399 | 189.541 | 189.110 | 236.764 | 907.910 |
| Zen 5 V3 | f32x8/chain64 | 137/250 | 159.790 | 155.663 | 178.068 | 79.106 | 539.819 |
| Zen 5 V3 | f32x16/stream64 | 218/250 | 2.423 | 2.874 | 2.914 | 3.557 | 14.273 |
| Zen 5 V3 | f32x16/stream4096 | 172/250 | 155.425 | 188.582 | 189.253 | 236.260 | 911.462 |
| Zen 5 V3 | f32x16/chain64 | 249/250 | 182.479 | 174.737 | 199.655 | 115.106 | 559.388 |
| Zen 5 V4 | f32x4/stream64 | 198/250 | 5.149 | 5.644 | 5.639 | 7.112 | 2.018 |
| Zen 5 V4 | f32x4/stream4096 | 150/250 | 378.964 | 372.408 | 373.190 | 467.232 | 120.439 |
| Zen 5 V4 | f32x4/chain64 | 51/250 | 159.578 | 163.241 | 188.905 | 78.823 | 46.932 |
| Zen 5 V4 | f32x8/stream64 | 119/250 | 2.576 | 2.804 | 2.830 | 3.601 | 1.569 |
| Zen 5 V4 | f32x8/stream4096 | 248/250 | 156.378 | 195.617 | 194.287 | 236.127 | 80.120 |
| Zen 5 V4 | f32x8/chain64 | 150/250 | 160.254 | 155.879 | 178.534 | 80.881 | 47.183 |
| Zen 5 V4 | f32x16/stream64 | 228/250 | 1.490 | 1.678 | 1.512 | 1.830 | 1.298 |
| Zen 5 V4 | f32x16/stream4096 | 102/250 | 95.591 | 112.906 | 99.992 | 120.610 | 68.872 |
| Zen 5 V4 | f32x16/chain64 | 198/250 | 164.082 | 155.508 | 177.982 | 79.498 | 46.864 |
| Zen 5 Scalar Caller | f32x4/stream64 | 100/250 | 7.155 | 82.599 | 72.878 | 7.132 | 20.410 |
| Zen 5 Scalar Caller | f32x4/stream4096 | 24/250 | 467.475 | 5288.968 | 4649.704 | 467.539 | 1245.367 |
| Zen 5 Scalar Caller | f32x4/chain64 | 100/250 | 95.254 | 664.800 | 525.473 | 95.295 | 444.332 |
| Zen 5 Scalar Caller | f32x8/stream64 | 99/250 | 7.148 | 97.926 | 110.557 | 7.141 | 19.638 |
| Zen 5 Scalar Caller | f32x8/stream4096 | 122/250 | 470.143 | 6248.108 | 7055.079 | 468.315 | 1249.225 |
| Zen 5 Scalar Caller | f32x8/chain64 | 145/250 | 154.978 | 774.304 | 975.894 | 154.906 | 476.517 |
| Zen 5 Scalar Caller | f32x16/stream64 | 126/250 | 7.155 | 82.381 | 88.699 | 7.164 | 20.116 |
| Zen 5 Scalar Caller | f32x16/stream4096 | 124/250 | 626.027 | 5228.650 | 5451.119 | 626.673 | 1250.451 |
| Zen 5 Scalar Caller | f32x16/chain64 | 110/250 | 147.716 | 1159.275 | 1159.746 | 147.637 | 512.147 |
| Skymont E V3 | f32x4/stream64 | 200/250 | 10.116 | 7.557 | 7.302 | 17.327 | 44.233 |
| Skymont E V3 | f32x4/stream4096 | 234/250 | 636.818 | 477.119 | 487.054 | 1114.530 | 2781.452 |
| Skymont E V3 | f32x4/chain64 | 250/250 | 254.365 | 221.567 | 256.652 | 89.325 | 477.148 |
| Skymont E V3 | f32x8/stream64 | 250/250 | 10.030 | 5.949 | 7.779 | 17.239 | 36.022 |
| Skymont E V3 | f32x8/stream4096 | 250/250 | 669.588 | 385.194 | 496.767 | 1114.311 | 2288.941 |
| Skymont E V3 | f32x8/chain64 | 250/250 | 270.713 | 237.063 | 290.884 | 90.492 | 586.195 |
| Skymont E V3 | f32x16/stream64 | 232/250 | 10.099 | 5.895 | 7.413 | 17.248 | 33.954 |
| Skymont E V3 | f32x16/stream4096 | 245/250 | 665.120 | 381.429 | 480.312 | 1117.742 | 2151.024 |
| Skymont E V3 | f32x16/chain64 | 200/250 | 310.468 | 258.360 | 308.714 | 278.387 | 724.220 |
| M6 NEON exploratory | f32x4/stream64 | 0/50 | 4.434 | 5.885 | 5.893 | 3.301 | 21.007 |
| M6 NEON exploratory | f32x4/stream4096 | 0/50 | 302.790 | 389.154 | 389.334 | 243.575 | 1360.018 |
| M6 NEON exploratory | f32x4/chain64 | 0/50 | 117.403 | 106.549 | 123.260 | 43.425 | 486.449 |
| M6 NEON exploratory | f32x8/stream64 | 0/50 | 4.412 | 5.845 | 5.863 | 3.266 | 23.096 |
| M6 NEON exploratory | f32x8/stream4096 | 0/50 | 298.577 | 390.461 | 390.309 | 221.167 | 1495.069 |
| M6 NEON exploratory | f32x8/chain64 | 0/50 | 177.714 | 153.707 | 172.704 | 65.711 | 564.168 |
| M6 NEON exploratory | f32x16/stream64 | 0/50 | 4.428 | 5.871 | 5.850 | 3.292 | 23.169 |
| M6 NEON exploratory | f32x16/stream4096 | 0/50 | 294.682 | 388.997 | 388.780 | 220.054 | 1471.507 |
| M6 NEON exploratory | f32x16/chain64 | 0/50 | 221.171 | 197.392 | 220.042 | 96.664 | 630.395 |

## Clean-cell evidence and tradeoffs

These paired intervals use every retained round; no IQR removal. The report
retains the within-process paired bootstrap and the independent-process paired
log-mean Student-t intervals with family correction, as in the original study.
Intervals below are the five-process family-adjusted diagnostics. They do not
turn a failed whole-suite confirmation into a successful one.

- **zen5/v3 v3/f32x8/stream4096**, 250/250 clean: existing 154.398625 ns → parallel 189.541447 ns; ratio **1.147759–1.312988**.
- **i265/v3 v3/f32x8/stream64**, 250/250 clean: existing 10.030281 ns → parallel 5.949427 ns; ratio **0.585080–0.601322**.
- **i265/v3 v3/f32x8/chain64**, 250/250 clean: existing 270.713135 ns → parallel 237.062823 ns; ratio **0.874518–0.876879**.

The clean Zen 5 long stream is above the 1.02 bound, so it fails replacement
promotion even before considering missing/dirty cells. Intel E-core clean
cells still favor the parallel graph. RCP14 hardware remains fast in observed
AVX-512 cells, but none of this run's AVX-512 groups have all 250 rounds clean.
Its emulation remains expensive on AVX2/NEON. Scalar Caller software-FMA costs
remain substantial. Current HIGHP division is already portable and often wins
chains. No universal precision/speed ordering follows from the profile names.

Actual zenbench bars (descriptive process means, ns):

```text
zen5/v3 v3/f32x8/stream4096
  magetypes recip          ████████ 154.4ns
  portable serial MIDP     █████████ 189.1ns
  portable parallel MIDP   █████████ 189.5ns
  portable HIGHP division  ████████████ 236.8ns
  canonical RCP14          █████████████████████████████████████████████ 907.9ns

i265/v3 v3/f32x8/stream64
  portable parallel MIDP   ████████ 5.9ns
  portable serial MIDP     ██████████ 7.8ns
  magetypes recip          █████████████ 10.0ns
  portable HIGHP division  ██████████████████████ 17.2ns
  canonical RCP14          ██████████████████████████████████████████████ 36.0ns

```

Missing coverage remains native Neoverse admission, physical Intel AVX-512,
WASM performance, first-use dispatch, bit-distributed performance inputs and
cold-build comparisons. Existing numeric evidence covers WASM but does not
supply its timing. No further timing campaign is implied by this report.

## Validation and durable evidence

The patched standalone package passes strict Clippy with default/no-default
features, MSRV 1.89 checks, and original-oracle/source-identity checks. The
production math libraries are unchanged. The Git pin is inside this unpublished
benchmark workspace; no shipped crate depends on the patched benchmark library.

Raw packets retain every independent process and interleaved round, including
failures. Source and zenbench archives, executable hashes, compiler identities,
trace logs and both analyses live under
`/mnt/v/output/archmage/recip-patched-zenbench-2026-11-11/`, mirrored with a
verified SHA256 manifest to
`/mnt/tower/output/archmage/recip-patched-zenbench-2026-11-11/`.
The old evidence directory is preserved independently.

`run-heavy` timing resource summaries: Zen 5 V3 **84 s / peak-RSS 0.25 GiB**,
V4 **80 s / 0.26 GiB**, Scalar **87 s / 0.26 GiB**; Intel **61 s / 0.03 GiB**.
M6 **13 s / BSD wrapper peak-RSS 0.02 GiB**, monitored process-tree guard rather
than a cgroup claim. Builds: AMD **13 s / 0.45 GiB**, Intel **9 s / 0.45 GiB**,
M6 **5 s / 0.24 GiB**. All admission failures remain visible.
