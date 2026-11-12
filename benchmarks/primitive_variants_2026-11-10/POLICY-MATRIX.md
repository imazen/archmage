# Complete primitive option matrix — 2026-11-11

AMD V4x and hosted WASM timings are in progress. AMD Scalar/V3/V4, Intel V3 and
the M6 exploratory matrix are complete: all 46 canonical flag sets, 66 workload groups and 1,566 arms,
including existing Magetypes baselines. No default or numerical implementation
changes in this expansion. Pow/exp/log timing expansion is deferred by the owner;
their supported options and redirects are fully inventoried.

The [reference](../../docs/PRIMITIVE-PERMUTATIONS.md) defines the domains,
precision contracts, measured-error scope and unmeasured frontend dimensions.
The generated tables live under `magemath/src/{minmax,recip}/timings/` and are
included in public rustdoc. Each row names the exact canonical flags; these map
to the builders in the adjoining permutation tables.

## Protocol

- Native frozen source `92f51085`; WASM source `643a7cf3`. Full source IDs,
  compiler, hashes and actual flags are in each external `SOURCE.json`.
  Both source archives are preserved. The public numerical kernels are unchanged
  between these inputs; WASM adds the guest/host harness.
- Rust 1.99.1, distribution baseline release compilation, no target-cpu=native.
  Patched zenbench `73302a1c115836b2646f6a9adc32058c20d1a66f`.
- Zen 5 measurements enforce the 32 MiB L3 class; Intel enforces the E-core
  class, not P-cores. Placement comes from retained packet metadata. These
  classes are not interchanged when interpreting timings. M6 remains unconfined.
  Frozen `SOURCE.json` identifies the compiled input; runtime `git_hash` can
  reflect later documentation commits while the same frozen binary runs.
- Min/max f32x4/x8/x16 and f64x2/x4/x8; reciprocal/rsqrt f32x4/x8/x16.
  Streams of 64 and 4096 elements, 64-step vector dependency chains, and
  min/max zero-clamp64 with signed zeros and subnormals. Normal configurations
  are unsupported on the zero workload and are never timed on invalid inputs.
- Ordinary inputs repeat a fixed 64-value pattern; min/max and reciprocal
  include both signs, rsqrt is positive. The min/max chain computes
  `op(x, 2) + 1/32`; reciprocal/root chains compute `op(x) + 1/2`, starting
  from opaque 0.75. Zero-clamp64 reads a zero-filled second buffer: it does
  not measure specialization with a compiler-known literal zero. Full-domain
  NaN/infinity-heavy streams remain unmeasured input distributions here.
- Typed vector proof retained, constant integer flags through the hot graph.
  Automatic/Caller are both instantiated. No scalar frontend/array dispatch,
  first-use detection, invalid-input latency or tail speed claim.
- Strict contexts prescribe five independent processes × 50 paired rounds;
  M6 one exploratory process × 50 pairs under the unrestricted scheduler.
  Every round and every resource failure is retained. No favorable retries.
- Reported means use all retained pairs with equal process weight. Paired-round
  95% intervals use 10,000 within-process bootstrap resamples, NumPy PCG64 seed
  0x531324. Strict table intervals are diagnostic log-process-mean Student-t
  (df=4), Bonferroni family 95% across all 1,500 baseline comparisons, plus timer
  floor. They do not replace zenbench admission/confirmation. Raw analysis keeps
  both intervals. The displayed point divides the all-retained mean ns;
  process intervals describe the geometric mean of process ratios. A clean
  cell is distinct from a qualified whole suite.
- The conservative promotion bound remains upper interval ≤1.02 in every
  required ordinary cell; optimization also needs an upper interval <1.00.
  Missing or flagged contexts prevent a blanket promotion.
- WASM host uses Wasmtime 49.0.5, one shared instance, preparation outside the
  timer and guest batches of 4096 short workloads / 256 long or chain workloads.
  Tables divide host-call ns by the batch size. The nonfusing engine's baseline
  x86_64 target changes additional CPU features; it is not an isolated FMA cost.

Every arm passes preflight before timing: exact min/max selection and requested
zero signs; HIGHP exact f32 graph; bounded approximate relative error; chain
versus repeated one-step results. Native Scalar/V3/V4/V4x, Intel V3, M6 NEON and
WASM strict/fusing/nonfusing pass. Actual WASM fusion probes read 0/1/0.
The fixed timing preflight is spot checking, separate from the earlier dense
mantissa scans and independent exact-rational fixtures.

The frozen strict module also runs with relaxed SIMD disabled. The relaxed
module is rejected specifically at its relaxed-SIMD instruction as the negative
control (Wasmtime CLI 50.0.1; this is validation, not the timed 49.0.5 host).
Feature-boundary validation takes 2 s / peak-RSS 0.06 GiB on M6. Both modules
are unchanged; this adds no timing retry.

All current timing binaries/guests use std + w512 + avx512 Cargo features.
The actual CPU capabilities still come from proofs and module flags. no_std
legality/correctness is inventoried and tested; no_std timing remains unmeasured
in this new matrix, alongside the other explicitly separate frontend leaves.

## M6 exploratory findings

All 3,300 group-round observations were resource-flagged; there are no strict
clean cells. These results are descriptive comparisons under the macOS
scheduler. No subset is promoted by its apparent speed.

For f32x4 zero-clamp64, existing min takes 4.3545 ns, finite unchecked ordered
min M21 takes 4.3525 ns, ratio 0.99953 [0.99080, 1.00817]. This admits zeros and
subnormals; Normal cannot execute this workload. Full portable Number min M00
takes 6.4162 ns, ratio 1.47347 [1.43534, 1.51279]. Specifying stronger full-domain
semantics has a measured exploratory cost here.

For f32x4 stream64 reciprocal, HIGHP R00 takes 3.3841 ns versus existing
Magetypes 4.5298 ns. Unchecked Caller portable MIDP R33 takes 5.9644 ns;
backend MIDP R3b takes 3.5550 ns and backend LOWP R3d 3.3764 ns. For rsqrt,
R33 takes 11.2269 ns, R3b 3.7240 ns and R3d 3.3726 ns versus existing 6.2288 ns.
These grades do not form a universal speed ladder. The full tables preserve
every grade, validation choice and execution flag rather than selecting winners.

Selected bars copied from zenbench's original native report (ordinary
`neon/min/f32x4/stream64`, 50 rounds; exploratory, all flagged):

```text
M18                     ███████████████████████ 4.3ns
existing Magetypes min  ███████████████████████ 4.3ns
M22                     ███████████████████████ 4.3ns
M21                     ███████████████████████ 4.3ns
```

M18 is Full with backend NaN/zero behavior; M22 is Normal unchecked; M21 is
Finite unchecked with ordered zeros. All use Automatic execution and retain
the supplied NEON token. Their contracts differ even where ordinary-input
timings are indistinguishable. Full bars and paired samples are preserved.

## Intel V3: clean cells inside a failed-admission suite

The aggregate is unreliable: 424/16,500 paired group-round observations are
flagged; 32/66 groups are entirely fresh and clean. All five processes and all
50 rounds remain in each comparison. There is no whole-suite promotion.

The clean f32x8 reciprocal chain (250/250 clean) shows a real portable MIDP
cost: R33 takes 290.8175 ns versus existing Magetypes 270.7577 ns, ratio
1.074088 with process-family95 [1.067217, 1.081004]. Its lower endpoint exceeds
the 1.02 bound. Unchecked Caller backend MIDP R3b instead takes 188.5439 ns,
ratio 0.696356 [0.690249, 0.702519]. Their error contracts differ; these are
named configuration tradeoffs, not an unnoticed numerical-default change.

The clean f32x8 reciprocal stream64 takes 10.1082 ns for existing Magetypes,
7.8038 ns for R33 and 4.5859 ns for R3b. HIGHP R00 costs 17.1906 ns here but
only 90.4758 ns on the chain. Throughput and dependency workloads can reverse
the ordering, so one cannot rank a grade from a single ns column.

The f32x8 zero-clamp64 group is flagged (245/250 clean); its descriptive costs
are 3.3892 ns existing min, 6.7906 ns M21 Finite unchecked ordered zeros and
3.3861 ns M31 Finite unchecked backend zeros. Unlike the M6 observation,
ordered zero repairs have visible work here. Normal is still not required for
the inexpensive permissive-zero path, and still cannot accept this workload.

Selected original zenbench bars for `v3/recip/f32x8/chain64`:

```text
R3d                       ██████████ 76.4ns
R00                       ████████████ 90.5ns
R35                       ████████████████████████ 176.6ns
R3b                       ██████████████████████████ 188.5ns
existing Magetypes recip  █████████████████████████████████████ 270.8ns
R33                       ████████████████████████████████████████ 290.8ns
```

R00 is Full HIGHP Automatic; R33/R35 are Normal portable MIDP/LOWP,
AssumeValid Caller; R3b/R3d are Normal backend MIDP/LOWP, AssumeValid Caller.
All retain the supplied V3 token. The same configured kernels process the
streams; no runtime-options argument is passed through the hot loop.

## Zen 5 Scalar: provider selection is a cost dimension

This suite has no entirely clean group: 9,963/16,500 observations are clean.
Every timing below is descriptive, flagged and excluded from promotion.

For f32x8 reciprocal stream64, existing Magetypes division takes 7.1325 ns and
HIGHP R00 takes 7.1326 ns. Portable MIDP AssumeValid Caller R33 takes 109.8284 ns;
the same precision/domain/validation with Automatic execution R13 takes
199.3708 ns. Portable LOWP AssumeValid Caller R35 takes 49.0341 ns. Backend
MIDP/LOWP R3b/R3d take 7.1316/7.1315 ns and route to division on this backend.

For chain64, existing division takes 153.3879 ns, R33 968.4987 ns and R13
1,536.1956 ns. Automatic Scalar selects std's scalar FMA provider; Caller uses
the current exact software graph in this baseline distribution build. These
are retained **typed ScalarToken** observations, not a measurement of the public
scalar/array entry's whole-operation dispatch. Hardware-provider selection is
not a demonstrated speed win here. No source or arithmetic change is made to
hide the cost; further provider optimization is a separate implementation step.

The interval family still contains only the 1,500 baseline comparisons.
These Automatic/Caller ns are not presented with an unregistered pairwise
confidence interval or a new promotion claim.

## Zen 5 V3: retained costs, no fully clean group

The five-process V3 suite retains 10,408/16,500 clean observations; none of its
66 groups is entirely clean. These are descriptive measurements, not a new
clean performance failure or promotion.

For f32x8 chain64, reciprocal takes 159.4310 ns with existing Magetypes,
80.6279 ns with HIGHP R00, 177.6733 ns with portable MIDP R33 and 126.2870 ns
with backend MIDP R3b. Rsqrt takes 218.4796 ns existing, 238.4796 ns R00,
274.7846 ns R33 and 188.6272 ns R3b. These flags have the same definitions
as the Intel section. All retained intervals and validation variants remain
in the nested tables; no favorable rounds are selected.

Selected original zenbench bars for `v3/recip/f32x8/chain64` (flagged):

```text
R3d                       █████████ 43.2ns
R00                       █████████████████ 80.6ns
R35                       ██████████████████████ 100.4ns
R3b                       ███████████████████████████ 126.3ns
existing Magetypes recip  ██████████████████████████████████ 159.4ns
R33                       ██████████████████████████████████████ 177.7ns
```

## Zen 5 V4: wide results remain flagged

The V4 suite has 10,032/16,500 clean observations and no fully clean group.
For f32x16 reciprocal chain64, existing Magetypes takes 170.8448 ns, HIGHP R00
83.2854 ns, portable MIDP R33 184.9541 ns and backend MIDP R3b 140.4351 ns.
For rsqrt chain64 the same order is 228.5043, 244.0332, 281.3762 and
202.5815 ns. These are descriptive costs under the V4 proof on Zen 5, not
confirmed optimizations. The tables preserve all other widths and flag sets.

## Orchestration and resources

The initial Intel run hit the default 300-second child supervisor timeout
before a complete packet. The next launch refused its orphan reservation.
The exact owner PID was verified absent before preserving/releasing only that
ticket. The subsequent full run uses `ZENBENCH_TIMEOUT=7200`; admission, 50
rounds, five processes, confidence and the 2% bound are unchanged. Partial
child files, logs and the ticket are retained. These are orchestration failures,
not measured speed regressions.

Native build/verification: 178 s / peak-RSS 6.19 GiB locally; Intel 172 s /
6.23 GiB. WASM build 130 s / 2.14 GiB; verification 1 s / 0.05 GiB.
M6 build takes 96 s; its wrapper's 0.01 GiB excludes possible sccache daemon
work and is not a whole-build memory claim. M6 analysis uses 2 s / 0.17 GiB.
M6 native timing uses 285 s / peak-RSS 0.03 GiB (minimum proxy-available RAM
9,279 MiB, peak load 2.69); its exploratory return code is zero despite the
retained resource flags. Zero exit does not qualify this unconfined run.
Intel native timing uses 1,601 s / peak-RSS 0.05 GiB, minimum available RAM
21,570 MiB and peak load 1.59. Its return code 1 preserves the confirmation
failure. Intel retained-pair analysis uses 3 s / 0.27 GiB on the M6.
Zen 5 Scalar timing uses 1,799 s / peak-RSS 0.29 GiB, minimum available RAM
42,245 MiB and peak load 2.62; its return code 1 retains failed confirmation.
Zen 5 V3 timing uses 1,692 s / peak-RSS 0.29 GiB, minimum available RAM
42,287 MiB and peak load 2.29; its return code 1 also retains failed confirmation.
Zen 5 V4 timing uses 1,735 s / peak-RSS 0.29 GiB, minimum available RAM
42,065 MiB and peak load 2.81; its return code 1 retains failed confirmation.
The first analysis launch lacked NumPy; a task-local venv supplies NumPy 2.4.2.
No timing rerun was needed for that analysis dependency.

## Reproduction and artifacts

Root justfile recipes: `primitive-policy-inventory`, `primitive-policy-tables`,
`primitive-policy-benchgen`, `primitive-policy-guest`,
`primitive-policy-native-run`, `primitive-policy-wasm-run`,
`primitive-policy-analyze`, `primitive-policy-render`.
Generate/check the catalog from the actual Rust builders. Build binaries once,
freeze hashes, verify, then measure without overlapping builds on that host.

External artifacts are under
`/mnt/v/output/archmage/primitive-options-2026-11-11/`.
The mirror and complete verification manifest will be recorded after all
prescribed contexts finish; an incomplete run is not marked archived/verified.
