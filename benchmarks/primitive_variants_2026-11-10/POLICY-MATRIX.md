# Complete primitive option matrix — 2026-11-11

AMD, Intel and hosted WASM timings are in progress. The M6 exploratory matrix
is complete: all 46 canonical flag sets, 66 workload groups and 1,566 arms,
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
- Min/max f32x4/x8/x16 and f64x2/x4/x8; reciprocal/rsqrt f32x4/x8/x16.
  Streams of 64 and 4096 elements, 64-step vector dependency chains, and
  min/max zero-clamp64 with signed zeros and subnormals. Normal configurations
  are unsupported on the zero workload and are never timed on invalid inputs.
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
