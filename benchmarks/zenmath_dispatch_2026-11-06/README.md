# Zenmath dispatch study

Unpublished implementation study, all changes local drafts. The default Zenmath core stays dependency-free. Optional acceleration shares the
existing Magemath algorithms and selects ordinary scalar paths from paired data.

The generated matrix covers f32 sqrt/floor/ceil/trunc/roundeven/round/FMA and the
four accurate transcendentals, plus f64 primitives/FMA. Power common and special/
rescue mixtures are distinct streaming groups. Buffers have 1, 16, 1,024 and
65,536 values. Feedback has 1,024 dependent scalar steps, including the documented
mapping cost; several mappings converge to fixed points. It is workload feedback,
not a universal instruction-latency measurement. No packed feedback is claimed.

Controls: non-dispatch scalar reference, std baseline, std whole-loop feature
contexts, cached-feature branches per value, typed OnceLock function pointer per
value, explicit held proofs per value, whole-loop proof contexts, accurate packed
four-double kernels and genuine eight-double AVX-512 candidate. Whole-loop context
is distinct from a boundary at every scalar call. Every pointer carries the real
selected proof in its context. Its proof match/indirect call are measured costs;
this does not implement glibc IFUNC. Cache state is inline and needs no heap
allocation; case closures allocate during benchmark setup. Initialization is
outside steady-state after verification/warmup. No token-disabling during a run;
no-FMA validation uses a fresh emulator process.

Native inputs/outputs/case closures allocate outside timed iterations. Strict WASM
allocates its case vector and buffers per host invocation, then repeats the kernel;
repeat count is explicit per size/mode in the host. Absolute cross-platform times
therefore include different wrapper costs. std is a timing control, not the
correct-rounding oracle. Inputs and all non-std stream/feedback adapters are checked
bit-exact against the original scalar algorithm before timing; NaN payloads are
unspecified and signed zero is preserved.

Reproduce source with `just generate` (includes scoped rustfmt). The production
primitive generator has its separate root recipe. Run checks/builds on eligible
remote hosts under run-heavy; never use target-cpu=native. Required-tier caller
configuration must accompany validation/timing. Raw logs/JSON/assembly/binaries
and per-build source identities go under the external artifact directory and NAS
mirror, not git. Exhaustive digest work belongs to the separate digest lane.


The complete matrices cover **94 groups** each: 1,094 native x86 cells, 678 NEON
cells and 678 WASM SIMD128 cells. All accepted policy groups have clean gates.
The ten dirty original native groups are preserved and replaced by a clean
119-cell repeat; a filter collision produced a separately retained one-group
partial repeat. The final driver verifies actual measured group sets before save.

[Measured controls and paired intervals](RESULTS.md) include every operation.
[Width comparisons](WIDTHS.md) support conservative V3 automatic selection;
[linked codegen](CODEGEN.md) records actual instructions and limits.
[Artifacts and reproduction](ARTIFACTS.md) identify each frozen source/binary,
locks, logs and mirrors. The separate exhaustive digest study covers 89 full-domain
cells of the unchanged kernels at `5a9cc685`; later facade/routing validation is
identified independently.

The result is selective acceleration, not a universal SIMD/dispatch win. Native
sqrt improves clearly; f32 FMA and scalar exp/ln/log2 have mixed controls and retain
reference paths on native targets. Cached pointers win some x86 FMA comparisons,
but most single-value selector intervals overlap zero and many ARM streams favor
the simpler branch. Packed pow improves common finite inputs but loses on the
special/rescue mixture. Eight-lane AVX-512 is not automatically promoted.
Zen 5, Windows and macOS performance were not measured in this study.

The matched floor-plus-four-exp external consumer has three cold release builds
per configuration on the Ryzen 7900X, Rust 1.99.0: default 0.66–0.67 s / 464,768 B;
archmage 2.56–2.62 s / 466,176 B; runtime-dispatch 2.62–2.63 s / 471,368 B.
These are this consumer's build time and executable size, not general crate cost.
The nine-build wrapper reported peak RSS 0.29 GiB. An earlier consumer did unequal
work and is retained only as an earlier observation.

Final ordinary facade checks are additional to those matrices: native 12 groups /
147 cells with two dirty groups replaced by a clean 24-cell repeat; final WASM
three groups / 24 cells, all clean, with `ordinary/baseline` required in catalog
and results. Its ordinary-minus-branch CIs are sqrtf large −0.060
[−0.172, −0.0001], floorf large +0.074 [−0.022, +0.181], and f64 round feedback
+0.265 [−0.054, +0.678] ns/value or step. Close differences are not a universal
ranking. A stale-guest 21-cell branch-only attempt is preserved and excluded.

Fresh SIMD and strict no-SIMD WASM targets each passed 18 nontrapping tests plus
the separately asserted exact length trap. Strict fallback required scalar,
forbade SIMD128, and disabled SIMD in Wasmtime. Actual final native no-FMA
facade tests passed under QEMU Nehalem. The complete Zenmath/Magemath
library/integration/examples/doc suites and lint passed on x86 and NEON; Rust
1.89 feature checks and actual external unification checks passed. Bare-metal
no_std default/archmage builds and lint passed. Exact commands are retained
in `final-validation.sh`, `build-wasm.sh`, `test-wasm.py` and full logs.

Preservation limit: the original untimed native `proof` helper was not retained.
The original measured executable was recovered to its exact SHA; all other
measured executables are archived. Each timing executable runs adapter validation
itself, and its logs/source remain available. This is reported explicitly in
`retention-status.json` and the final auditor.
