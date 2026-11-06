# Matrix scope and explicit unavailable cells

Unavailable: no executed Windows or macOS target; no Zen 5 timing. All identified
eligible timing hosts currently run Linux; the returned mac is excluded by the
owner. QEMU is not substituted for another OS. No full feedback-latency matrix:
only the four native representative recurrences in [FEEDBACK.md](FEEDBACK.md).

The requested nine f32 operations (powf, exp, ln, log2, log10, sqrt, floor,
ceil, ties-away round) and scalar f64 log2 are covered. Uniform 2.4 power and
separate ties-even roundeven are additional controls, for 12 operation groups.
Every group uses ordinary, finite edge-heavy and special data. The exact
222 source kernel families/expressions are in [api-matrix.csv](api-matrix.csv).
[coverage.csv](results/coverage.csv) records accepted cells/rounds per run.
Every accepted cell has at least 30 rounds; incomplete first passes are archived.

## Available candidates

Scalar: std and correctly rounded Zenmath at baseline, plus whole scalar loops
inside explicit scalar/V3/V4x/NEON/WASM128 contexts. Other alternatives are
libm 0.2.16, fpmath 0.2.0, micromath 2.1.0, metallic 0.3.1 where public APIs exist;
Zenmath portable/unfused midp/lowp are additional approximation controls.

Magetypes widths f32x4/x8/x16 cover lowp/midp/midp_precise and native primitives.
Magemath f32x8 covers all available requested correct operations and checked
portable/unfused/relaxed approximations. General packed approximate power uses
the existing PowBehavior policy, with a different exponent per lane. No
splat/pow/extract scalar-exponent adapter is timed as a packed general kernel.

Correct Magemath proof and once-buffer runtime-dispatch controls use public
slice APIs. Their existing common path processes 4 f32 inputs with 4 f64
intermediates per chunk. The `/x8` label identifies the public Magemath family,
not a claim that the correct slice loop's internal chunk is 8. Approximate
f32x8 bodies operate on the actual 8-lane vector. Both controls include the
same input-to-in-place-buffer copy required by that slice API; other output
kernels directly write the provided output slice. This is a measured API-shape
cost and is disclosed, not hidden as a pure register-kernel comparison.

## Unavailable public APIs, never substituted

- Magemath correct log10 or f64 log2; Magemath vector widths beyond f32x8.
- Magetypes generic f64x2/x4/x8 log2 methods. All three attempted widths failed
  to compile with E0599 on V3/V4x/scalar; current generated transcendentals are
  f32-only. No lane-map adapter is invented for those f64 vector cells.
- Magetypes packed per-lane-exponent pow; Magemath/Magetypes ties-away round.
  Their ties-even operation is reported separately.
- Metallic public sqrtf/floorf/ceilf; micromath f64; vector APIs in the pinned
  scalar alternative crates.
- Stable core scalar sqrt/floor/ceil/round/mul_add/transcendentals. Raising the
  compiler minimum for algebraic operators does not create these APIs.

A runtime failure is separate from an unavailable API: Metallic's WASM
powf/special cell traps. Its other 23 operation/dataset checks succeeded;
[ACCURACY.md](ACCURACY.md) and the trap manifest identify the exact hole.

## Tiers, compilation and admission

All native builds use baseline target settings, without target-cpu=native.
Proof tokens are captured once per whole-buffer callable, not summoned for
each vector. The direct proof body contains the entire loop. Scalar backend
means ScalarToken, not a ban on baseline compiler auto-vectorization.
V3/V4x/NEON/WASM128 rows are separate explicit proof contexts; the exact
supported/unsupported tier inventory is retained per execution. Caller
requirements assert expected native tiers; WASM inventories are checked for
both scalar and wasm128. Unsupported tiers never become timed omissions
presented as completion.

Intel timings run CPU 0 in enforced performance-core cpuset 0–7; Ryzen runs CPU 0;
Neoverse runs CPU 0 on a quiet homogeneous host. zenbench admission owns benchmark
isolation. Builds use explicit memory/thread budgets, not a global exclusive
heavy-job lock. Every heavy job runs through run-heavy with full retained logs.

The benchmark fixes 1,024 values to compare kernels and public API shapes; it
fits no production constants or size/quality model. No four-axis codec sweep
or scaling claim follows from this fixed-size math workload.
