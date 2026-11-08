# Exp-boundary portability and fused arithmetic study

The exceptional-value candidate fails the 2% performance promotion gate on
real NEON scalar callers. It needs an explicit correctness/cost decision before
landing. Quiet AMD and Intel timings and a complete WASM timing matrix are still
missing; busy-host results cannot fill those gaps. The fixed-fused polynomial
candidate is experimental and is not the library default. The separate x86_64
std f64 fallback candidate has correct FMA-only execution but unmeasured
no-FMA and hardware-path costs; it also remains unpromoted.

## Change and contracts

The exceptional-value change affects only generic `exp2_lowp` and checked
`exp2_midp` clamps. Native backends
put the input second in both min/max operations; ScalarToken uses `f32::clamp`.
NaN inputs return NaN on every backend, with payload/sign unspecified. The
existing relaxed polynomial, coefficients, overflow/underflow masks, public
signatures and dependencies remain unchanged. A separate f64 std fallback
change is described below; it does not change polynomial rounding.

Exp aliases and sigmoid/SiLU inherit NaN classification. Checked pow now
preserves internal NaNs from negative bases, NaN exponents and invalid
logarithm-times-exponent products (`1^inf`, `0^0`, `inf^0`). Lowp pow's zero mask
still wins, including `pow_lowp(0, NaN) = +0`. NaN bases still enter the legacy
bitwise log: shared log/pow NaN defects remain. This is not a std `powf` contract.

For finite inputs to a clamp, reversing the operands preserves its result:
the bounds are distinct nonzero values, so signed-zero ties cannot arise.
`f32::clamp` also returns the original in-range value. The remaining arithmetic
is identical to frozen main. Sampled ordinary valid inputs additionally check
all 35 names at widths 4/8/16 against a mechanically extracted main graph.
This is not an exhaustive binary-pow or full-f32-domain accuracy claim.

## Frozen experiment

Baseline: `6bfd83d744e41259ec758054631dcb0928fc4ea1` (0.9.31).
The core-clamp body is frozen at `22f5b2dd91b2`. The later x86_64 std f64
fallback candidate `68320dbd` is excluded by cfg on the measured NEON/WASM
paths; it has not passed the x86 performance gate. Rust 1.99.1 (`58cf8f9c3`), LLVM 23.1.1,
release, one codegen unit, thin LTO, baseline distribution target settings;
no `target-cpu=native`. See [MATRIX.md](MATRIX.md) for all mandatory axes and
controls, including the combined workload, cached dispatch and first use.

`freeze.py` extracts actual baseline methods; no baseline polynomial is
handwritten. Its experimental fused trait replaces only `mul_add_relaxed` with
`mul_add_fused` in the candidate graph. The library retains relaxed arithmetic.
`oracle.py` independently transcribes that graph using rational RNE FMA/division,
then compares separate MPFR 4.2.2 directed references. A matching graph is not a
mathematical accuracy guarantee.

Each timing cell retains 50 shuffled paired rounds. The common zenbench lock
and strict admission gates must all be clean and `unreliable=false`. Analysis
uses every retained candidate/baseline ratio and 100,000 bootstrap samples.
PASS requires upper paired 95% CI <=1.02; lower >1.02 is FAIL; a crossing
interval is INCONCLUSIVE. No outlier filtering, best-of-run selection or
aggregate offsetting of failing cells. The first-use cell uses the single preregistered extension to200 total pairs;
its original28 strict/25 fusing pairs hit a wall safety cap with clean gates.
All original pairs are retained. See MATRIX.md for the frozen extension budget.

## Logged regressions

Real Neoverse-N1, ScalarToken, x4, 64 elements, ordinary throughput streams;
R is the arithmetic mean of paired candidate/main ratios. Every row below is
admitted and fails the 1.02 promotion bound.

| Operation | R | Paired 95% CI |
|---|---:|---:|
| exp2_lowp | 1.076899 | [1.072619, 1.081332] |
| exp_lowp | 1.067653 | [1.064642, 1.070597] |
| pow_lowp | 1.058845 | [1.054604, 1.063480] |
| exp2_midp | 1.194328 | [1.191663, 1.196972] |
| exp_midp | 1.217332 | [1.214774, 1.219837] |
| pow_midp | 1.157436 | [1.154507, 1.160276] |
| sigmoid_midp | 1.193225 | [1.189623, 1.196772] |
| silu_midp | 1.214437 | [1.208486, 1.219944] |

The matching NEON-token x4/64/ordinary stream cells pass. That does not cover
all cells: exp2_lowp/x8/NEON/64 stream is inconclusive, CI
[1.001444, 1.020480]. Exp2_lowp/x16/NEON/64 stream passes while being slower:
R=1.014061, CI [1.009499, 1.018636]. Per-cell results remain authoritative.

The combined workload also fails for scalar x4/64 ordinary streams:
R=1.149418, CI [1.145333, 1.153668]. Scalar x16 in that workload passes,
R=0.997774, CI [0.994839, 1.000703]; matching NEON x4 passes at
R=1.003176, CI [0.999472, 1.007744]. None offsets the scalar x4 failure.

The core study completed ten canonical operations, 96 cells each, before
admission became invalid during log10_lowp; the supervisor stopped. The invalid
operation is retained and establishes neither a pass nor a regression.
Supplemental runs retain their own source/binary identity and results.

The rejected first mask prototype had a clean AMD V3 x4/64/ordinary stream
pilot R=1.162432, CI [1.157125, 1.167872]. Later busy AMD runs were unreliable.
The ordered-clamp prototype and the final core clamp are separately retained;
none of their measurements are silently substituted for another candidate.
A proposed scalar-to-NEON bridge was rejected by the existing structural
soundness checks; those checks were not weakened.

## Fused arithmetic and independent graph

Existing fused primitives already use native FMA on supported tokens, cached
WASM fusion probes and the baseline x86 f64 V3 helper. F32 scalar dispatch is
not unconditionally replaced by software or by per-call hardware detection.

The f64 std candidate preserves the full-V3 paired helper and delegates below
V3 to safe `f64::mul_add`. Rust 1.99.1's compiler-builtins selected
`fma_with_fma` under Haswell with AVX2 disabled; the executed bytes decode to
`vfmadd213sd` plus return. Independent exact-rational fixtures pass under that
model, no-FMA Nehalem, full V3 and no_std. Emulation establishes correctness
and the selected path, never speed.

The safe standard operation owns its complete hardware/software selection;
no FMA bit authorizes an all-V3 helper. A manually added secondary detector
was rejected by the unchanged structural guard. The final candidate adds no
feature detector or proof API in magetypes. It also delegates the no-FMA case
to std software, whose cost may differ from the prior vectorized array path.
Those costs and isolated first-use costs on FMA-only hardware are unmeasured:
hold this candidate independently of the measured NaN tradeoff. F32 dispatch,
no_std, i686, NEON and WASM paths retain their implementations.

The real NEON FMA study completed all 256 predeclared cells, 50 pairs each,
with every admission check clean. It compares current fused, copied exact array
software and once-per-buffer hardware controls, covering f32 1/4/8/16 and
f64 1/2/4/8, scalar/NEON callers, both lengths/shapes and four distributions.
These are measurements of existing paths and experimental call granularity;
these NEON paths are unaffected by the x86-only fallback change. For example, real NEON scalar
f32x4/64 ordinary streams measured whole-buffer hardware/current R=0.941664,
CI [0.934024, 0.949027], while array software/current R=8.039430,
CI [7.978791, 8.095745]. The corresponding f64x4 hardware/current ratio was
1.003065, CI [0.993975, 1.010995]. Those controls demonstrate why portable
fusion must not be equated with unconditional software arithmetic.

The fixed-fused graph has zero non-NaN bit or NaN-classification mismatches in
150,528 sampled lane results spanning scalar, V3, native NEON, strict WASM and
explicit fusing/nonfusing relaxed WASM. Pure fusion changes 2,004 of those 150,528 sampled lane results versus the
corrected relaxed graph, all within the sampled ordinary domains. There are 2,348 NaN-classification changes versus main in those same samples,
counted separately from fusion. This covers 14 canonical operations;
aliases share their bodies, and cbrt has no relaxed multiply-add. MPFR error
samples are independent. For scalar x4, sampled midp upper errors include
exp2 1.203032 ULP, exp 45.090223 ULP, pow 69.409022 ULP and log10 1.650535 ULP.
These are sample bounds, not full-domain maxima; activation error bounds were
not established. Lowp errors near zero-valued log results can span many ULP.

Fixed fusion also fails performance promotion in existing clean cells. Against
the corrected graph, scalar x4/64/ordinary log2_lowp R=1.058579,
CI [1.053992, 1.063363], and ln_lowp R=1.045564,
CI [1.040506, 1.050613]. Some exp/pow cells improve, but those cannot offset
failures. No default fusion policy is accepted by this PR.

## Correctness and validation

- Every positive/negative nonzero f32 NaN payload (16,777,214 patterns per
  width), widths 4/8/16: scalar, V3, V4, real NEON, strict WASM and both
  relaxed WASM engine controls classify both exp2 clamps
  as NaN. Mixed lanes and exceptional cascades are tested separately.
- Existing independent exact-rational f32/f64 fixtures now also exercise the
  named fused operations at every generic width and available one-lane APIs.
  NaN payloads are unspecified; all other bits, including zero signs, match.
- Baseline/V3, no-FMA, FMA-without-V3, native NEON, i686, no_std, strict WASM
  and explicitly probed relaxed fusing/nonfusing controls pass focused checks.
  Emulation is used only for correctness.
- Full `just ci` at `22f5b2dd`: all 19 steps passed, including generation,
  registry/tokens, structural soundness, Miri, parity, ARM, WASM, docs and
  consumer codegen. Both package semver checks passed. Later harness/doc
  verification is recorded alongside its final identity. Report head 2d3d1e49
  passed all 19 steps again (248s, peak RSS 0.74GiB), and every GitHub CI job
  passed at that head, including Windows ARM, macOS ARM and i686 cross.

## Remaining timing coverage

Native NEON completed 13/14 canonical operations plus combined, 1,344/1,440
cells admitted. The original 96 log10_lowp cells remain invalid. All 256 native
FMA cells are admitted. There are no clean current AMD/Intel comparisons;
common admission checks refused, so the new std f64 fallback's hardware and
no-FMA costs remain unknown.

Of 784 steady-state cells per WASM mode, strict admitted 112 (64 FMA and 48
exp2_lowp); exp_lowp 48 became unreliable and stopped that mode. Fusing admitted
64 FMA cells; exp2_lowp 48 became unreliable and stopped that mode. Remaining
strict 624 and fusing 672 cells were not attempted. No quiet nonfusing timing
is available. These fractions prevent an unqualified promotion claim.

The strict first-use result retains all 200 pairs and passes the exceptional
candidate's gate. The fusing 200-pair population retains one dirty gate record
and is INADMISSIBLE, even though the suite's unreliable flag is false. No round
was removed and no second extension is permitted. Isolated native first-call
latency is also unproven; fresh-process counters include startup.

## Cold build and resource diagnostics

Three alternating fresh linked-consumer builds per source, same lockfile and
compiler, candidate `22f5b2dd`; the worker was shared. No runtime speed gate is
inferred from build times.

| Pair | Main seconds / bytes | Candidate seconds / bytes |
|---|---:|---:|
| 0 | 4.66 / 470080 | 4.62 / 471392 |
| 1 | 4.71 / 470080 | 4.71 / 471392 |
| 2 | 4.63 / 470080 | 4.62 / 471392 |

Maximum RSS in KiB, main: 353796/353812/354240; candidate:
354688/354688/354792. Size includes the complete linked consumer.

`run-heavy` resource lines: final CI+semver 267s, peak RSS 0.74GiB,
min available 19836MiB, peak load 6.10; build+all-cell analysis+cold builds 153s,
peak RSS1.37GiB, min available 19836MiB, peak load 6.10; NEON FMA 323s,
peak RSS 0.03GiB, min available 14880MiB, peak load 0.86; combined workload 68s,
peak RSS 0.01GiB, min available 14886MiB, peak load 0.46. Caps were explicit;
no other workload was interrupted.

Fresh native-process counters include startup and therefore are not primitive
latency: three-repetition instruction counts were empty2532533, f32 2532542,
f64 2533272 on the counted Intel atom PMU. Core-PMU events were not counted.
The strict first-buffer cell pooled all 200 preregistered pairs (28+172):
exceptional/main R=0.985225, CI [0.975316, 0.995332], PASS;
fixed-fused/corrected R=6.405867, CI [6.347259, 6.465085], FAIL.
It measures the first complete 64-element exp2_midp x4 buffer, including
the actual first fused-cache probe where applicable. It is not an isolated
FMA-instruction latency. The per-call WASM first-buffer experiment is separate. Isolated native first-call
latency was not established by these process counters.

## Reproduction and retained evidence

Use the project `portability-*` just recipes on an admitted worker under
`run-heavy`. The standalone Cargo.lock freezes benchmark dependencies; no
benchmark dependency enters a released crate. Every attempt is retained,
including failed builds/probes and unreliable runs.

Raw retained rounds, commands, source archives, binary hashes, all logs and
resource lines live under
`/mnt/v/output/archmage/math-portability-fma-2026-11-07/`, mirrored to the same
project path on durable storage. [Artifact index](http://dev.lan:3300/archmage/math-portability-fma-2026-11-07/)
includes `arm-results/core-study`, `final-source/core-cells.tsv`,
`arm-results/final-fma.json`, `cold-final`, independent oracle dumps and
per-control WASM directories. Large generated dumps and binaries stay outside git.

The initial FIFO transport failed because atomic save replaced the FIFO. Two
completed native packets (combined and log2_midp) were recovered intact from the
worker; their original transport exit markers are retained. Their full sample
packets and successful run-heavy logs establish admission. Later transport uses
an explicit stdout packet and retains the complete SSH stdout/stderr log.

Core native study binary SHA256:
`2913dbda401e74f52a113515fa3b7552b864620f93327e5535134f9be6c86c73`.
FMA/combined ARM host binary SHA256:
`624ca752f2d911f61e37209a8004fc4e0b5cb375f13627d7db79419ff82c96dd`.

[PR128](https://github.com/imazen/archmage/pull/128) is historical preliminary
data for an earlier implementation. Its reported costs are not measurements of
this candidate. The current report uses only the retained runs described here.


Normative context: [WebAssembly relaxed operations](https://webassembly.github.io/spec/core/exec/numerics.html#relaxed-operations)
fix the implementation's relaxed choice during a program execution, while NaN
encodings can remain nondeterministic. The deterministic profile selects
unfused multiply-add. Engine controls must be probed: Wasmtime40.0.1 still
reported fused projections with its deterministic flag in this experiment;
the nonfusing correctness control instead explicitly disabled x86 FMA.
[NaN canonicalization](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#method.cranelift_nan_canonicalization)
is a separate setting. Correctness runs use Wasmtime40.0.1; the benchmark host
locks Wasmtime49.0.5 and verifies its actual f32/f64 projection before timing.
