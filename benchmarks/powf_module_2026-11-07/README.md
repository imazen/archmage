# Bounded powf: portability and cost study

Promotion is held. The checked API fails the 2% cost gate in matching native
callers. Complete quiet AMD/Intel matrices, quiet WASM timings, native first-use
cost and physical below-V3/std fallback timings are missing. The new API is
additive; existing power methods are unchanged.

The production implementation is `56059d2c`, based on main `6bfd83d7`.
[The contract](../../docs/POWF.md) defines accepted inputs, exact graph identity,
whole-vector rejection, caller preservation and runtime whole-operation FMA
selection. `std::powf` is not an admitted backend. `std::mul_add` can supply the
specified FMA operation after a cached independent bug screen.

## Accuracy and exactness

Actual library outputs were checked against an independent rational/RNE graph
mirror and MPFR 4.2.2 at 256 bits, using directed reference enclosures. There
were 26,823 domain-tagged cases: 17,735 accepted and 9,088 rejected. All result
bits and rejection stages matched the independent graph. These are sampled
binary-function cases, not exhaustive coverage of power.

| Profile | Accepted samples | Maximum measured ULP error upper bound |
|---|---:|---:|
| PositiveNormal | 5,576 | 91.258 |
| NonNegativeNormal | 5,614 | 92.613 |
| Gamma | 6,545 | 25.397 |

The table rounds enclosure upper bounds upward. It does not establish a
whole-domain accuracy bound. On this std/compiler build, `std::powf` differed
from the canonical graph in 9,848 of the 17,735 accepted cases; this compares
contracts rather than judging std against mathematical truth.

The rational unit corpus has 2,247 boundary/structured cases. Mixed accepted
lanes, positive-zero handling and rejection masks are checked separately at
every logical width. Each NaN family and each nonzero-subnormal family has
16,777,214 sign/payload patterns. Full sweeps exercise bases and uniform exponents in all three profiles at
widths 4/8/16. Inputs pass an optimizer barrier so classification executes at
runtime. Explicit tier assertions prevent silent scalar fallback.

The Gamma range-check omission has a rational/RNE interval certificate over
its entire admitted input domain: the magnitude of the computed exp2 argument
is below 68. That certificate concerns range, not mathematical power accuracy.

## Clean paired measurements

[The frozen matrix](MATRIX.md) uses 50 retained interleaved rounds per cell.
The analyzer bootstraps every paired candidate/control ratio with zenbench
`MeanCi`, 100,000 resamples, and no round filtering. PASS means upper 95% CI
<=1.02; FAIL means lower >1.02; otherwise INCONCLUSIVE. Every reported clean
cell has all resource flags true and `unreliable=false`.

The native control is the same graph with validation removed. These ratios
measure the total checked API boundary, validation and dispatch cost, including
lost optimization across a call boundary. They are not comparisons with legacy
`pow_midp`, nor isolated mask-branch costs. The runner records iterate/exploratory
mode; the source, matrix and 50-round plan were fixed for these paired studies.
No confirmed baseline promotion or general optimization claim follows.

Neoverse-N1 completed all 168 ordinary cells, including both buffer lengths,
streams/chains, all profiles, both input distributions and all caller/width
pairs. Its 336 comparisons are:

| Candidate versus prevalidated graph | PASS | FAIL | INCONCLUSIVE |
|---|---:|---:|---:|
| Checked Caller | 28 | 134 | 6 |
| Checked Automatic | 96 | 72 | 0 |

All 96 scalar-context Automatic cells pass; all 72 matching-NEON Automatic
cells fail. Automatic can accelerate the complete software graph while keeping
canonical bits. That gain does not hide native-caller costs.

Selected ordinary Gamma streams, exponent equal to f32 2.4:

| CPU / caller / width / elements | Candidate | Paired ratio | 95% CI | Gate |
|---|---|---:|---|---|
| Neoverse-N1 / NEON / x4 / 64 | Caller | 1.37644 | 1.36650–1.38245 | FAIL |
| Neoverse-N1 / NEON / x8 / 4096 | Caller | 1.20208 | 1.19634–1.20619 | FAIL |
| Neoverse-N1 / NEON / x16 / 4096 | Caller | 1.42472 | 1.42102–1.42813 | FAIL |
| Neoverse-N1 / Scalar / x4 / 64 | Automatic | 0.16233 | 0.16082–0.16408 | PASS |
| Neoverse-N1 / Scalar / x16 / 4096 | Automatic | 0.13324 | 0.13280–0.13370 | PASS |
| Zen 4 / V3 / x4 / 64 | Caller | 1.25275 | 1.24278–1.26602 | FAIL |
| Zen 4 / V3 / x4 / 64 | Automatic | 1.21554 | 1.20868–1.22171 | FAIL |

The Zen 4 rows are the preregistered clean pilot. Its full-matrix attempt
stopped after the first cell was unreliable. Intel placement refused admission:
the available resource scope did not provide an effective cpuset boundary.
Thread affinity did not satisfy the harness's process-placement requirement.
Neither failure supplies a speed claim.

All 48 Neoverse-N1 rejection cells retained clean rounds. Automatic versus
Caller passes 44 cells, with four inconclusive and none failing. These compare
two complete checked APIs; an unchecked invalid-arithmetic graph is not used
as a rejection control.

The strict WASM pilot retained 50 flagged rounds and was unreliable after
waiting behind another admitted benchmark. Its timings are inadmissible;
remaining WASM cells and first-use timings are unmeasured. Strict/fusing/
nonfusing numerical tests pass with asserted runtime projections. The Wasmtime 49.0.5 host verifies all 168 ordinary cells in each of the three
engine controls with projections 0/1/0; correctness CLI runs use Wasmtime 40.0.1.
The nonfusing host control selects the entire baseline x86-64 engine ISA, so
future differences cannot be attributed solely to FMA.

Earlier runtime-domain and blanket-inline prototype runs are retained separately.
Const domain specialization at the feature boundary reduces the measured native
cost, but the final candidate still fails promotion. No failed or dirty run is
removed, and no clean rerun was selected to replace an unfavorable measurement.

## Cold build diagnostics

Rust 1.99.1, distribution target settings, one codegen unit, thin LTO, fresh
target directories, paired order, three repetitions. Dependencies were cached;
paths were remapped. The host was shared, so these are raw build diagnostics,
not a precise build-speed regression claim.

| Consumer | Build seconds, all repetitions | Executable bytes, every repetition |
|---|---|---:|
| Main, legacy pow | 4.390363 / 4.495764 / 4.434248 | 406,160 |
| Candidate, same legacy source | 4.448116 / 4.487319 / 4.529940 | 406,152 |
| Candidate, new bounded powf | 4.599275 / 4.589518 / 4.553297 | 421,872 |

The third row changes the called API and its rounding/dispatch contract; it
is not a same-consumer patch-size comparison. Per-build GNU time RSS logs,
compiler identity, source archives and executable hashes are retained.

## Reproduce and review

Run builds, tests and analysis on an eligible worker through the resource
wrapper; run timing only through zenbench's common admission and strict gates.
The justfile supplies `powf-contract`, `powf-payloads`, `powf-wasm-build`,
`powf-wasm-contract`, `powf-accuracy`, `powf-study-build`, `powf-study-matrix`,
`powf-study-analyze`, `powf-study-cold` and `powf-study-lint`.

`prepare.py` mechanically extracts the production graph and existing exact
array software FMA. It verifies the two validation sites before removing them.
WASM keeps the existing fused primitive at each control node, to compare
per-node cached probing with whole-operation selection. Every selected arm's
lane bits are checked before timing. No std power output supplies graph truth.

Review found no changes to existing arithmetic contracts, coefficient generators
or soundness/codegen expectations. Generator and vector delegates change together.
The implementation has no new library dependency or handwritten unsafe code.
Package/public-API checks, no_std, i686, emulated no-FMA/FMA-only, real native tiers
and WASM controls remain required; see the recorded validation logs.

Raw rounds, failed admission attempts, MPFR inputs/results, compiler flags,
source/reference hashes and resource logs live under:
`/mnt/v/output/archmage/powf-module-2026-11-07/`, with the durable mirror under `/mnt/tower/output/archmage/powf-module-2026-11-07/`.

The analysis hashes are:

- ordinary N1: `5a76cbf4e3aca9ec564dc944fe115938fd2823a0dd12cdfd44ba8c19db82d2d0`
- rejection N1: `8aae3c69ddc779853c81c70890b0fcf2219ec7c414b750aedb67b3ebceba0e11`

Resource summaries: full CI plus both semver checks, 379 s, peak RSS 1.00 GiB,
minimum available 23,181 MiB, peak load 4.61; N1 ordinary matrix, 148 s, peak
RSS 0.03 GiB, minimum available 14,899 MiB, peak load 0.64; cold study, 41 s,
peak RSS 0.35 GiB, minimum available 24,213 MiB, peak load 2.20. Full stdout and
stderr preserve the remaining wrapper lines. These values describe those runs.

CI follow-up: the full cross-target warning sweep passes after using fixed-size
chunks in the WASM adapter, implicit Any coercions and private probe visibility.
No fixture values changed. The Nehalem job uses Ubuntu 24.04 because Ubuntu
26.04 cloud images default to an amd64v3 userland; the failing runner executed
AVX in its dynamic loader before the test binary started. SDE 10.8 passes the
root and bounded-power unit suites in the baseline Ubuntu 24.04 container.
All CPU instruction checks and test assertions remain active. See
[Ubuntu's image policy](https://ubuntu.com/cloud/public-cloud/docs/all-clouds-explanation/architecture-variants/).
