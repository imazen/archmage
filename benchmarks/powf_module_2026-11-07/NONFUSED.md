# Backend-dependent LOWP/MIDP rounding, 2026-11-08

Separate multiply/add is a viable numerical choice for a backend-dependent
legacy quality family on the bounded Gamma domain. It cannot implement the
BitExact contract. Use native FMA when available: the Apple M6 experiment
finds separate arithmetic slower in every resource-clean NEON cell.
If both rounding graphs satisfy a selected BackendDependent contract,
`mul_add_relaxed` can express that choice directly: strict WASM separates,
relaxed WASM follows its engine, and native FMA-capable tiers fuse. A fusion
correction probe is needed for BitExact, not simply to admit either graph.
Whole-operation dispatch is still an independent policy decision.
No production graph, existing fixture, accuracy threshold or default changes.
This historical study left LOWP/BackendDependent/HIGHP options proposed. The
later [implemented policy comparison](POLICIES.md) adds those public draft
configurations, preserving this study's frozen graphs and timings. The
[preset specification](../../docs/POWF.md#quality-and-portability-presets) states
range, validation and execution explicitly.

Missing evidence: this study has no qualified full AMD, Intel, Neoverse-N1
or WASM timing matrix. M6 timings use owner-approved unrestricted OS scheduling
and cannot satisfy strict promotion. Intel's native pilot is flagged.

## Numerical comparison

`rounding_prepare.py` extracts the owning LOWP and MIDP expressions without
changing coefficient bits. A const selects either exact `mul_add_fused` or
ordinary RNE multiplication followed by addition at each Horner node. Other
operations, inputs, zero rules and validation are shared. `rounding_oracle.py`
models each graph independently with exact rational IEEE operations; it does
not obtain expected bits from Rust or std power.

The preserved 26,857 domain-tagged inputs, deterministic reduction boundaries
and 24 pinned external adaptive searches give 27,762 pairs. Each of four
graphs matches its rational oracle exactly, including rejections. Each forced
rounding mode agrees between scalar and native V3; actual scalar/NEON outputs
replay the same four graphs on M6. All positive accepted ordinary results
compared with the corresponding legacy method agree: separate with legacy
scalar, fused with legacy FMA-native. Zero and saturation semantics are
intentionally reported separately, rather than claimed equivalent.

The complete 27,762-pair replay also passes with no graph mismatches under
Nehalem no-FMA emulation, strict WASM SIMD, and relaxed WASM with both
fusing and nonfusing providers. WASM uses Wasmtime 40.0.1 with explicit
Cranelift `has_fma` selection; a cancellation probe verifies the actual
provider before each replay. These executions establish correctness coverage,
not emulated or WASM speed. All 24 selected scalar/native assembly probes
on x86, M6, strict WASM and relaxed WASM are inspected. Every separate graph
has multiplication/addition and zero FMA instructions; native x86/NEON
fused probes contain FMA instructions. No assembly-equality gate is changed.

The final width audit emits expected values directly from the independent
rational oracle, then checks every lane at x4/x8/x16, including whole-vector
rejection precedence and exact masks on mixed inputs sharing one exponent.
Forty-eight additional zero/large-exponent cases expand that corpus to 27,810.
They first expose eight mismatches in the benchmark-only LOWP normal profile:
admitted +0 with n=2^30 or f32::MAX is incorrectly rejected by its result-range
guard because LOWP log2(1) has a small bias. Excluding admitted zero lanes from
that guard restores the already-declared +0 result. Expected oracle bits are
unchanged. The Gamma guard is omitted already, so frozen Gamma timings and
production math are unaffected. Original failures remain in the addendum.

After correction, every lane matches the rational expectations at all three
widths on scalar/V3, scalar/NEON, no-FMA Nehalem, strict WASM and both probed
relaxed-WASM providers, plus scalar i686. Each context checks 83,430 broadcast vectors and
13,498 mixed vectors against all four graphs. Before/after input files and
independent expected-bit fixtures are byte-identical. The new coverage does
not turn the sampled mathematical-error maxima into global accuracy bounds.
Validation pipelines report peak RSS 0.25 GiB on x86, 0.28 GiB on M6,
0.18 GiB under Nehalem emulation and 0.35 GiB for both WASM builds/replays.

Directed MPFR-256 supplies mathematical truth. Its binade is resolved before
conversion to double so intervals for `1-epsilon` do not select the wrong ULP.
These are sampled maxima, including adaptive witnesses, not global guarantees.

| Gamma graph | Largest measured ULP error | Largest measured relative error |
|---|---:|---:|
| MIDP separate | 48.56180874 | 0.000292088% |
| MIDP fused | 47.90180805 | 0.000289836% |
| LOWP separate | 93411.46518661 | 0.556775805% |
| LOWP fused | 93408.85148075 | 0.556760180% |

Gamma admits +0 or x in [2^-16,2^16], and normal or signed-zero n in [0,4].
Whole-vector checks reject negative zero bases, NaNs, infinities, nonzero
subnormals and out-of-domain inputs. Admitted 0^0 is 1; 0^positive is +0.
Independent RNE interval certificates bound the computed argument magnitude
by 66.85743713378906 for MIDP and 67.52277374267578 for LOWP, for both rounding
modes. Thus Gamma can omit the output-range branch. Mixed-lane masks, invalid
uniform exponents, zero rails and checked/assumed equality pass at widths
4/8/16 in scalar, V3 and NEON contexts.

Of 6,641 accepted Gamma pairs, separate versus fused changes 625 MIDP outputs
and 1,020 LOWP outputs. Outside Gamma, MIDP also changes accept/reject decisions
at two boundary inputs in each normal-only domain. BackendDependent therefore
permits both numerical bit variation and rounding-sensitive range rejection;
it cannot promise the BitExact accepted set near the range boundary.

Unrestricted exponents are especially unsuitable for LOWP. At base bits
`3f7fffff`, exponent bits `cb317217`, the true result is approximately
1.99999992591481, while both LOWP graphs return bits `4ad3af2c`, a relative
error exceeding 3,468,242. Its biased logarithm near 1 is magnified by the
exponent. A computed normal result does not establish accuracy. Gamma or a
separately validated fixed-exponent band must be part of a LOWP preset.

The existing legacy accuracy test uses 15% relative error for LOWP and
128 ULP plus 1e-5 relative error for MIDP on its fixed set of exponents and
inputs. Those existing tests and thresholds remain unchanged. This study's
Gamma measurements fall inside those numerical budgets, but neither those
tests nor this sampled search proves the budgets over every Gamma pair.

## Equal-work timings

The [frozen plan](NONFUSED-PLAN.md) has 96 cells: x4/x8/x16, scalar/native,
64/4096 values, stream/normalized chain, literal n=2.4/0.5 and
checked/assumed Gamma. Every cell contains the four controlled graphs,
actual legacy LOWP/MIDP and per-lane std f32 power. Pre-timing verification
checks every controlled output is finite and checked/assumed bits agree.
No fast-math flags or `target-cpu=native` are used.

M6 completes all 96 cells with 50 interleaved retained rounds each: 64
resource-clean, 32 flagged. The following original zenbench bar lengths are
shown with unrounded-result means converted to ns per 4,096-value buffer.
Cell: x4/NEON, stream, Gamma, literal exponent 2.4, checked validation.
This cell is resource-clean; placement remains exploratory.

```text
legacy pow_lowp  ██████████                          1531.50 ns
LOWP fused       ██████████████                      2146.40 ns
MIDP fused       ████████████████                    2482.36 ns
LOWP separate    █████████████████                   2589.18 ns
legacy pow_midp  ██████████████████                  2708.23 ns
MIDP separate    ██████████████████████              3264.27 ns
std f32 powf     █████████████████████████████████████████████████████ 7985.94 ns
```

All-round paired MeanCi uses 100,000 resamples. MIDP separate/fused ratio is
1.315402 [1.305909,1.325252]; LOWP separate/fused is
1.206291 [1.196408,1.218655]. Both exceed the unchanged 1.02 cost bound.
All 33 clean NEON cells exceed that bound for both families. Among 31 clean
scalar-context cells, MIDP separate/fused has 6 within-bound, 13 above-bound
and 12 inconclusive; LOWP has 23 above-bound and 8 inconclusive. All are
exploratory; none is promoted by those descriptive classifications.

The Intel invocation first omitted the pilot selector and recorded the first
two scalar matrix cells before a resource gate stopped it. Preserve that
protocol mistake and both cells; do not treat them as a valid pilot or retry
them. The correctly selected native pilot subsequently retained 50 pairs with
enforced performance-core placement, but is flagged and unreliable. It blocks
the full Intel matrix. The earlier zero-sample wrapper path error is preserved
separately. No noisy result is logged as a qualified performance regression.

Resource records: M6 matrix 96s, wrapper peak RSS 0.02 GiB, minimum reclaimable
page proxy 7,118 MiB, peak load 1.70. The macOS wrapper monitors its process tree
but has no cgroup cap. Intel accuracy plus adaptive search/tests takes 59s,
peak RSS 0.48 GiB, minimum available 24,082 MiB, peak load 14.15. Final
MPFR replay/analysis has separate complete logs. Resource limits and failures
are retained with raw evidence.

## Reproduction and evidence

Use `just powf-rounding-build`, `just powf-rounding-check`, the
`powf-rounding-accuracy`, `powf-rounding-matrix`, `powf-rounding-codegen` and
corresponding audit/analysis recipes. Run heavy commands through the resource
wrapper on an eligible worker. `--rounding` adds a separate mode to the
existing matrix/search drivers; their original defaults remain unchanged.

Raw samples, exact source, reference binaries, compiler identities, generated
kernel hashes, all failures and checksums live under
`powf-nonfused-expectations-2026-11-08`, separate from the immutable earlier
replacement record. Reference binaries are retained outside Cargo targets.
The SHA manifest records the final artifact inventory and verified mirror.
