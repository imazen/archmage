# Implemented powf policy permutations

The six presets and all 64 supported const configurations are implemented in
the held draft. Missing evidence comes first: **there is no qualified new timing
matrix for these 64 configurations**. Every cell in the three M6 runs is flagged;
macOS placement is exploratory. The earlier Intel confined pilot times out behind another
zenbench owner without collecting a sample. A final 71 continuation after
that owner exits collects 50 pairs with enforced performance-core placement;
all 50 are dirty and unreliable. It stops before any full matrix. No promotion follows these timings.
Existing power methods, default BitExact MIDP graph, dependencies, numerical
fixtures and assembly expectations are unchanged.

[The contract](../../docs/POWF.md) defines the builder and each preset's range,
validation and dispatch. [The frozen protocol](POLICY-PLAN.md) defines the
comparisons before timing. This report supersedes the earlier documents' claim
that LOWP, BackendDependent and HIGHP flags are only proposed.

## What each axis buys

| Choice | Verified saved work | Cost or intentional tie |
|---|---|---|
| Const flags versus opaque runtime options | Policy decoding disappears before Horner evaluation | A different complete call shape can still lose in dependency chains |
| AssumeValid versus Checked | Numerical input/output rejection checks disappear | Caller must establish the same numerical preconditions; no validation on invalid inputs |
| Gamma versus NonNegativeNormal | Computed-result rejection is omitted | Narrower explicit input bounds; with AssumeValid the range checks are already absent |
| LOWP versus MIDP | Seven rather than ten native Horner FMAs | Larger measured error, including about 0.557% relative Gamma error |
| BackendDependent versus BitExact | Scalar caller and strict/nonfusing WASM avoid software fused correction | Bits and boundary rejection can differ; native FMA policies intentionally tie |
| Caller versus Automatic | Autonomous feature selection is removed | Retaining software arithmetic can cost much more than upgrading the whole operation |
| Full HIGHP versus restricted HIGHP | Numerical domain checks disappear | Std precision and platform special values; all Full validation/execution choices intentionally tie |

Static normalized instruction counts exclude unresolved/external callees.
For x4 V3: checked NonNegativeNormal MIDP has 127 instructions, checked Gamma
117, assumed Gamma 63, and assumed Gamma LOWP 55. Native Horner FMA counts
are ten and seven. M6 NEON corresponding counts are 162,145,83,73.
For scalar Caller Gamma, BitExact versus BackendDependent expands to 2,106
versus 420 x86 instructions and 1,621 versus 504 M6  instructions. These are
codegen evidence, not measured hardware execution counts or timing ratios.

All 384 policy bodies per target are retained, including compiler aliases.
Native LOWP must have fewer FMAs; scalar BackendDependent Caller must have
no FMA/software-fusion/probe calls. An additional 22 probes retain the public
array/std/local-reference shapes. On M6, Full HIGHP public arrays and direct
std have identical instruction counts at widths 1/4/8/16: 11/32/57/112.

The original public HIGHP x4 native wrapper emitted 58  instructions. Removing
Result::map reduced it to 48; separating validation from direct vector
construction reduced it to 38. The scalar/array route is 32. The locally copied
Result-map provider already optimizes to 38 in its new module; its timing must
not be described as the original public-method baseline. See the protocol.

## Exactness and accuracy

All 48 bounded configurations and their runtime adapters match the unchanged
independent four-graph rational/RNE fixtures. There are 27,810 domain-tagged
pairs, every broadcast lane at widths 4/8/16, and mixed vectors with exact
whole-vector rejection masks. Each execution context checks 83,430 broadcast
vectors and 13,498 mixed vectors; AssumeValid executes only admitted inputs.

Zero mismatches in 14 contexts: Intel scalar/V3; M6 scalar/NEON; i686 scalar;
Nehalem no-FMA emulation; scalar/native strict WASM; scalar/native relaxed WASM
with both independently probed fusing and nonfusing engines; V4/V4x under SDE.
Emulation establishes correctness, not physical-machine performance.

All 16 HIGHP configurations, widths 1/4/8/16, public vector/array and const/runtime
routes agree with the same platform's runtime std provider on 27,909 cases on
Intel and M6. NaN payloads are unspecified; non-NaN bits, classes, masks and
rejection stages are checked. Directed MPFR-256 finds sampled maximum ULP error
upper bounds 0.506155662238598 on Linux x86 and 0.5116611998528243 on Apple ARM;
17 and 121 results respectively differ from correctly rounded power.
These finite samples do not establish a global ULP guarantee. The same 16-policy
vector/array equivalence replay also passes under no-FMA Nehalem emulation,
i686, and scalar/native strict and both relaxed-WASM engine controls.

Full HIGHP inherits a measured std special-value difference: signaling NaN
bits 7f800001 raised to either signed zero yields NaN on Linux x86 and 1 on
Apple ARM. Two unique input pairs repeat across four corpus domain tags.
Restricted Checked configurations reject the NaN base. HIGHP is therefore
BackendDependent; it is neither a canonical-special-values nor ONE_ULP promise.

The [earlier MPFR/search comparison](NONFUSED.md) measures all four polynomial
graphs. Sampled Gamma maximum ULP errors are47.90180805 fused MIDP,
48.56180874 separate MIDP,93408.85148075 fused LOWP and93411.46518661 separate
LOWP. Unbounded exponents amplify LOWP log bias near 1 catastrophically.
PositiveNormal LOWP is an available range choice, not an accuracy preset.
No prior accuracy threshold is loosened.

## Retained exploratory timings

Three complete M6 runs each contain 56 workloads and 50 interleaved rounds per
workload. The first two use 69 implementations. Gate identity tracing in the
second identifies Storage activity. The owner then closes Storage and requests
one comparison of the updated implementation; it has 71 arms, including the
local Result-map provider and direct public HIGHP arrays. Other OS services
remain busy, so **all 56 cells are flagged in every run**. Every round remains.
No additional favorable retry is selected. The final Intel native x4/64
stream pilot (literal 2.4) collects 50 dirty pairs, unreliable=true, with
cpuset 0-7 enforced. No full matrix follows it. Its resource wrapper reports
8 s, peak RSS 0.03 GiB, minimum available 23,135 MiB, load 3.41.

Final binary SHA-256:
`5c60abb900e8c09684db89ac043ba5902198e61064ad8217eb772ec5dcfcacdc`.
Actual formatted worker source 6e69588c is frozen before this run, with source
archive, Cargo.lock, compiler and executable. Earlier 69-arm runs retain their
actual formatted source archives because their embedded git identity refers
to the worker's prior parent; that field alone is not their source identity.

Selected final cell: M6 NEON x4, 4,096 values, stream, literal f32 exponent 2.4.
These are the original zenbench bar lengths with complete names and means
converted to ns per buffer. Checked/AssumeValid labels are explicit.

```text
legacy pow_lowp                          ████                    1562.19 ns
BitExact LOWP Gamma AssumeValid Auto     █████                   1875.48 ns
BitExact LOWP Gamma Checked Auto         █████                   2186.13 ns
BitExact MIDP Gamma AssumeValid Auto     ██████                  2253.72 ns
BitExact MIDP Gamma Checked Auto         ██████                  2532.94 ns
legacy pow_midp                          ███████                 2763.67 ns
BitExact MIDP NonNegative Checked Auto   ███████                 2901.32 ns
runtime BitExact MIDP Gamma Checked Auto █████████               3530.22 ns
std f32 powf                            ████████████████████    8049.00 ns
direct array HIGHP Full                  ████████████████████    8053.17 ns
Backend HIGHP Full Checked Auto vector   █████████████████████   8408.14 ns
```

All-retained paired MeanCi uses 100,000 resamples. Descriptive 95% ratio bounds:

| Numerator / denominator | Paired 95% interval |
|---|---|
| Checked Gamma MIDP / checked NonNegativeNormal MIDP | 0.871630–0.874519 |
| Assumed Gamma MIDP / checked Gamma MIDP | 0.887942–0.891740 |
| Checked Gamma LOWP / checked Gamma MIDP | 0.861322–0.864902 |
| Const checked Gamma MIDP / opaque runtime same options | 0.716381–0.718673 |
| Checked Gamma MIDP / legacy MIDP | 0.914825–0.918244 |
| Checked Gamma LOWP / legacy LOWP | 1.396575–1.402394 |
| Direct array Full HIGHP / direct std f32 | 0.998986–1.002176 |
| Native vector Full HIGHP / direct std f32 | 1.040675–1.051065 |

These intervals do not repair interference or unverified placement. The native
vector HIGHP overhead and checked LOWP cost remain tradeoffs, not hidden wins.
Across the 56 final workloads, array HIGHP/std is descriptively within the2%
bound in 56/56. Vector HIGHP/std is within in 40, above in 14, inconclusive in 2.
Checked Gamma MIDP/legacy MIDP is within in 56/56; checked Gamma LOWP/legacy
LOWP is within in 32 and above in 24. Const/runtime is within in 48 and above in 8;
all 8 reversals are scalar x4/x8 dependency chains. Shorter LOWP reverses in 4
of 1,344 predeclared LOWP/MIDP edges, Full/Gamma HIGHP in 2 of 224, and
AssumeValid/Checked in 38 of 1,792. Caller/Automatic is above in 768 of 1,792,
consistent with retaining expensive exact software FMA. The full paired record
keeps all edges, ties and reversals. **Zero cells are promotion-eligible.**

The cost rule is unchanged: qualified upper 95% <=1.02 passes, lower >1.02
fails, overlap is inconclusive; an optimization needs upper <1.00. Missing
admitted physical x86, ARM and WASM timings remain missing. New exceptional-
lane, bit-distributed-input and first-use timing coverage is also missing. Fresh-target
compile time and linked size for representative const and runtime-option clients
are now recorded in the [three-repetition diagnostic](../powf_providers_2026-11-09/COLD.md).
First-use runtime cost remains unmeasured. The earlier cold-build diagnostics
describe the earlier default-only API.

## Debug runtime stack correction

GitHub CI and local reproduction expose WASM linear-stack exhaustion in the
existing NaN/subnormal payload test. The 64-arm runtime decoder forces all
graphs into each unoptimized caller. The payload-test frame is 1,068,352 bytes before correction and 874,288
afterward. Retaining separate decoder and selected
graph frames in debug builds fixes the test with the original stack limits,
payload stride, widths, masks and assertions. Release builds with debug assertions disabled retain the
original always-inline attributes. All 406 expanded release probe bodies
match before/after on x86, strict WASM and relaxed WASM, after normalizing
only LLVM unique private-symbol suffixes. Symbol bases, operands, registers
and immediates remain significant; the existing assembly gate is unchanged.
Thus the frozen release comparisons remain applicable. The focused WASM replay passes all 13 powf tests after correction
(15 s, peak RSS 1.70 GiB), versus the reproduced trap before correction
(35 s, peak RSS 6.05 GiB). These are build/test resource records, not runtime
speed comparisons. The final rerun passes all 19 CI stages, including Miri,
then the full WASM debug crate tests and doctests: 766 s, peak RSS 4.72 GiB,
minimum available 18,869 MiB, peak load 6.87. Original failing module and corrected module are preserved
in a separate addendum; the sealed core evidence remains unchanged.

## Validation and retained evidence

Full 19-stage local CI passes, including Miri and soundness. A new HIGHP test
initially assumed exact std(-2)^3; Miri demonstrates std's unspecified precision.
Only that newly introduced test changes to its declared sign/classification
contract. Pre-existing numerical and assembly expectations are preserved.
All-target/all-feature study Clippy passes after re-downloading an incomplete
external dependency into a task-owned Cargo home and simplifying generated
x1 arithmetic. The failed shared-cache build is retained untouched.

Resource records: final M6 timing 502 s, peak process-tree RSS 0.03 GiB,
minimum free/inactive/speculative proxy 10,527 MiB, peak load 3.00; analysis 37 s,
peak 0.04 GiB. M6's monitor is not a cgroup hard cap. Native HIGHP verification
2 s, peak 0.04 GiB. Intel native HIGHP plus V4/V4x validation 128 s, peak 4.02 GiB,
minimum available 20,108 MiB, load 3.20. The full CI wrapper 812 s reaches
peak 1.97 GiB, minimum available 22,024 MiB, load 4.42; its 19 CI stages pass,
then a subsequent study lint fails on the incomplete dependency. The final isolated-Cargo-home pipeline passes study Clippy, all HIGHP
cross-provider checks, codegen emission and both semver/target-feature gates:
437 s, peak RSS 5.27 GiB, minimum available 18,743 MiB, peak load 3.26.
The unchanged legacy assembly gate passes 1,623/794/794 probes with zero
reviewed differences: 23 s, peak RSS 0.44 GiB, minimum available 20,154 MiB,
peak load 3.11. Its baseline allowances remain unchanged.

Raw rounds, gate traces, original and corrected failures, source/binary/lock
identities, compiler settings, MPFR actuals and inputs, assembly and all checks
live at `/mnt/v/output/archmage/powf-policy-permutations-2026-11-09/`, with
a durable mirror at `/mnt/tower/output/archmage/powf-policy-permutations-2026-11-09/`.
Generated data and reference executables stay outside git. Reproduce using
`powf-policy-check`, `powf-policy-cross`, `powf-policy-avx512-check`,
`powf-highp-accuracy`, `powf-highp-cross`, `powf-policy-codegen-audit`, and
`powf-policy-matrix-analyze`. Draft remains held; no merge or release.
