# Magetypes powf portability accuracy and performance

Updated 2026-11-09. The replacement API remains in [draft PR144](https://github.com/imazen/archmage/pull/144), with promotion held. Canonical HIGHP now has scalar and packed implementations. Legacy power methods retain their behavior. Performance observations below identify the actual implementation and machine; exploratory intervals cannot satisfy the promotion gate.

## Numerical contracts and bounds

| Precision and portability | Arithmetic | Numerical promise |
|---|---|---|
| BitExact MIDP | Fixed fused binary32 log2/exp2 graph | Identical accepted result bits; no global mathematical ULP bound |
| BitExact LOWP | Shorter fixed fused binary32 graph | Identical accepted result bits; different accuracy from MIDP |
| BackendDependent MIDP or LOWP | Native FMA where available; separate rounding otherwise | Bits and range rejection may depend on backend |
| BitExact HIGHP | Canonical Arm v26.07 TOINT=0 graph, binary64 intermediates, nine fused nodes, final binary32 conversion | Identical non-NaN result bits and NaN classification; NaN payload/sign unspecified |
| BackendDependent HIGHP | Scalar `f32::powf` from std | Std platform behavior and accuracy; requires std |

Every bit-consistency promise requires round-to-nearest and gradual underflow. The API promises result values, not matching errno or floating exception flags. HIGHP is a quality family, not a correctly rounded or ONE_ULP guarantee. Finite startup probes cannot certify arbitrary std pow inputs against the canonical graph.

| Domain | Base inputs | Uniform exponent | Checked output rule |
|---|---|---|---|
| PositiveNormal | Positive finite normal | Finite normal or signed zero | MIDP/LOWP computed exp2 argument in [-126,127.5]; HIGHP result normal |
| NonNegativeNormal | Same, plus +0 for a nonnegative exponent | Same | Same, plus admitted +0 result for positive exponent |
| Gamma | +0 or [2^-16,2^16] | Normal or signed zero in [0,4] | Normal or +0; 0^0=1; range check omitted |
| PositiveGamma | Positive normal [2^-16,2^16], excluding zero | Normal or signed zero in [0,4] | Same nonzero bits as Gamma; omits zero masks |
| Full | All f32 values | All f32 values | HIGHP only; overflow, underflow, signed zero and NaN classification follow the selected graph |

Checked rejects the entire vector with the exact lane mask at the first failing stage: width/options, exponent, base, output range. AssumeValid removes numerical checks while retaining width/option checks and memory safety. Violating an assumed domain has unspecified numerical results. Gamma's bounds justify deleting range/special-case work; checking those stricter input bounds still costs instructions.

## Const options and dispatch

Use `try_powf_with::<{ PowfOptions::PORTABLE_GAMMA_HIGHP.with_validation(PowfValidation::AssumeValid).bits() }>(x, n)`, or the matching vector delegate. The integer stays const through delegates, dispatch and kernels. There are 104 supported configurations with std and 84 without it; 64 are bounded MIDP/LOWP configurations. Runtime adapters decode to the same const graphs.

Automatic preserves a capable caller or upgrades the whole operation. Caller prevents autonomous upgrades. FMA selection and packing remain private tuning choices. BackendDependent HIGHP preserves its std call path and may inherit std's own dispatch.

## MIDP/LOWP implementation matrix

| Family | FMA-capable matching caller | ScalarCaller / nonfusing backend | Single values and tails |
|---|---|---|---|
| BitExact MIDP | Ten native fused nodes | Compile-proven FMA, otherwise exact software nodes | Same selected graph via `try_powf_with_token` |
| BitExact LOWP | Seven native fused nodes | Compile-proven FMA, otherwise exact software nodes | Same selected graph via `try_powf_with_token` |
| BackendDependent MIDP | Ten native fused nodes | Separate multiply/add | Same token required for identical bits |
| BackendDependent LOWP | Seven native fused nodes | Separate multiply/add | Same token required for identical bits |

All four families support PositiveNormal, NonNegativeNormal, Gamma and
PositiveGamma, Checked/AssumeValid, and Automatic/Caller: 64 bounded encodings.
Each graph has one division. PositiveGamma removes Gamma's zero masks; its
assumed preset also removes validation. Matching native portable/backend and
Caller/Automatic variants intentionally collapse to the same arithmetic.
Declining automatic upgrades can make portable software FMA expensive.

See the [complete bounded replacement report](../benchmarks/powf_module_2026-11-07/MIDLOW.md)
for scalar, packed, tail and fallback costs on each measured ISA.

## ISA implementations and measured coverage

| ISA or caller context | Canonical HIGHP implementation | Correctness coverage | New library timing status |
|---|---|---|---|
| Scalar x86-64 | Scalar graph; std exact mul_add or no_std software FMA; Automatic can upgrade | Independent 27,909-pair corpus, all widths | Linux pilot inadmissible; no qualified speed claim |
| x86 without V3, including FMA without AVX2 | Caller scalar graph; std may use hardware FMA through its own implementation; no_std exact software | Full corpus on emulated Nehalem and Haswell with AVX2 disabled | Not measured |
| x86 V3 AVX2 and FMA | Four f64 lanes per packed graph; scalar exceptional-pack fallback | Full corpus, required V3, mixed/broadcast lanes | Initial Zen 5 pilot retained and inadmissible |
| x86 V4 and V4x AVX-512 | Eight f64 lanes, four-lane packing for x4, scalar tail for x1 | Full corpus in explicit V4 and V4x contexts | Zen 5 paired pilot inadmissible; raw diagnostics below |
| AArch64 NEON Neoverse-N1 | Two f64 lanes per graph, native FMA | Native std/no_std fixtures, full corpus and NaN-family scan | Uniform-hardware pilot inadmissible; raw diagnostics below |
| AArch64 NEON Apple M6 | Same two-lane graph | Native std/no_std fixtures and full corpus | Two 32-cell exploratory matrices; no class confinement |
| WASM strict SIMD128 | Scalar exact fused fallback | Full corpus with and without std | Not measured; execution supplies no native timing claim |
| WASM relaxed SIMD128 | Two-lane graph only when the cached f64 probe confirms fusion; exact fallback otherwise | Full corpus with and without std, ordinary/deterministic modes and forced FMA on/off; cached probe agrees with both choices | Not measured |
| i686 | Scalar canonical graph | Full corpus in scalar and Automatic contexts | Not measured |
| ARM bare metal and Thumb | Exact no_std scalar fallback | Compilation on aarch64-unknown-none and thumbv7m-none-eabi | Runtime and performance not measured |

The corpus runs cover widths 1/4/8/16, broadcasts and mixed bases. They found zero promised bit/class mismatches. The separate exhaustive scan enumerates all 16,777,214 f32 NaN operand encodings for base^0, base^2.4 and NaN exponents over [1,0,-1,2], with 201,326,568 lane checks per context. Scalar/V3/V4x, M6 scalar/NEON, N1 scalar/NEON and both relaxed-WASM FMA choices report zero classification mismatches. This is exhaustive for those one-operand families, not the binary powf domain. NaN payload differences remain in complete variance ledgers. The exact graph oracle uses rational IEEE rounding and separately frozen C-extracted coefficients; MPFR independently measures mathematical accuracy.

## Measured mathematical accuracy

| Implementation and sample | Observed maximum error | Correctly rounded-result differences |
|---|---|---|
| New canonical HIGHP, 27,909-pair broad sample, positive finite reference subset | 0.506155662239 ULP | 17; same outputs as the prior scalar Arm graph and Linux std on this sample |
| New canonical V3 and V4x HIGHP, 515 hard witnesses | 0.500007618219 ULP | 110; identical to the prior scalar graph |
| New Full, checked Gamma and assumed Gamma HIGHP on V3/V4x, 8,698 Gamma pairs | 0.501247502864 ULP | Eight per implementation and width |
| Existing canonical MIDP, expanded Gamma sample | Below 47.902 ULP | See the [replacement study](../benchmarks/powf_module_2026-11-07/REPLACEMENTS.md) |

These are observed maxima, not whole-domain bounds. Older provider comparisons, including libm, libmvec, SLEEF, Arm C and AOCL, retain their separate [provider evidence](../benchmarks/powf_providers_2026-11-09/README.md) and complete variance inventories.

## M6 NEON performance

The current pack-zero-predicate implementation uses one predicate per Gamma pack and the scalar graph for a pack containing zero. The previous iteration restored zero per lane. Both matrices retain every round. All cells use 50 interleaved rounds but are exploratory: placement is unrestricted and all resource rounds are flagged. No result is eligible for promotion.

For x8, a 4096-element stream and opaque exponent 2.4, the exact means are ns per 1,048,576 actual powers, including one socket request:

| Actual implementation | Mean ns | Paired ratio to Apple Rust std, 95% CI |
|---|---:|---|
| Apple Rust std | 2,121,972.40 | Reference |
| Rust libm 0.2.16 | 13,536,148.65 | See retained cell |
| Prior scalar Arm full fused graph | 1,639,837.35 | See retained cell |
| Prior scalar Arm assumed Gamma graph | 1,378,604.59 | See retained cell |
| Magetypes Full packed HIGHP | 1,425,111.87 | [0.670772,0.672403] |
| Magetypes checked Gamma packed HIGHP | 1,435,146.04 | [0.675395,0.677267] |
| Magetypes assumed Gamma packed HIGHP | 1,335,257.81 | [0.627942,0.630543] |
| Existing Magetypes MIDP | 760,834.79 | Different accuracy contract |
| Existing Magetypes LOWP | 471,456.30 | Different accuracy contract |

Zenbench bars for that cell, with exact ns means beside them:

```text
Legacy LOWP                 █           471,456.30 ns
Legacy MIDP                 ██          760,834.79 ns
Packed assumed Gamma HIGHP  ████      1,335,257.81 ns
Scalar assumed Gamma        ████      1,378,604.59 ns
Packed Full HIGHP           ████      1,425,111.87 ns
Packed checked Gamma HIGHP  ████      1,435,146.04 ns
Scalar Full                 █████     1,639,837.35 ns
Apple Rust std              ██████    2,121,972.40 ns
Rust libm 0.2.16             ████████████████████████████████████████ 13,536,148.65 ns
```

The first Gamma implementation took 1,450,365.67 ns in this shape versus 1,372,896.49 ns for its paired scalar Gamma graph. The pack-zero iteration takes 1,335,257.81 ns versus its paired scalar graph's 1,378,604.59 ns, ratio CI [0.966269,0.970802]. These compare each implementation with its interleaved reference; the two implementation versions were collected in separate matrices.

Across the current 32 cells, cost-only intervals against std yield Full: 24 within the 2% bound, eight above; checked Gamma: 27 within, four above, one inconclusive; assumed Gamma: 28 within, four above. All remain exploratory. Stream gains do not imply chain gains, and a stricter checked domain can cost more than Full. No universal ranking is claimed.

## Zen 5 V3 and V4x diagnostics

The paired x8/4096 stream pilot at exponent 2.4 retained all 50 rounds with
verified confinement to the 96 MiB cache class. Every round was rejected for
`HeavyProcesses(1)`; system CPU load at the first observation was
0.03865448147 and available RAM was 46,443,479,040 bytes. These are raw
inadmissible observations, not qualified speed results. The later V3/V4x
pilot had no overlapping study analysis. No full Linux matrix followed.

| Actual implementation | Mean ns per 1,048,576 powers | Paired cost ratio to glibc 2.43 Rust std, 95% CI |
|---|---:|---|
| Rust std | 2,620,902.99 | Reference |
| V3 Full packed HIGHP | 3,324,966.53 | [1.254362,1.285724], inadmissible |
| V3 checked Gamma packed HIGHP | 1,634,087.63 | [0.608158,0.642470], inadmissible |
| V3 assumed Gamma packed HIGHP | 1,570,451.12 | [0.580914,0.620718], inadmissible |
| V4x Full packed HIGHP | 1,661,829.05 | [0.618078,0.653259], inadmissible |
| V4x checked Gamma packed HIGHP | 2,160,017.91 | [0.806828,0.845416], inadmissible |
| V4x assumed Gamma packed HIGHP | 2,004,250.81 | [0.749110,0.783125], inadmissible |

The graph and domain affect the preferred width: the Full graph favors V4x
in this pilot, while Gamma favors V3. This observation does not change a
shipping default or satisfy the 2% gate. The run-heavy summary is
6 s, peak RSS 0.26 GiB, minimum available 44,292 MiB, peak load 1.83.

Hardware counters separately cover 104,857,600 actual powers in each
whole-process x8/4096 stream diagnostic, including startup and the adapter:

| CPU and actual implementation | Retired instructions | User cycles or Darwin cycles elapsed |
|---|---:|---:|
| Zen 5 Rust std | 7,570,501,953 | 1,373,317,244 |
| Zen 5 V3 Full packed HIGHP | 4,670,465,359 | 1,740,293,603 |
| Zen 5 V4x Full packed HIGHP | 1,510,375,150 | 825,106,772 |
| Zen 5 V3 assumed Gamma packed HIGHP | 2,481,435,546 | 773,364,366 |
| Zen 5 V4x assumed Gamma packed HIGHP | 1,916,465,518 | 1,005,305,732 |
| M6 Rust std | 8,295,440,304 | 982,294,258 |
| M6 Full packed HIGHP | 4,989,420,751 | 654,335,864 |
| M6 checked Gamma packed HIGHP | 5,436,140,383 | 660,834,683 |
| M6 assumed Gamma packed HIGHP | 4,060,605,837 | 613,592,618 |

Fewer instructions do not guarantee fewer cycles: the two Zen 5 Gamma
widths reverse that ordering. These are hardware diagnostics, not estimates
or isolated-call latency measurements. The counter runs report Zen 5
2 s, peak RSS 0.03 GiB, minimum available 44,540 MiB, peak load 1.62;
M6 2 s, peak RSS 0.02 GiB, minimum available proxy 10,127 MiB, peak load 1.61.

## Neoverse N1 NEON diagnostics

The homogeneous-hardware x8/4096 stream pilot with exponent 2.4 retained
50 pairs. All were inadmissible. The first resource observation reports
CPU load 0.35765243292, six heavy processes and 28,709,601,280 available
RAM bytes. The complete matrix stopped at this pilot. The setup attempt
using the older scope helper collected no timing and remains preserved.

| Actual implementation | Mean ns per 1,048,576 powers | Status |
|---|---:|---|
| Rust std | 12,938,364.24 | Inadmissible reference |
| Scalar Arm full fused graph | 13,514,370.76 | Inadmissible |
| Full packed HIGHP | 10,968,313.60 | Inadmissible |
| Checked Gamma packed HIGHP | 10,877,038.60 | Inadmissible |
| Assumed Gamma packed HIGHP | 8,582,757.92 | Inadmissible |

The pilot reports 9 s, peak RSS 0.03 GiB, minimum available 27,378 MiB,
peak load 6.16. Separate counters cover 104,857,600 actual powers per
implementation, including the same startup and adapter:

| Actual implementation | Retired instructions | User cycles |
|---|---:|---:|
| Rust std | 8,813,181,149 | 3,309,344,378 |
| Scalar Arm full fused graph | 6,427,878,870 | 3,212,221,737 |
| Full packed HIGHP | 4,973,045,255 | 2,572,449,168 |
| Checked Gamma packed HIGHP | 6,323,037,116 | 2,388,539,227 |
| Assumed Gamma packed HIGHP | 4,160,353,692 | 2,025,858,545 |

Counter collection reports 5 s, peak RSS 0.03 GiB, minimum available
27,400 MiB, peak load 6.22. The initial permission refusal and corrected
run are both retained.

## Promotion and regression gates

A canonical correctness pass requires zero non-NaN bit mismatches, zero NaN-class mismatches, exact rejection masks and unchanged existing expectations. MPFR results are reported separately from graph identity.

A performance pass requires every required comparison cell to have 50 retained, clean interleaved pairs, verified placement and a paired 95% upper cost ratio at most 1.02 against its named reference. A lower ratio bound above 1.02 is a logged regression; overlap is inconclusive. An optimization claim also requires an upper ratio bound below 1.00 in a predeclared intended-use cell while every mandatory cell meets 1.02. One extension to 200 pairs is allowed only when preregistered and retaining all earlier valid rounds. Failed admission and exploratory ratios never become passes by dropping rounds. Linux stops on its first inadmissible pilot/cell.

The current 32-cell timing matrix covers ordinary streams and normalized dependency chains, widths 1/4/8/16, sizes 64/4096 and exponents 2.4/0.5. Bit-distributed valid-input timings, sparse/dense exceptional timings, first-use dispatch cost and qualified no-FMA timing remain outstanding. Numerical corpus coverage does not substitute for these performance axes.

The initial user-scope Linux setup lacked the cpuset controller and collected no timing. The system-scope retry preserved the hard memory cap and verified CPU confinement, but its 50-pair pilot was inadmissible. An overlapping analysis job also invalidates any quiet-run claim for that pilot. Its data remain retained; no clean Linux speed claim follows.

Full CI on the recipe-corrected source `28276c44` passes all 19 checks in 682 s, peak RSS 2.50 GiB, minimum available 37,805 MiB, peak load 5.68. The narrow-pack correction also passes the full-corpus explicit V4/V4x execution. Current scalar and no_std code also compile on Rust 1.89. Later architecture-only import cleanup passes warning-denied i686, Thumb and AArch64 bare-metal checks plus native all-target Clippy. The existing strict-WASM debug suite and all doctests pass with unchanged default stack limits (93 s, peak RSS 6.26 GiB, minimum available 37,307 MiB, peak load 3.74). ARM Miri retains the separate unsupported-NEON-shift tool gap; native tests supply execution coverage.

## Evidence and updates

The current integration evidence is sealed under `powf-highp-simd-2026-11-09`: 1,278 files verify on the independent mirror, with [manifest identity and contents](../benchmarks/powf_providers_2026-11-09/ARTIFACTS.md#canonical-highp-integration-and-packed-kernels). It remains separate from the sealed [scalar port study](../benchmarks/powf_providers_2026-11-09/ARM-RESULTS.md) and [earlier M6 study](../benchmarks/powf_providers_2026-11-09/M6-NEON.md). New results update this reference with exact source/binary identity, flags, ISA, input domain, workload, ns denominator, paired interval, admission status, resource summary and variance-ledger link. Superseded measurements remain identifiable.
