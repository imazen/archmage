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
| Full | All f32 values | All f32 values | HIGHP only; overflow, underflow, signed zero and NaN classification follow the selected graph |

Checked rejects the entire vector with the exact lane mask at the first failing stage: width/options, exponent, base, output range. AssumeValid removes numerical checks while retaining width/option checks and memory safety. Violating an assumed domain has unspecified numerical results. Gamma's bounds justify deleting range/special-case work; checking those stricter input bounds still costs instructions.

## Const options and dispatch

Use `try_powf_with::<{ PowfOptions::PORTABLE_GAMMA_HIGHP.with_validation(PowfValidation::AssumeValid).bits() }>(x, n)`, or the matching vector delegate. The integer stays const through delegates, dispatch and kernels. There are 80 supported configurations with std and 64 without it. Runtime adapters decode to the same const graphs.

Automatic preserves a capable caller or upgrades the whole operation. Caller prevents autonomous upgrades. FMA selection and packing remain private tuning choices. BackendDependent HIGHP preserves its std call path and may inherit std's own dispatch.

## ISA implementations and measured coverage

| ISA or caller context | Canonical HIGHP implementation | Correctness coverage | New library timing status |
|---|---|---|---|
| Scalar x86-64 | Scalar graph; std exact mul_add or no_std software FMA; Automatic can upgrade | Independent 27,909-pair corpus, all widths | Linux pilot inadmissible; no qualified speed claim |
| x86 without V3, including FMA without AVX2 | Caller scalar graph; std may use hardware FMA through its own implementation; no_std exact software | New below-V3 runtime execution pending | Not measured |
| x86 V3 AVX2 and FMA | Four f64 lanes per packed graph; scalar exceptional-pack fallback | Full corpus, required V3, mixed/broadcast lanes | Initial Zen 5 pilot retained and inadmissible |
| x86 V4 and V4x AVX-512 | Eight f64 lanes, four-lane packing for x4, scalar tail for x1 | Full corpus in explicit V4 and V4x contexts | Comparative native study in progress |
| AArch64 NEON Neoverse-N1 | Two f64 lanes per graph, native FMA | Native std/no_std fixtures and full corpus | Not measured for the new library |
| AArch64 NEON Apple M6 | Same two-lane graph | Native std/no_std fixtures and full corpus | Two 32-cell exploratory matrices; no class confinement |
| WASM strict SIMD128 | Scalar exact fused fallback | Full corpus with and without std | Not measured; execution supplies no native timing claim |
| WASM relaxed SIMD128 | Two-lane graph only when the cached f64 probe confirms fusion; exact fallback otherwise | Full corpus with and without std, ordinary and deterministic engine modes | Not measured |
| i686 | Scalar canonical graph | Full corpus in scalar and Automatic contexts | Not measured |
| ARM bare metal and Thumb | Exact no_std scalar fallback | Compilation on aarch64-unknown-none and thumbv7m-none-eabi | Runtime and performance not measured |

The corpus runs cover widths 1/4/8/16, broadcasts and mixed bases. They found zero promised bit/class mismatches. NaN payload differences remain in complete variance ledgers. The exact graph oracle uses rational IEEE rounding and separately frozen C-extracted coefficients; MPFR independently measures mathematical accuracy.

## Measured mathematical accuracy

| Implementation and sample | Observed maximum error | Correctly rounded-result differences |
|---|---|---|
| New canonical HIGHP, 27,909-pair broad sample, positive finite reference subset | 0.506155662239 ULP | 17; same outputs as the prior scalar Arm graph and Linux std on this sample |
| Prior canonical scalar graph, 515 separately identified hard witnesses | 0.500007618219 ULP | 110; new library replay pending |
| Existing canonical MIDP, expanded Gamma sample | Below 47.902 ULP | See the [replacement study](../benchmarks/powf_module_2026-11-07/REPLACEMENTS.md) |

These are observed maxima, not whole-domain bounds. The hard-witness row describes the frozen scalar study, not a new SIMD measurement. Older provider comparisons, including libm, libmvec, SLEEF, Arm C and AOCL, retain their separate [provider evidence](../benchmarks/powf_providers_2026-11-09/README.md) and complete variance inventories.

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

## Promotion and regression gates

A canonical correctness pass requires zero non-NaN bit mismatches, zero NaN-class mismatches, exact rejection masks and unchanged existing expectations. MPFR results are reported separately from graph identity.

A performance pass requires every required comparison cell to have 50 retained, clean interleaved pairs, verified placement and a paired 95% upper cost ratio at most 1.02 against its named reference. A lower ratio bound above 1.02 is a logged regression; overlap is inconclusive. Failed admission and exploratory ratios never become passes by dropping rounds. Linux stops on its first inadmissible pilot/cell.

The initial user-scope Linux setup lacked the cpuset controller and collected no timing. The system-scope retry preserved the hard memory cap and verified CPU confinement, but its 50-pair pilot was inadmissible. An overlapping analysis job also invalidates any quiet-run claim for that pilot. Its data remain retained; no clean Linux speed claim follows.

Root full CI passed all 19 checks in 688 s, peak RSS 2.51 GiB, minimum available 41,235 MiB, peak load 7.14. The subsequent narrow-pack correction passes the HIGHP fixtures, full-corpus V4/V4x execution and Clippy. Current scalar and no_std code also compile on Rust 1.89. ARM Miri retains the separate unsupported-NEON-shift tool gap; native tests supply execution coverage.

## Evidence and updates

The current integration evidence is collected under `powf-highp-simd-2026-11-09`, separate from the sealed [scalar port study](../benchmarks/powf_providers_2026-11-09/ARM-RESULTS.md) and [earlier M6 study](../benchmarks/powf_providers_2026-11-09/M6-NEON.md). New results update this reference with exact source/binary identity, flags, ISA, input domain, workload, ns denominator, paired interval, admission status, resource summary and variance-ledger link. Superseded measurements remain identifiable.
