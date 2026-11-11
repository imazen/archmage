# Portable seed and correction study — 2026-11-11

**No public arithmetic graph changes.** This study tests whether masking native
estimates can make a common seed, and evaluates two deterministic MIDP graph
alternatives. Both alternatives remain benchmark-only. Numerical scans pass;
**every timing group failed admission**, so no speed or replacement promotion
is qualified by this run.

## Accuracy bits are not common prefix bits

Intel specifies a relative-error bound for RCPPS, ≤1.5×2^-12; that is not a
promise that truncating its estimate produces a particular common bit pattern.
See [Intel's RCPPS instruction specification](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2b-manual.pdf)
and [the VRCP14PS specification](https://cdrdv2-public.intel.com/835757/325383-sdm-vol-2abcd.pdf).
ARM defines its own reciprocal estimate algorithm; see
[Arm's floating-point estimate and step description](https://support.arm.com/documentation/ddi0487/mc/-Part-E-The-AArch32-Application-Level-Architecture/-Chapter-E1-The-AArch32-Application-Level-Programmers--Model/-E1-3-Advanced-SIMD-and-floating-point-instructions/-E1-3-11-Pseudocode-description-of-floating-point-operations?lang=en).

Mathematical consequence: an error interval can straddle any quantization
boundary. Two accurate estimates can therefore truncate or round into adjacent
cells. An accuracy bound alone does not establish a shared mantissa prefix.

Actual raw-instruction witnesses:

| Input | ISA / instruction | Reciprocal estimate bits |
|---|---|---|
| 1.0 (`3f800000`) | Zen 5 V3 RCPPS | `3f7ff000` |
| Same | Zen 5 AVX-512 VRCP14PS, logical x16 | `3f800000` |
| Same | M6 NEON FRECPE | `3f7f8000` |

Clearing **all 23 fraction bits** leaves `3f000000` (0.5) versus `3f800000`
(1.0) in the two x86 forms: the exponent boundary survives the mask.

There is also a direct x86/NEON witness for keeping eight fraction bits:

| Input `3f804000` | Raw estimate | `bits & ffff8000` |
|---|---|---|
| V3 | `3f7f7800` | `3f7f0000` |
| NEON | `3f7f8000` | `3f7f8000` |

Round-to-nearest-even before clearing low bits does not solve this generally.
A finite, targeted boundary search found these real V3/VRCP14 disagreements:

| Fraction bits kept | Input bits | Quantized V3 | Quantized VRCP14 |
|---:|---|---|---|
| 4 | `3f783e08` | `3f800000` | `3f880000` |
| 8 | `3f7f8038` | `3f800000` | `3f808000` |
| 12 | `3f7ff7f8` | `3f800800` | `3f800000` |

No keep-zero rounding witness appeared in that finite sample; that observation
is not a universal guarantee. These are seed/quantizer witnesses, not a proof
that every possible refinement of those seeds must differ. Convergence plus a
correct rounding/certification algorithm is a different possible contract.

A guarded canonicalizer could define one coarse mathematical result, use native
estimates to propose a cell, and certify its boundaries with residuals. For
positive reciprocal, exact comparisons of x times the cell endpoints with 1
identify the right cell. FMA residuals can support those comparisons when their
sign is proved reliable on the selected domain. Boundary cases then need repair
or fallback. The error interval and maximum correction count need a proof;
masking alone supplies neither. Its added arithmetic, comparisons and selection
cost are **not measured here**.

The estimate API itself is another trap: NEON's backend `rcp_approx` already
performs one Newton step. The first diagnostic dump used that refined backend
operation; it is preserved, but not labeled as raw FRECPE evidence. The corrected
dump invokes FRECPE/FRSQRTE directly under a NEON proof. The generator and all
four affected backend trait docs now describe backend approximations accurately.
No implementation or numerical expectation changed in that documentation fix.

## Deterministic alternatives

Both retain the existing integer seed and a fixed IEEE graph, so they need no
cross-ISA hardware-estimate agreement. The unpublished study uses native FMA
under V3/NEON proofs and the existing exact fused primitive for Scalar.
Neither graph is wired into a public option or the public WASM dispatcher.

Reciprocal, using y as the integer seed:

```text
e = fma(-x, y, 1)
a = fma(y, e, y)
b = fma(e, e, 1)
result = a*b
```

In real arithmetic this is y×(1+e)×(1+e²), the same fourth-order error reduction
as two Newton iterations. Rounded results differ from the current graph.
The two middle nodes are independent. The V3 probe emits three FMA operations
and one multiplication, versus two FMA and two multiplications in the current
graph; the instruction count alone does not capture the shorter dependency path.

Rsqrt uses a shorter quadratic correction:

```text
y = fixed_rsqrt_seed(x)
e = fma(-x, y*y, 1)
p = fma(0.375, e, c1)
result = y*fma(p, e, 1)
c1 = RN_f32(0.5 + 15*0.07²/64) = bits 3f004b44
```

The coefficient absorbs the linear Chebyshev component of the omitted cubic
term over |e|≤0.07. This is a measured bounded approximation, not a correctly
rounded rsqrt or a complete-domain analytic proof. Its V3 probe has three FMA
operations and two multiplications. The current two-step rsqrt graph retains
more accuracy; the quadratic intentionally spends part of MIDP's error budget.

## Numerical evidence

Fixed-size x4 scans enumerate all 16,777,216 floats in [1,4), plus directed
positive edge-exponent cases through the existing Normal operating domain.
Every output is compared bitwise with an independent scalar expression of its
fused graph. Scalar, V3 and native M6 NEON pass with identical maxima/witnesses.
References for relative error use f64 reciprocal/sqrt, not an MPFR ULP claim.
The MIDP assertion stays **relative error ≤1e-4**.

| Graph | Maximum relative error | Witness input |
|---|---:|---|
| Current reciprocal MIDP | 1.141549399374e-5 | `3ff12999` |
| Parallel reciprocal correction | 1.145700113618e-5 | `3ff13129` |
| Current rsqrt MIDP | 4.723778778297e-6 | `406eb88c` |
| Quadratic rsqrt correction | 2.779628820479e-5 | `4004662a` |

The rsqrt change is a genuine accuracy tradeoff within the unchanged bound.
No full-normal-domain enumeration, negative-input qualification, WASM prototype
qualification, special-value contract or new checked adapter is claimed.

## Exploratory timing only

One frozen Zen 5 V3 x8 run: 50 interleaved rounds, four groups/four arms, strict
admission, enforced uniform cpuset, no target-cpu=native, no retry/extensions.
Source `3f557f9c05e491cbff4399b68dad183849238971`; executable hash, compiler,
command and frozen plan are in raw SOURCE.json. Later raw-NEON diagnostic and
documentation fixes do not change the measured x86 kernels.

**None of these groups qualifies.** Gate observations report HeavyProcesses(1),
even while aggregate foreign CPU load is below 1%; the historical competing PID
was not recorded, so its identity is not established. No process was killed and
no gate was relaxed. All samples remain in
[the paired analysis](correction-zen5-pairs.json).

Times are ns per 64 inputs (streams) or 64 dependent eight-lane steps (chains).
They describe this small hot fixture, not pure instruction latency or long buffers.

| Implementation | recip stream | recip chain | rsqrt stream | rsqrt chain |
|---|---:|---:|---:|---:|
| Current portable MIDP fused | 2.927 | 176.692 | 5.046 | 274.324 |
| Parallel reciprocal / quadratic rsqrt | 2.806 | 154.226 | 3.694 | 246.346 |
| Existing Magetypes recip/rsqrt | 2.502 | 158.590 | 3.597 | 217.970 |
| HIGHP division/composition | 3.557 | 77.044 | 10.430 | 237.864 |
| Clean rounds / retained | 15/50 | 0/50 | 27/50 | 25/50 |

Zenbench's reciprocal chain bars, **all 50 rounds flagged**:

```text
HIGHP division              ██████████████████ 77.0ns
parallel residual correct…  █████████████████████████████████████ 154.2ns
magetypes recip             ██████████████████████████████████████ 158.6ns
current portable MIDP fus…  ██████████████████████████████████████████ 176.7ns
```

The apparent reciprocal-chain crossover versus legacy is only a hypothesis.
All-round paired ratios (10,000 paired bootstrap resamples, no IQR removal)
are in the JSON, but flagged admission prevents treating even narrow intervals
as qualified evidence. The rsqrt prototype remains slower than legacy in both
means. Every required clean cell still needs upper95% time ratio ≤1.02 before
any replacement promotion; these results do not meet that gate.

The first next qualification target is the parallel reciprocal graph. The
quadratic rsqrt graph is a separate speed/accuracy tradeoff. A compact table
seed or certified coarse native estimate is another possible experiment, not
an implemented result. HIGHP division is already a portable option and has the
lowest reciprocal-chain time in this exploratory fixture.

Resource records: corrected dense scan/build/lint 3 s, peak-RSS 0.24 GiB;
probe validation 4 s, peak-RSS 0.42 GiB; timing rc=1 for retained admission
failure, 7 s, peak-RSS 0.25 GiB, minimum available 42007 MiB, peak load 0.56.
Native M6 numeric validation reports 4 s and wrapper RSS rounded to 0.00 GiB;
that is not zero memory. Corrected M6 raw-instruction/lint run: 5 s, peak-RSS
0.36 GiB. Raw output, binaries, logs and source snapshots live under
`/mnt/v/output/archmage/portable-corrections-2026-11-11/`, mirrored under
`/mnt/tower/output/archmage/portable-corrections-2026-11-11/`.

Reproduce with `just primitive-seed-study --dense`, `--seeds`, or `--masks`
(the latter requires V3 and AVX-512). `--measure --correction-study` selects the
frozen timing matrix in the main study binary. `primitive-correction-analyze`
uses every retained pair. The exhaustive numerical work stays opt-in.
