# Bounded MIDP / LOWP implementations and replacement costs

The 2026-11-09 implementation has all four families: portable/backend-dependent
MIDP/LOWP, each with four domains, checked/assumed validation and automatic/caller
execution: 64 bounded encodings. `PositiveGamma` excludes zero from Gamma's
[2^-16,2^16] base range and [0,4] exponent range. Every positive output retains
Gamma bits. It removes zero masks; AssumeValid also removes validation.

`try_powf_with_token::<FLAGS,_>(token,x,n)` supplies the scalar/tail form with
the same feature proof as the vector. All 64 policies agree across single values,
x4/x8/x16 and lengths 0..33 plus 64/65/127/255/511/512/513, including mixed
admitted zero lanes. The no-token scalar Caller form explicitly uses ScalarToken.

The owner requests rough numerical equivalence rather than legacy exceptional
compatibility. Backend-dependent separate multiply/add is allowed. The existing
Gamma sample observed MIDP maxima of 47.90180805 ULP fused and 48.56180874 separate;
LOWP maxima were 93408.85148075 and 93411.46518661 ULP, about 0.56% relative error.
These are observations, not global bounds. Large unrestricted exponents can
magnify LOWP's log bias catastrophically with either rounding choice; see
[NONFUSED.md](NONFUSED.md). PositiveNormal alone is not a LOWP accuracy preset.

## Admission and limits

Latest public-tail measurements: 48 dev scalar/native/tail cells, four all-policy
cells, twelve ScalarCaller diagnostics; 38 M6 scalar/native/tail and four
all-policy cells. Every cell retains all 50 paired rounds. Only two of the 48 dev
cells have fully clean admission; all M6 cells remain exploratory. No overall
performance promotion is claimed. The full run plan and all earlier source
versions remain in [MIDLOW-PLAN.md](MIDLOW-PLAN.md).

Portable positive-Gamma MIDP beats unchecked legacy MIDP in 39/40 native cost
intervals, with one inconclusive. Backend-dependent MIDP has 37/40 within the
2% bound, two above and one inconclusive; the failing cells are flagged for
interference. Both families beat checked MIDP in all 40 native cost intervals.
Portable and backend LOWP beat legacy LOWP in all 40 native intervals. All eight
actual scalar Automatic cells favor the new positive-Gamma configurations.
These counts describe cost intervals, not promotion eligibility.

## Representative times

Opaque exponent 2.4, x8, 4096-value stream; ns per complete buffer. The new rows
use PositiveGamma, AssumeValid, Caller. These representative cells are flagged.

| ISA / actual method | Mean ns | Paired 95% ratio interval to named legacy |
|---|---:|---|
| V3 portable MIDP | 1396.06 | [0.950146,0.954324] versus `pow_midp_unchecked` |
| V3 legacy `pow_midp_unchecked` | 1466.13 | Reference |
| V3 legacy `pow_midp` | 2288.87 | Reference; new MIDP ratio [0.609091,0.610792] |
| V3 portable LOWP | 958.29 | [0.755705,0.774104] versus `pow_lowp` |
| V3 legacy `pow_lowp` | 1253.14 | Reference |
| M6 NEON portable MIDP | 1942.70 | [0.957523,0.966387] versus `pow_midp_unchecked` |
| M6 NEON legacy `pow_midp_unchecked` | 2019.69 | Reference |
| M6 NEON legacy `pow_midp` | 2993.92 | Reference; new MIDP ratio [0.645696,0.652376] |
| M6 NEON portable LOWP | 1568.92 | [0.854192,0.868786] versus `pow_lowp` |
| M6 NEON legacy `pow_lowp` | 1821.80 | Reference |

## Cost of portability

For the same positive-Gamma/assumed/caller configuration and measured buffer:

| Context | MIDP backend → portable ns | LOWP backend → portable ns |
|---|---:|---:|
| V3 native FMA | 1396.78 → 1396.06 | 959.70 → 958.29 |
| M6 NEON native FMA | 1939.23 → 1942.70 | 1566.66 → 1568.92 |
| ScalarCaller on dev, exact software FMA | 14823.71 → 37103.36 | 13505.23 → 29076.13 |

Native FMA uses the same arithmetic graph; the tiny observed differences do not
establish extra arithmetic cost. Software FMA adds 22279.65 ns per 4096 values
for MIDP (5.44 ns/value), and 15570.90 ns for LOWP (3.80 ns/value). These timings
also have flagged admission. They measure ScalarCaller on dev, not an older CPU.
Automatic avoids that software path when it can upgrade to native FMA.

## Flag rankings

In the four dev all-policy cells, all 128 AssumeValid/Checked comparisons favor
AssumeValid; all 128 LOWP/MIDP, 64 Gamma/NonNegativeNormal and 64
PositiveGamma/Gamma comparisons favor the smaller graph/range work. M6 has
123/128 validation comparisons favoring AssumeValid and five overlapping; the
other three comparisons all favor the same directions. Caller/Automatic and
portable/backend variants intentionally share arithmetic on matching FMA tiers;
they are not guaranteed independent speed wins.

## Verification and evidence

Exact-graph fixtures remain separate from MPFR mathematical error measurements.
The complete family grids pass strict WASM and relaxed WASM with verified fused
and separate engine choices; no-FMA QEMU and native NEON also pass. Unoptimized
WASM library tests pass with the original stack limit. Numerical tail grids live
in four integration targets to bound debug compiler memory; only these new safe
numerical grids are excluded by the explicit Miri runner list. Existing Miri and
regression expectations are unchanged.

The final source/review/CI and artifact hashes are appended after verification.


## Legacy entry-point remapping

The existing power methods now route through the const-configured engine. Their
private clamp/edge configurations reuse the existing log building blocks; the
scalar path keeps its established composition to preserve register allocation.
All 72 actual-before/actual-after x86 probes are identical across scalar, V3,
V4 and V4x, widths 4/8/16, dynamic and literal exponents. All non-NaN output bits
and NaN classification agree with the frozen log/exp composition in the retained
structured corpus. No global error-bound or full IEEE powf claim follows.

The separate comparison against a hand-written composition initially reported
70/72 matching x86 bodies. The two scalar x16 LOWP differences already existed
before remapping: using the actual frozen entry points as the compiler oracle
resolves both without weakening assembly normalization. Every earlier report
and failed implementation attempt is retained.
