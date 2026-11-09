# MIDP / LOWP replacement completion, 2026-11-09

The production graph begins at 458e0978. Keep every round, failure and frozen
binary. This study changes no existing accuracy, bit identity or performance
threshold. Actual legacy `pow_lowp`, `pow_midp`, and `pow_midp_unchecked` are
separate named comparisons. LOWP aliases and MIDP precise aliases retain their
existing contracts; they are not independent performance arms.

All 48 bounded const encodings are implemented. Verify scalar single-lane
execution, widths 4/8/16 and every tail remainder (lengths 0..33), all selected
contexts and exponents 0, -0, 0.5, 2.4, 4. Use mixed distinct bit-distributed
Gamma inputs and admitted zero lanes. A tail must retain the vector's token.
Backend-dependent ScalarCaller intentionally uses separate arithmetic; calling
that from a native fused loop is not a matching tail. Portable uses one fused
graph everywhere. Existing independent rational fixtures remain the graph oracle.

The first optimization removes only range work implied by the domain: Gamma
range clamps and its pre-log zero substitution; positive-normal LOWP clamps.
The post-result zero mask remains. General nonnegative LOWP keeps its clamp:
zero's biased log times a huge exponent must not reach an invalid conversion.
Gamma zero's intermediate exponent remains between -512 and 0; masked results
have no observable numerical change under the existing floating environment.

Freeze mandatory dev timing cells before collection: V3 x4/x8/x16 and V4x
x16, stream and normalized dependency chain, sizes 64 and 4096, exponents
2.4 and 0.5; additionally lengths 65 and 4097 for tail throughput at exponent
2.4. Each cell compares portable and backend-dependent Gamma AssumeValid
Caller MIDP/LOWP to all three actual legacy methods. This is 40 cells (32 aligned and 8 tail cells). Native
portable/backend graphs may be instruction-identical; report ties as ties.
Then collect the range/validation matrix in representative V3 x8 stream/chain,
sizes 64/4096, exponent 2.4: all 48 encodings plus the legacy implementations.
Caller/Automatic native aliases are verified rather than claimed as savings.
No blanket promotion follows from a faster subset: scalar/no-FMA, NEON and WASM
remain separately identified coverage requirements.

50 paired rounds per cell, strict zenbench gates, verified placement, no removed
rounds. Upper paired 95% ratio <=1.02 for no regression and <1.00 for a declared
optimization. Retain busy runs as unqualified evidence. No benchmark overlaps
this session's build, numerical analysis, or codegen work. Means and zenbench
bars are reported in ns per complete buffer; pair analysis uses all samples.

Artifacts: dev `/home/lilith/tmp/powf-mid-low-final-20261109`, promoted to
`/mnt/v/output/archmage/powf-mid-low-final-2026-11-09` with a verified NAS mirror.

## Second frozen revision: positive Gamma

The first 40-cell revision retained zero. It wins against checked MIDP in all
40 cost-only intervals, but loses to unchecked MIDP in 15 portable / 16 backend
cells; five cells have fully clean admission. This motivates a new explicit
`PositiveGamma` domain, not changing the old Gamma semantics. It excludes +0
and removes the result mask. Checked mode rejects zero; AssumeValid promises
positive inputs. Every positive result remains bit-identical to Gamma.

The second source revision retains all earlier output and adds 16 positive
Gamma encodings (64 bounded total). All 40 native cells compare old Gamma and
positive Gamma side by side, plus all three legacy methods. Eight scalar API
cells use Automatic selection and actual public scalar calls, compared with
actual legacy ScalarToken splats (legacy has no scalar power method). The four
range/validation cells now contain all 64 bounded encodings plus three legacy
methods. These exact 52 cells are frozen before timing the second revision.
