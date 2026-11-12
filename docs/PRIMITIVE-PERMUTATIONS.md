# Primitive configuration cells and evidence

This reference separates **supported options**, **canonical configuration
aliases**, **numerical contracts**, and **measured execution costs**. It covers
Magemath min/max (f32/f64), reciprocal and rsqrt (f32). Pow/exp/log have complete
option inventories in rustdoc; their remaining timing expansion is outside this
phase. None of the benchmark-only reciprocal research designs is a public option.

## Zero is an ordinary clamp operand

`NORMAL` / `UNCHECKED_NORMAL` are specialized **finite normal operands only,
zero excluded** configurations. Both operands must satisfy that restriction.
`min(x, 0)` and `max(0, 5)` therefore require another configuration, even when the
result would be normal. Checked Normal returns an error; unchecked Normal gives
unspecified numerical output on those calls.

For finite inputs including zeros/subnormals, choose `MinMaxDomain::Finite`.
Keep Ordered zeros when min must select -0 and max +0 on opposite-zero ties;
choose Backend zeros explicitly if either sign is acceptable. Full-domain
presets also admit zeros. `RELAXED` permits backend NaN and zero behavior and
needs no range check; it is not a promise of portable zero bits.

## Layer 1: every option permutation

The [min/max matrix](../magemath/src/minmax/permutations.md) contains all 72 raw
builder combinations, redirecting equivalent `.bits()` encodings to 24 cells.
The [reciprocal/rsqrt matrix](../magemath/src/recip/permutations.md) contains all
48 combinations: 16 Full-domain approximations are unsupported; 32 supported
combinations redirect to 22 cells. The tables are included directly in rustdoc.

These counts come from compiling the actual option sources and their private
validity predicates, with and without std. No second flag encoder is used.
`Mxx` / `Rxx` are traceability IDs using the current hexadecimal flags. Construct
the public API with builders and `.bits()`, not handwritten numeric flags.

| Family | Domain | Result policy | Validation | Execution |
|---|---|---|---|---|
| Min/max | Full / Finite / Normal | NaN Number/Propagate/Backend × Zero Ordered/Backend | Checked / AssumeValid | Automatic / Caller |
| Reciprocal/rsqrt | Full / Normal | Highp/Midp/Lowp × BitExact/BackendDependent | Checked / AssumeValid | Automatic / Caller |

Configuration redirects mean equal encoded flags. A backend algorithm alias or
equal compiled function is a different claim, scoped to its ISA, width and entry
form. A finite-domain NaN-policy redirect does not expand the accepted domain.
In particular, equal machine code never makes zero valid for Normal.

## Layer 2: numerical contract and evidence

| Cells / graph | Contract on accepted inputs | Measured error evidence |
|---|---|---|
| All min/max cells | Exact selected operand; requested NaN and zero policies | No arithmetic approximation. Zero signs and NaN class are checked separately, not hidden in a relative-error score. |
| Reciprocal HIGHP | Rounded f32 `1/x`, non-NaN bits portable | Independent exact-rational fixtures; the fixed benchmark preflight also compares to f32 division. |
| Rsqrt HIGHP | Rounded f32 `1 / rounded_sqrt(x)` | Independent graph fixtures. This is not correctly rounded mathematical rsqrt; observed mathematical deviation is reported separately. |
| Portable LOWP recip / rsqrt | Fixed seed + one fused Newton step; relative error ≤6e-3 | Full x4 [1,4) scan: 3.362326737260e-3 / 1.752338672098e-3 versus f64 references. |
| Portable MIDP recip / rsqrt | Fixed seed + two fused Newton steps; relative error ≤1e-4 | Same scan: 1.141549399374e-5 / 4.723778778297e-6. |
| Backend LOWP recip / rsqrt | ISA/width-dependent estimate/graph; relative error ≤6e-3 | V3 x4 scan: 3.152981953462e-4 / 2.586608049593e-4; M6 NEON x4: 2.853393554688e-3 / 3.276822880217e-3. |
| Backend MIDP recip / rsqrt | ISA/width-dependent refinement; relative error ≤1e-4 | V3 x4 scan: 1.538581244089e-7 / 2.108307648628e-7; M6 NEON x4: 8.228816000135e-6 / 1.618327600566e-5. |

The approximation domain is 2^-125 ≤ abs(x) ≤ 2^125; rsqrt additionally requires
x > 0. Every Checked/AssumeValid and Automatic/Caller variant retains its
declared numerical contract. Portable means result agreement, not software-only
execution: native FMA is used when proved, with exact software otherwise.

The [fusion report](../benchmarks/primitive_variants_2026-11-10/FUSION.md) and
[primitive contract reference](PRIMITIVE-CONTRACTS.md) preserve scan scope,
source identities and witnesses. Measured maxima are not silently promoted to
stronger whole-domain promises. In particular, backend x4 observations are not
presented as measurements of every wider vector or every future CPU estimate.

## Layer 3: timing dimensions and statuses

A timing leaf is identified by **cell × operation × scalar type × logical width
× CPU/ISA × entry form × workload × compiled source/dependencies**. A precision
grade is not a universal speed ordering. A measurement for one leaf is not
silently copied to another because its preset name sounds similar.

| Status | Meaning |
|---|---|
| Unsupported options | Const API cannot instantiate the requested option combination. |
| Unsupported input | This workload violates the selected domain; no unchecked timing is collected for it. |
| Redirect | Equal flags; use the indicated canonical cell, retaining all other dimensions. |
| Clean retained cell | Every prescribed paired observation is clean; report its interval and the whole-suite status separately. |
| Descriptive, flagged | Timings retained, but resource admission failed; no confirmed promotion claim. |
| Exploratory | Explicitly unconfined M6 scheduler timings; excluded from the strict promotion gate. |
| Unmeasured | No attributable measurement of this exact leaf. It is not a measured failure or a zero-cost operation. |

The first complete canonical-cell kernel run is being qualified. Its frozen
matrix covers all 46 supported flag sets, min/max in f32/f64, reciprocal/rsqrt
in f32, and all three logical widths (f32x4/x8/x16; f64x2/x4/x8). Each has
stream64, stream4096 and chain64 workloads. Min/max also has zero-clamp64,
including signed zeros and subnormals; the four Normal cells are explicitly
unsupported on that workload. There are **66 groups / 1,566 arms per context**,
including the existing Magetypes baseline in each group.

These are **typed kernel** timings: the supplied token and width are retained.
Caller/Automatic still selects the scalar fusion provider where applicable.
Scalar/array API-dispatch overhead, first-use detection and invalid-input error
latency are separate leaves and remain explicitly unmeasured by this kernel
matrix. Existing matching-tail correctness tests do not substitute for a tail
microbenchmark.

Strict runs retain five independent processes × 50 paired rounds, pinned
zenbench73302a1c, 95% family confidence and the unchanged 2% bound. M6 is one
exploratory process. Every round, admission failure and source/binary identity
is retained; there is no favorable retry or threshold change. Per-case numerical
preflight runs before timing. Its observed error on fixed inputs/chain states
is reported independently from the prior full scans and normative bounds.

The complete Zen 5 Scalar, Intel V3 and M6 exploratory tables are included directly in the
[min/max rustdoc](../magemath/src/minmax/mod.rs) and
[reciprocal/rsqrt rustdoc](../magemath/src/recip/mod.rs), nested by operation,
type and width. They contain all supported canonical cells and workloads,
including the explicit Normal/zero-clamp exclusions. Intel has 32/66 fully
clean groups, but its whole suite fails admission. Zen 5 Scalar has no fully
clean group; Automatic scalar fusion selection is explicitly not a speed win
in its flagged data. All 3,300 M6 group-round
observations are resource-flagged; none establishes strict promotion.
The [matrix report](../benchmarks/primitive_variants_2026-11-10/POLICY-MATRIX.md)
records source identity, method, resource usage and remaining contexts.

Pending contexts are not filled with historical timings. Earlier measurements
remain available, with their exact flags, source and harness limitations, in
the [initial study](../benchmarks/primitive_variants_2026-11-10/README.md),
[fusion study](../benchmarks/primitive_variants_2026-11-10/FUSION.md), and
[patched reciprocal study](../benchmarks/primitive_variants_2026-11-10/CERTIFIED-RECIP.md).

## Regeneration

`primitive-policy-inventory OUTPUT` compiles the real builders with/without std.
`primitive-policy-tables OUTPUT` renders all five family inventories;
`primitive-policy-tables OUTPUT --check` rejects drift. Generated Markdown stays
under src so published rustdoc includes it. The benchmark registry is generated
from the same supported canonical flags; it instantiates literal const arguments
through the hot kernels instead of passing runtime options.
