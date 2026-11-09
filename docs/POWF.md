# Configurable f32 power

`magetypes::powf` owns the MIDP/LOWP approximation graphs and scalar std HIGHP
backend for scalar and vector power. `f32x1`, generic `f32x4`, `f32x8` and enabled `f32x16` delegate to it.
The existing `pow_lowp`, `pow_midp` and related methods are unchanged.

```rust
use magetypes::powf::{PowfOptions, PowfDomain, PowfExecution, try_powf_with};
const OPTIONS: u32 = PowfOptions::new()
    .with_domain(PowfDomain::Gamma)
    .with_execution(PowfExecution::Caller)
    .bits();
assert_eq!(try_powf_with::<OPTIONS>(4.0, 0.5).unwrap(), 2.0);
```

`try_powf_with::<OPTIONS>` takes an integer const configuration produced by
a const builder ending in `.bits()`. Portability, precision, domain, validation
and execution are independent const choices.
The flags remain const through vector delegates,
backend selection and the graph. A capable caller evaluates the vector directly
in its feature region, without an array dispatch boundary. Reserved encodings and unsupported combinations are compile errors.
`Checked` enforces the declared domain; `AssumeValid` removes numerical checks.

For options selected at runtime, `try_powf_runtime` and
`try_powf_array_runtime` accept the same builder value. Their policy decoder
selects literal const instantiations outside the graph. These APIs share the graph and rejection contract. The array const
entry is `try_powf_array_with::<OPTIONS, N>`; the vector method is
`vector.try_powf_with::<OPTIONS>(exponent)`.

`PowfErrorKind` is non-exhaustive so future rejection reasons can be added.
External matches retain a fallback arm; the numerical study panics on an
unknown reason rather than silently classifying it as an existing stage.

The default BitExact/MIDP/Checked contract is identical result bits across
admitted backends for the same
accepted inputs and options, under round-to-nearest and gradual underflow.
It is an approximation to `exp2(n * log2(x))`, not correctly rounded power.
The exponent is uniform across a vector. All lanes must pass; an error returns
its reason and a low-bit-first rejected-lane mask, with no partial result.
Width and exponent checks precede base checks, which precede output-range checks.

| Domain | Base | Exponent | Computed exp2 argument |
|---|---|---|---|
| `PositiveNormal` (default) | Positive finite normal | Finite normal or signed zero | Checked in [-126, 127.5] |
| `NonNegativeNormal` | Same, plus +0 for nonnegative exponent | Same | Checked in [-126, 127.5] |
| `Gamma` | +0 or [2^-16, 2^16] | Normal or signed zero in [0, 4] | Proven in range; check omitted |

Negative zero bases, negative bases, infinities, NaNs and nonzero subnormal
inputs are rejected. Admitted +0 produces +0 for positive exponents and 1 for
zero exponents. Bounds refer to the graph's computed argument, not exact real
power. No whole-domain accuracy bound is promised. Unsupported array widths
return `UnsupportedWidth` with mask zero; supported widths are 1, 4, 8, and 16
when `w512` is enabled.

`PowfExecution::Automatic` preserves a capable caller token or upgrades the
whole operation through the supported runtime dispatch. V3/V4/V4x and NEON
use native fused multiply-add. Relaxed WASM uses native madd only when its
existing fusion probe passes; otherwise it uses packed exact software. Strict
WASM uses the packed software path. Scalar fallback uses exact array software.
With `std`, an independent cached FMA fixture screen can admit `f32::mul_add`
on otherwise unsupported native configurations. The standard's FMA contract
supplies the correctness requirement; finite probes only screen known bugs.
They cannot certify direct `std::powf` against the fixed graph. Direct std power
is therefore available only as BackendDependent HIGHP, never BitExact.

`PowfExecution::Caller` prevents autonomous upgrades. A scalar caller then
uses software, making backend tests explicit. This option also works in an
integration test whose dependency was compiled without `cfg(test)`.
It is not a compile-time removal of other backends.

The graph retains the existing midp coefficients, with four log2 FMA steps,
one multiplication by the exponent, and six exp2 FMA steps. Domain selection
specializes the internal const flags. For `Gamma`, an independent rational/RNE
interval certificate bounds the reduced ratio below 1/4, the positive Horner
polynomial below 2.96, the log2 approximation in magnitude below 17, and its
product with the exponent below 68. This removes the second rejection branch
without changing the graph or invoking fast-math assumptions.

The independent rational fixtures cover boundaries, mixed exceptional lanes,
gamma/PQ exponents and deterministic random samples. The separate MPFR study
measures mathematical error from actual library outputs. Its 26,823 sampled
pairs had zero graph mismatches. The later adaptive search expanded the identical-input comparison to 26,857
pairs. Its largest measured canonical errors were below 438.504 ULP for
`PositiveNormal`, 416.579 ULP for `NonNegativeNormal`, and 47.902 ULP for
`Gamma`. These replace the earlier sampled maxima of 91.258, 92.613 and
25.397 ULP; none is a worst-case guarantee. Scalar and V3 canonical bits
agree. On this corpus, std f32 power stays below 0.507 ULP on Linux x86
and below 0.511 ULP on Apple ARM. See the
[replacement comparison](../benchmarks/powf_module_2026-11-07/REPLACEMENTS.md)
for accuracy/performance tradeoffs against the actual legacy/std methods. See the
[study](../benchmarks/powf_module_2026-11-07/MATRIX.md) for frozen cost gates and
missing performance coverage. The API remains a draft pending its review and
performance evidence.

## Quality and portability presets

All six named presets below are implemented as `PowfOptions` const builder
values. Append `.bits()` for a const entry, or pass the builder value to a runtime
adapter. Every preset states its range, validation and execution policy.
`with_precision`, `with_portability`, `with_domain`, `with_validation` and
`with_execution` specialize those axes through the caller and feature kernel.

```rust
use magetypes::powf::{PowfOptions, PowfValidation, try_powf_with};
const FLAGS: u32 = PowfOptions::BACKEND_GAMMA_LOWP
    .with_validation(PowfValidation::AssumeValid).bits();
let result = try_powf_with::<FLAGS>(0.5, 2.4).unwrap();
assert!(result.is_normal());
```

| Preset | Portability / quality | Base / uniform exponent | Output / validation | Execution |
|---|---|---|---|---|
| `PORTABLE_GAMMA_MIDP` | BitExact / MIDP graph | +0 or [2^-16,2^16] / normal or signed zero in [0,4] | Normal positive or +0; 0^0=1; checked whole-vector rejection, no result-range branch | Automatic |
| `PORTABLE_GAMMA_LOWP` | BitExact / LOWP fused graph | Same Gamma interval and exponent bounds | Same zero semantics; checked whole-vector rejection | Automatic; exact fused fallback where needed |
| `BACKEND_GAMMA_MIDP` | BackendDependent / MIDP graph | Same Gamma interval and exponent bounds | Same zero semantics; checked whole-vector rejection | Automatic; hardware FMA where supported, separate arithmetic otherwise |
| `BACKEND_GAMMA_LOWP` | BackendDependent / LOWP graph | Same Gamma interval and exponent bounds | Same zero semantics; checked whole-vector rejection | Automatic; hardware FMA where supported, separate arithmetic otherwise |
| `PORTABLE_NORMAL_MIDP` | BitExact / MIDP graph | Positive finite normal / finite normal or signed zero | Computed argument [-126,127.5], normal output; checked whole-vector rejection | Automatic |
| `STD_HIGHP` | BackendDependent / HIGHP | Full f32 domain, including negative bases where defined, subnormals, infinities and NaNs | Std-style specials, overflow and underflow; std precision | Scalar std or another measured backend; requires `std` |

The default remains BitExact/MIDP/Checked/PositiveNormal/Automatic. LOWP retains
the legacy coefficients and exp2 clamp; its shorter graph changes accuracy.
All restricted approximation domains require a finite normal-or-zero exponent.
`Full` requires Highp + BackendDependent + the `std` feature. Highp with a
restricted domain enforces the same input rules; PositiveNormal and
NonNegativeNormal check the **std result** is normal (or an admitted zero),
rather than evaluating a second approximation solely to check its argument.
Gamma Highp needs no output check. Full Highp permits all f32 inputs and results.
Unsupported runtime combinations return `UnsupportedOptions`, mask zero.

MIDP and LOWP are quality-family names. `HIGHP` similarly leaves room for a
scalar backend; `ONE_ULP` would be an additional quantitative guarantee that
requires evidence beyond finite startup screening. Direct std delegation
cannot satisfy BitExact merely by passing startup samples.

When a BackendDependent quality/range contract admits both rounding graphs,
its Horner steps can use `mul_add_relaxed`. Strict WASM then uses separate
arithmetic and relaxed WASM may fuse according to its engine. That mode need
not pay a fusion-correction probe/branch merely to force one rounding choice.
BitExact retains its probe and exact fallback. Whole-operation Automatic/Caller
selection remains independent; the const portability choice removes the
unused arithmetic-policy path during monomorphization. Hardware FMA is preferred
where supported: the measured M6 NEON separate graphs cost more than the fused
graphs. The public API remains in a held draft PR pending review and performance evidence.

The normative [WASM relaxed-operation rules](https://webassembly.github.io/spec/core/exec/numerics.html#relaxed-operations)
fix the global `R_fmadd` choice throughout one program execution. For admitted
non-NaN arithmetic, the two uniform graphs therefore cover its fused and
separate choices; this does not assume arbitrary per-node fast-math rewrites.

Validation defaults to `Checked` in every restricted preset. The
`AssumeValid` override removes input and output checks, with the same numerical
preconditions and zero rules. Invalid inputs would have unspecified numerical
results, never permission for memory unsoundness. Width and option validity remain checked. Full HIGHP has no numerical
restrictions to validate, so Checked and AssumeValid intentionally coincide. `Caller` is a separate execution override; choosing it must
not change the selected quality, input domain or validation policy.

The old methods do not supply one uniform preset: LOWP clamps computed exp2
arguments and forces 0^0 to +0, while MIDP has its own zero/overflow handling.
`pow_lowp_unchecked` and `pow_midp_precise` are aliases; their names add neither
a validation choice nor extra precision/range. A replacement must explicitly
choose whether it preserves those legacy special cases or uses the new
checked zero convention above.

The controlled [non-fused study](../benchmarks/powf_module_2026-11-07/NONFUSED-PLAN.md)
compares both rounding paths against independent rational graphs and MPFR.
Gamma's exponent bound is essential: LOWP's log bias near 1 can be magnified
catastrophically by large exponents even when its computed result remains in
normal range. PositiveNormal alone is therefore not a LOWP accuracy preset.

The [public policy protocol](../benchmarks/powf_module_2026-11-07/POLICY-PLAN.md)
compares all 64 supported configurations and records where flags save work,
collapse to identical native graphs, or trade validation/dispatch cost against
arithmetic cost. HIGHP inherits the [std power precision contract](https://doc.rust-lang.org/std/primitive.f32.html#method.powf),
which does not guarantee a fixed ULP bound. Finite probes cannot strengthen it.
