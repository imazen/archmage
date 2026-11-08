# Bounded portable f32 power

`magetypes::powf` owns one fused approximation graph for scalar and vector
power. `f32x1`, generic `f32x4`, `f32x8` and enabled `f32x16` delegate to it.
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
`PowfOptions::new().with_domain(...).with_execution(...).bits()`.
The flags remain const through vector delegates,
backend selection and the graph. A capable caller evaluates the vector directly
in its feature region, without an array dispatch boundary. Reserved encodings
are compile errors. Runtime input checks still enforce the declared domain.

For options selected at runtime, `try_powf_runtime` and
`try_powf_array_runtime` retain the explicit `PowfOptions { domain, execution }`
adapter. These APIs share the graph and rejection contract. The array const
entry is `try_powf_array_with::<OPTIONS, N>`; the vector method is
`vector.try_powf_with::<OPTIONS>(exponent)`.

The contract is identical result bits across admitted backends for the same
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
They cannot certify direct `std::powf` against the fixed graph, so the module
does not delegate to `std::powf`.

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
