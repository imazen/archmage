# Proposed compile-time power policy

Draft design, not a public API or a change to existing results. The owner proposed
`PowBehavior`, a const `to_u64`, and scalar/vector power operations parameterized
by that packed value (2026-11-06).

## Shape

Use a struct with private bits, typed enum setters, and named presets. A single
enum would either enumerate a combinatorial set of configurations or bundle
unrelated settings. The packed `u64` is the const-generic transport, not the
primary interface callers should manipulate.

Illustrative API (not implemented):

```text
const POLICY: PowBehavior = PowBehavior::PORTABLE_ULP1
    .subnormal_inputs(SubnormalInputs::Preserve)
    .subnormal_outputs(SubnormalOutputs::Preserve)
    .range_hint(PowRange::PositiveNormal);

powf_with::<{ POLICY.to_u64() }>(x, y)
pow_f32x8_with::<{ POLICY.to_u64() }>(xs, ys)
pow_f32_with::<{ POLICY.to_u64() }>(xs, ys)
```

Keep existing default wrappers. Const parameters on free functions do not give
an ordinary `powf(x,y)` call an inferred policy; adding one to the existing
signature would break callers. Put the policy on the operation, not on the
vector type: unrelated addition, logarithms and storage should not acquire a
power-policy type parameter. Scalar and vector entry points share one contract.

`const { assert!(valid(P)) }` at monomorphization should reject unknown bits,
reserved values and unsupported combinations. Validation must also cover
literal `u64` callers that bypass `PowBehavior`. Code generation should contain
only the selected policy branches. CPU feature proof remains separate: dispatch
once per buffer and use matching feature contexts inside the kernel.

## Distinct parts of the contract

| Part | Meaning |
|---|---|
| Input transformation | Preserve/flush subnormals; optional explicit clamping or absolute-value transformation |
| Special cases | Negative bases and integer exponents; signed zero; NaN precedence/canonicalization; infinities |
| Accuracy | Correct rounding, a specified true-ULP bound, or a named approximation with a documented domain |
| Consistency | Identical output bits across supported targets, excluding explicitly unspecified NaN payloads, or backend-dependent results within the accuracy contract |
| Optimization hint | An input/exponent range class selecting a fast path with a defined fallback |
| Output transformation | Preserve/flush subnormals; explicit saturation/clamping; signed-zero normalization |

A range hint is not input clamping. A wrong hint must still reach a defined
fallback. Do not turn a safe hint into `unreachable_unchecked` or silently remove
special-value handling. A preconditioned mathematical kernel, if later needed,
is a separate explicit contract.

Specify the order of input transformation, special-case resolution, core
calculation, and output transformation. For example, `NaN^0`, `1^NaN`, negative
zero to a negative odd integer, and flush-to-zero before a negative exponent
are not independent flags whose order can be left implicit. Input and output
subnormal policies must be separate. A negative-base policy is also distinct
from whether an output zero retains its sign.

Correct rounding of unmodified IEEE power is incompatible with silently
flushing or clamping its result. If transformations are requested, accuracy
must name the transformed function to which it applies. Do not advertise a
full-range bound for an algorithm validated only on a narrow interval.

Bit consistency is stronger than an error bound: two implementations may both
be within one ULP yet return different bits. A bit-consistent approximate preset
must pin its arithmetic/rounding sequence, including FMA choices, or establish
that its alternative paths yield the same answer. Correctly rounded paths can
use different algorithms because the required finite result is unique.

## Encoding and supported presets

Do not allocate permanent bit positions before the semantic matrix is settled.
Use enum-valued fields rather than unrelated booleans where modes are mutually
exclusive. Reserve bits for validation/versioning. Arbitrary pairs of f32 range
endpoints already consume 64 bits, so the initial range field should select
named classes, not pretend to encode arbitrary intervals alongside every other
setting. Treat the packed value as compile-time transport, not a persistent file
format, unless a stable representation is explicitly promised.

Start with a small supported set, not every representable combination:

- Correctly rounded, full range, preserved subnormals and IEEE special cases.
- Portable near-one-ULP, with a documented bound and fixed output bits.
- Named approximation presets with explicit input domains and edge behavior.

These are proposed contract categories, not a claim that every implementation
already meets them. The portable-f32 study evaluates the candidate algorithms.
An ARM-style fast pow path with the complete CORE-MATH fallback is a candidate
for the second category; its measured error is not a global proof. Existing
correctly rounded power supplies the first category. Keep version-specific libc
compatibility contracts separately identified.

Only instantiate policies callers use. Avoid eagerly compiling the Cartesian
product of every setting and CPU tier. Review compile time, binary size,
scalar/vector bit parity, cross-target outputs, ordinary-input throughput,
exception-heavy inputs, and worst-case error per supported preset. A packed
policy makes branch removal possible; it does not make arbitrary combinations
accurate, portable, or fast.
