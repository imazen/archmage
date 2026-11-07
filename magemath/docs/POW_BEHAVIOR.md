# Compile-time power behavior (draft, implemented)

The [composable-contract proposal](NUMERICAL-CONTRACTS.md) reviews which parts of
this API can be shared with other operations. It does not change the implemented
encoding or behavior documented here.

Unpublished API in zenmath/magemath. Existing `powf`, array and slice functions
keep their signatures and output-bit contracts. The new policy belongs to the
operation, not the vector type. The packed `u64` is compile-time transport, not
a stable serialization format.

```rust
use magemath::{PowBehavior, PowSign};
use magemath::portable::{array, slice};

const P: u64 = PowBehavior::CORRECT
    .sign(PowSign::SignedMagnitude)
    .to_u64();

assert_eq!(zenmath::scalar::powf_with::<P>(-4.0, 0.5), -2.0);
assert_eq!(array::pow_f32x8_with::<P>([-4.0; 8], [0.5; 8]), [-2.0; 8]);
let mut values = [-4.0; 11];
slice::pow_f32_with::<P>(&mut values, &[0.5; 11]);
assert_eq!(values, [-2.0; 11]);
```

`PowBehavior` and its typed enums are the same reexported types in both crates.
`CORRECT`, `MIDP`, and `LOWP` are the presets. `to_u64` validates the final builder;
literal const callers also undergo validation at monomorphization. Unknown bits,
reserved enum values and unsupported combinations are compile errors. There is
no near-one-ULP preset: the experimental ARM hybrid has measured evidence, not a
new global bound, and remains outside this API.

| Setter | Supported modes | Meaning |
|---|---|---|
| `precision` | `CorrectlyRounded`, `Midp`, `Lowp` | CORE or the existing named polynomial with the rescue rules below |
| `consistency` | `PortableFused`, `PortableSeparate` | Fixed one-rounding or two-rounding polynomial steps on every target; Separate is approximation-only |
| `sign` | `Mathematical`, `Magnitude`, `SignedMagnitude` | Ordinary negative-base parity, absolute base, or explicit signed-power extension |
| `subnormal_inputs` | `Preserve`, `FlushSignedZero` | Applies to both base and exponent, independently of output handling |
| `subnormal_outputs` | `Preserve`, `FlushSignedZero` | Preserve or flush a subnormal result without losing its sign |
| `nan` | `CoreMath`, `CanonicalResult` | CORE special-case precedence; optional final canonical bits `0x7fc00000` |
| `infinity` | `Preserve`, `RejectInputsToNaN` | Ordinary semantics or reject either transformed infinite operand, including identity cases |
| `range` | `Full`, `ClampUnitInterval`, `PositiveNormalPrecondition` | Full handling, explicit base clamp, or a separate approximation precondition |
| `output` | `Preserve`, `SaturateFinite`, `ClampUnitInterval` | Preserve, map infinities to signed MAX, or clamp to `[0,1]`; retain NaNs |
| `debug_assertions` | `false`, `true` | Original-input precondition assertion in debug builds; only valid with the precondition range |

CPU tier is independent from policy. Both rounding sequences are portable,
excluding unspecified NaN payload/sign. No backend-relaxed mode or extra runtime
upgrade was added. Correct rounding requires the CORE evaluation sequence, so
`CorrectlyRounded + PortableSeparate` is rejected.

## Order and special values

1. If requested, clamp the base to `[0,1]`. Ordered comparisons preserve NaN and
   negative zero; negative finite values clamp to +0, positive infinity to 1.
2. If requested, flush subnormal base and exponent to their signed zeros.
3. Apply input-infinity rejection to those transformed operands.
4. For Magnitude/SignedMagnitude, take the absolute base and retain its sign bit.
5. Resolve power and special cases using the rules below.
6. SignedMagnitude restores the transformed base's sign to any non-NaN result.
7. Apply output subnormal flushing, then output saturation/clamping.
8. Optionally canonicalize a NaN result.

Optional debug checks inspect the **original** operands before these steps.
Their entire calculation is under `cfg(debug_assertions)`, including its log2,
so release builds do not evaluate it even with optimization disabled.

`Mathematical` is ordinary power: a finite negative nonzero base needs an integer
exponent; only an odd integer makes a negative result. Every representable finite
f32 integer at magnitude >=2^24 is even. The implementation examines exponent and
fraction bits; it never unconditionally copies the negative base's sign.

`Magnitude` computes `pow(abs(x), y)`. `SignedMagnitude` explicitly extends this to
`signbit(x) | pow(abs(x), y)` for a non-NaN result. It is intentionally different:
`(-4)^0.5 = -2`, `(-2)^4 = -16`, and `(-4)^0 = -1` under this extension. It also
restores negative zero/infinity. A NaN result stays NaN. Identity resolution still
happens before sign restoration, so even a negative quiet-NaN base to exponent0
has its finite identity result's sign restored. This is not mathematical pow.

With the default policy: `0^0 = 1`, `1^qNaN = 1`, but a signaling NaN in those
identity positions propagates NaN under the existing CORE contract. `(-0)^-3 = -inf`,
`(-0)^2 = +0`, `(-inf)^-3 = -0`, and negative noninteger finite bases return NaN.
Input-infinity rejection deliberately overrides identities such as `inf^0`.
Output transformations deliberately change IEEE results; accuracy then names the
transformed function, not unmodified mathematical power.

## Approximation and range semantics

`Full` and `ClampUnitInterval` use the selected polynomial only for normal
magnitudes other than1, finite exponents other than0/1, mathematical negative
bases with integer exponents, and the selected approximate log2-product in
`[-126,127.5]`. All other lanes use complete CORE power on the transformed operands.
Mathematical negative fast lanes restore their sign only for odd integer exponents.

This exposes existing named approximation accuracy, not a new uniform error bound.
The original `pow_midp_f32`/`pow_lowp_f32` APIs retain their own edge behavior;
policy Full adds CORE special/out-of-band rescue and therefore is not generally
bit-identical to those legacy formulas outside the working inputs.

`PositiveNormalPrecondition` is explicitly different. It requires original base
positive and normal, exponent finite, and the selected log2-product in
`[-126,127.5]`. It is approximation-only and rejects sign, denormal, input-infinity
and output transformations that conflict with that contract. NaN-result
canonicalization and fixed rounding selection remain available.

With `debug_assertions(true)`, invalid original operands panic in a debug build.
In release, or with that option false, execution evaluates exactly the existing
named checked approximation formula without the outer CORE classifier/fallback,
**also on violations**. Thus legacy lowp at `(0,0)` returns +0, while the ordinary
Full policy returns1. Negative invalid bases produce the named formula's NaN;
its documented exponent clamping applies. The formula is safe Rust with defined
special handling, not `unreachable_unchecked` or a memory-safety precondition.
Inner primitive special-value masks can remain; the mode does not promise that
every classification instruction disappears.

There is no range-hint field with no effect. A future safe hint would require a
measured alternative path plus defined fallback; an explicit clamp is a semantic
transformation, not that hint.

## Packed implementation and entry points

- `zenmath::scalar::powf_with::<P>(x,y)` is the scalar implementation.
- `magemath::portable::array::pow_f32x8_with::<P>(x,y)` uses eight independent
  exponents; correctly rounded work uses two packed f64x4 CORE calls.
- `magemath::portable::slice::pow_f32_with::<P>(data,exponents)` checks matching
  lengths and dispatches once per buffer. Tail filler is the valid pair(1,0).

Both array and slice functions have the existing attune family: dispatcher,
`_v3_t`/`_neon_t`/`_wasm128_t` proof wrappers, and direct `_v3`/`_neon`/`_wasm128`
matching-feature entries. Feature-gated V4x follows the existing convention.
Direct matching-feature kernels add no CPU probe. The array dispatcher selects
features per call; the slice dispatcher selects once per buffer. Shared preparation and postprocessing
live in zenmath's doc-hidden same-workspace bridge to prevent scalar/vector
precedence from drifting.

All correctly rounded policies feed transformed inputs to the packed CORE kernel,
then postprocess results; they are not scalar mappings. Midp/Lowp use packed
log2 × per-lane exponent → exp2 with Fused or Unfused polynomial steps, and scalar
CORE rescue only for exceptional/out-of-band lanes. The default policy retains
the original packed path with compile-time policy branches removed.

Packed CORE now admits negative finite bases with integer exponents, evaluates
absolute magnitude and restores sign bitwise for odd exponents. Specials, x=±1,
y=0/1, range exceptions and ambiguous rounding keep the original complete fallback.
Parity work is guarded by whether any lane is negative. This optimization is
bit-preserving, not an unconditional speedup: paired performance and source hashes
are recorded in the [policy measurements](../../benchmarks/pow_behavior_2026-11-06/README.md).

The numerical environment remains nearest-even and gradual underflow before
explicit policy transformations. errno, exception flags, dynamic rounding modes
and unspecified NaN payloads are outside this draft contract.
