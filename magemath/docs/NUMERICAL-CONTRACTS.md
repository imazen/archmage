# Composable numerical contracts

Design proposal, 2026-11-07. This page proposes a successor to parts of
[PowBehavior](POW_BEHAVIOR.md); none of the API spellings below are implemented.
It does not change existing numerical contracts or the deprecated Magetypes
methods. Source review baseline: `d6415bed` on the Zenmath draft.

## Recommendation

Use typed const builders with two independent const-generic words: a shared
floating-point policy and operation-specific options. Put policy on operations,
not vector storage types. Keep operation meaning, admitted inputs, promised
results, and implementation choices distinguishable.

The core model is: **for each admitted input, specify the set of permitted
results**. An implementation is eligible when every result it can produce is in
that set. Faster hardware does not change the set. A narrower input domain can
make a cheap instruction eligible without relaxing the result at all.

Some requirements also relate multiple evaluations: monotonicity compares
different inputs, and reproducibility compares targets or repeated calls.
Formally the contract contains an input domain, allowed results per input, and
relations across evaluations. Membership in the same two-answer set does not
prove two targets choose the same answer. Fixed implementation choices and
cross-target identity therefore need tests beyond per-call set membership.

This model also covers conversions, comparisons, masks, shuffles, reductions,
quantized dot products and approximate reciprocals. These operations share a
contract vocabulary; they need not share every field or one enormous enum.

## Five questions, not one fast-math switch

| Question | Examples | Where it belongs |
|---|---|---|
| What function are we computing? | Ordinary power, power of magnitude, signed power; saturating conversion | Operation name or operation options |
| Which inputs are admitted? | Positive finite base; integer exponent; canonical mask | Per-operand domain, including relations between operands |
| Which answers are acceptable? | Correct rounding; a proved error bound; exact zero anchors; matching bits | Result requirements |
| Which evaluations are permitted? | Fixed recipe; fused or separate madd; reordered reduction | Explicit evaluation permissions, subject to result requirements |
| How is it executed? | Scalar/packed; CPU token; runtime dispatch; table layout | Implementation selection |

An output requirement dominates an evaluation permission. Permission to fuse
does not license a result outside the requested error bound. Conversely, a
correctly rounded function may use different internal graphs on different CPUs
if each graph certifies the same final result.

Avoid an enum ordered `low < medium < high < portable`. Accuracy,
reproducibility, domain coverage and cost are independent. An approximation can
be bit-reproducible; a tightly bounded approximation can vary by ISA. A correctly
rounded finite result says nothing about NaN payloads.

## What we should borrow from WebAssembly

The [linked SIMD document](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/simd/SIMD.md)
describes the strict instruction set. Its explicit names distinguish truncating
saturation, rounding, min/max, and pseudo-min/max. In particular, `pmin` selects
`b < a ? b : a`; it is operand-order-sensitive for NaNs and ties. Treat such
behavior as an operation definition, not a generic accuracy setting.

The [relaxed SIMD overview](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
identifies specific choices for madd, lane selection, conversion, swizzle,
min/max and integer dot products. It does not provide a blanket permission to
reassociate every expression. Approximate reciprocal and reduction appear in
its motivation, not its final instruction list. Do not infer available Wasm
instructions from that motivation.

The current [core numeric semantics](https://webassembly.github.io/spec/core/exec/numerics.html#relaxed-operations)
make implementation choices fixed during program execution, but the selected
operation can still be nondeterministic. Relaxed madd permits fused or separate
evaluation; the deterministic profile selects separate evaluation. Relaxed
conversion outside its defined input range can admit arbitrary integer results,
so a historical hardware sentinel is not a portable contract.

The [profiles model](https://webassembly.github.io/spec/core/appendix/profiles.html)
is particularly useful: a profile restricts permitted behavior, and compatible
profiles intersect. Its deterministic profile also constrains arithmetic NaNs;
ordinary Wasm execution does not automatically have that guarantee. We should
make reproducibility an explicit result requirement, rather than infer it from
the presence of Wasm or a fixed engine choice.

Practical candidate optimizations inspired by those instruction families:

| Operation | Facts worth proving | Candidate benefit |
|---|---|---|
| Float-to-integer truncation | Finite value whose truncated result is representable | Remove NaN/out-of-range fixups |
| Lane selection | Every mask lane is all-zero or all-one | Hardware blend and bitselect agree |
| Byte swizzle | Every index is below the source lane count | Remove out-of-range correction |
| Min/max | No NaNs and no opposite-signed zero ties | Several hardware forms agree |
| Q15 rounded multiply | Exclude the minimum-times-minimum overflow case | Remove that saturation correction |
| Quantized dot | Second operand in nonnegative 7-bit range; partial sums in range | Avoid signedness/saturation differences |

These are eligibility conditions to validate against each lowering, not claimed
measurements or new APIs. For dot products, specify intermediate widths and final
wrapping/saturation separately. For mask operations, a numeric `1` is not an
all-one mask. The most valuable relaxation is often a domain restriction that
leaves exactly one acceptable result.

## Rust's baseline is not a precision preset

[Rust's float documentation](https://doc.rust-lang.org/std/primitive.f32.html)
distinguishes correctly rounded basic arithmetic, `sqrt` and `mul_add` from
transcendentals with unspecified precision. It also permits arithmetic NaN bit
patterns to vary between evaluations. Bit-preserving operations such as `abs`
have different guarantees. Thus “like std” is neither a universal one-ULP bound
nor a bit-reproducibility promise.

Rust's `algebraic_*` methods allow much broader transformations and promise no
particular individual result beyond a valid float. Finite inputs alone do not
turn those methods into a bounded-error implementation. Keep an explicitly
unbounded algebraic surface separate; do not implement `MaxUlp(1)` by enabling
algebraic arithmetic.

Our proposed contracts assume round-to-nearest, ties-to-even and gradual
underflow unless explicitly stated otherwise. We should not silently change a
thread's floating-point environment. Supporting a foreign environment requires
an explicit entry contract, correction or a software implementation. This is a
library contract decision, not a claim that every nondefault environment is Rust
undefined behavior. Floating exception flags and traps are not part of the
proposed initial API.

## What can be shared

Use a small `FloatPolicy` vocabulary for:

- **Reproducibility:** backend-dependent answers within the stated contract,
  or matching non-NaN bits across supported ISAs, widths and scalar tails.
- **NaN result encoding:** unspecified payload/sign, or a specified canonical
  quiet NaN. This applies only after the operation decides its result is NaN.
- **Zero result encoding:** preserve the operation's zero-sign rule, or
  explicitly canonicalize zero to positive zero at the output boundary.
- **Subnormal result handling:** preserve, or explicitly flush to signed zero
  at the output boundary.

Keep the default operation's value semantics intact. Output zero/subnormal
transforms define a different, explicitly transformed operation: accuracy is
checked against that transformed reference, not advertised as IEEE correct
rounding of the original function. They never authorize flushing intermediates.

Require canonical NaNs as well as reproducible non-NaN bits when the caller asks
for complete bit identity. Preserving signed zeros is compatible with complete
identity if the operation fixes their signs. It is unnecessary to erase them.

Do not share a universal `HandleNaN`, `HandleInfinity`, `Positive`, or
`NoZeroCrossing` boolean. Their meanings are operation-specific:

| Concern | Why a single switch is insufficient |
|---|---|
| NaN inputs | An operation may have identities that suppress an input NaN; canonicalizing a NaN result must not change those identities |
| Infinite inputs | `exp(-inf)`, `log(+inf)` and an invalid subtraction have different results; finite inputs can still overflow |
| Signed zero | Division, branch-sensitive functions, rounding and odd powers use its sign differently |
| Subnormal inputs | Pow's base and exponent have different roles; flushing a tiny exponent to zero can change the result to one |
| Intermediate underflow | Normal inputs do not imply normal intermediates or outputs |
| Crossing zero | Positive inputs do not prevent a logarithm crossing zero at one; a small relative error is not useful at a root |

Input transformations should be named adapters or explicit per-operand options.
`abs_then_pow`, signed power and clamped power are useful functions, but none is
ordinary power with a vaguely weaker accuracy promise. Their order matters:
clamp, reject infinities, and take absolute value do not commute.

## Operation-specific requirements

Each operation builder should reuse small typed components where they mean the
same thing: interval endpoints, endpoint inclusion, finite/normal classification,
error requirements and outside-domain handling. Each operation maps these onto
its actual operands and validates the combinations.

Accuracy should distinguish:

1. Correctly rounded output with the operation's specified special cases.
2. A proved `MaxUlp(k)` or absolute/relative envelope on a named domain.
3. An explicitly named legacy recipe, whose current evidence and limitations
   remain attached to that recipe.
4. Unbounded algebraic evaluation, with no invented numerical error promise.

Do not expose a bounded option before an eligible implementation or stronger
fallback exists. Define ULP distance against the correctly rounded finite
reference, with zero, underflow, overflow and NaN rules separately. For an
absolute/relative envelope, state the actual inequality, for example
`abs(y - f(x)) <= abs_limit + rel_limit * abs(f(x))`, evaluated mathematically.
Special values are not covered by that inequality.

Monotonicity, preservation of an exact root, endpoint values and output range
are separate requirements. Small error does not establish any of them. A future
“zero crossing” component should name an interval and its root/anchor obligations,
not assert that all positive inputs have positive outputs.

Recipe settings belong here too. `Madd::Fused`, `Madd::Separate` and
`Madd::Relaxed` are useful for an explicit madd operation. For a transcendental,
the implementation's certificate determines its internal arithmetic; callers
normally select the output contract. A caller who needs an old polynomial's
exact bits chooses a versioned recipe. libc compatibility likewise needs a named
implementation/version/target recipe, not `Accuracy::Libc`.

## Proposed const-generic shape

The following is syntax-only design notation, not a runnable example:

```text
const IMAGE: u64 = FloatPolicy::DEFAULT
    .reproducibility(Reproducibility::CrossIsa)
    .nan_result(NaNResult::Canonical)
    .to_u64();

const GAMMA: u64 = PowOptions::DEFAULT
    .accuracy(Accuracy::CorrectlyRounded)
    .domain(PowDomain::SrgbDecode)
    .outside(Outside::Fallback)
    .to_u64();

type ImageMath = Math<IMAGE>;
let y = ImageMath::pow::<GAMMA>(x, exponent);

// Lower-level shape; the same C and O reach scalar, array and token entry points.
fn pow_with<const C: u64, const O: u64>(x: f32, y: f32) -> f32;
fn kernel<const C: u64, const O: u64>(...) {
    ... pow_with::<C, O>(x, y) ...
}
```

`PowDomain::SrgbDecode` would include the exact admitted exponent as well as the
base interval. Wrong exponent or out-of-range base takes the complete fallback
in this example. `Math<C>` is a zero-sized namespace, not a new vector type.
Token entry points retain their normal feature context and can use intrinsics
without depending on Magetypes. Magemath supplies the vector adapters.

[Stable const generics](https://doc.rust-lang.org/reference/items/generics.html#const-generics)
accept integer parameters, not these builder structs or floating-point values.
Two independent words can be forwarded unchanged through generic code. Avoid a
design requiring `::<{ P & MASK }>` or other calculations on a generic parameter
in const arguments. Concrete const builders can perform validation and packing.
Arbitrary interval endpoints or fixed exponents can later use separate bit-valued
const arguments; do not force every possible domain into a 64-bit word.

Reserve an operation/schema tag in operation words, reject unknown bits and
invalid combinations, and validate literal integer callers as well as builders.
The encoding is not a serialization format. Never ask users to OR raw policy
words. Builders normalize equivalent encodings; irrelevant settings are rejected
or explicitly documented as such rather than creating accidental specializations.

Two words are slightly noisier than one. The namespace and named presets hide
that in ordinary code, while generic pipelines can share `C` and retain different
operation options. Retain `PowBehavior` as a compatibility adapter with its
existing encoding and behavior; do not silently reinterpret previously compiled
constants. Existing named functions remain convenient entry points.

## Composition and checked domains

Requirements compose by intersection of acceptable results. A combined
requirement cannot silently weaken either input requirement. Incompatible
recipes, impossible domains and unsupported combinations are compile-time errors.
Explicit setters replace a setting; a separate `require` operation strengthens
requirements and rejects conflicts. Those are different operations.

Input-domain intersection restricts the caller's valid inputs; it does not prove
those inputs exist at runtime. Nor can two function contracts be composed by
blindly merging options: the first function's output must satisfy the second's
input domain. Error through a pipeline depends on conditioning and intermediate
rounding. Per-call one-ULP errors do not imply a one-ULP pipeline.

Offer two clearly distinguished outside-domain behaviors initially:

- `Fallback`: classify, use the specialized kernel when admitted, otherwise use
  a complete implementation. The public function retains its full-domain contract.
- `RequireInputs`: the numerical guarantee is conditional on the domain. Optional
  debug assertions check original inputs. Every input must nevertheless remain
  memory-safe; a violation cannot justify unchecked indexing, an unchecked cast,
  or an optimizer assumption that could cause undefined behavior.

If a bounded range is only an optimization hint, use `Fallback`. If a consumer
can establish a range once for a whole buffer, a validated immutable view may
amortize checks. Its proof must be invalidated by mutation and must cover every
needed predicate. A CPU token proves instructions are available, not that data
is finite or in range.

For vectors, specify per-lane results independently of neighbors. All-lanes
fast-path classification and per-lane rescue are implementation choices. Masking
the final result does not prevent invalid lanes from executing earlier arithmetic;
sanitize lanes before table indices, conversions or other constrained operations.

## Reproducibility beyond elementwise functions

Cross-ISA reproduction must cover width, tail handling and feature dispatch, not
just whether FMA is enabled. For reductions, define a tree independent of SIMD
width and threading, or offer an exact accumulator when order independence is
required. A fixed tree, compensated summation and an exact accumulator are three
different contracts. None follows automatically from “portable FMA.”

Separate current cross-target reproducibility from persistence across library
releases. The latter requires a named versioned recipe or a unique mathematical
answer plus fully specified exceptional encodings. Do not freeze every default
approximation's coefficients accidentally.

## Gamma is the first concrete example

The existing [certified gamma paths](../../zenmath/docs/GAMMA.md) already
demonstrate domain specialization without weakening correct rounding. Their
exponent bits are `0x4019999a` and `0x3ed55555`; these are f32 arguments, not the
exact real rational numbers 12/5 and 5/12. That distinction is part of the
operation definition.

The scalar piecewise kernel and packed global kernels use different polynomial
graphs. Their bounds certify the output or send it to CORE rescue. Preserve
those graph-specific certificates. A universal “force FMA everywhere” flag would
invalidate this evidence even though it sounds more accurate.

For a linear-sRGB caller, first determine whether the requirement is ordinary
f32 `powf` at those exponent bits, a real-rational transfer curve, or identical
final quantized pixels. They are three different reference functions/contracts.
If the last is sufficient, investigate certifying that an interval lies entirely
inside one quantization bin. That could avoid demanding a uniquely rounded
intermediate float. It still requires exact threshold, tie, clipping and
whole-transfer analysis; this is a proposed optimization, not a measured result.

Keep the linear branch, breakpoint and continuity/monotonicity requirements with
the transfer function. A power-only domain cannot prove the whole curve's
behavior. Preserve current bit contracts while measuring any alternate contract.

## Migration and crate boundaries

| Current draft field/surface | Proposed destination |
|---|---|
| `PowPrecision::Midp/Lowp` | Named recipes until independent accuracy bounds justify stronger names |
| `PowConsistency::PortableFused/PortableSeparate` | Separate reproducibility requirement and recipe arithmetic |
| `PowRange::ClampUnitInterval` | Explicit base transformation |
| `PowRange::PositiveNormalPrecondition` | Per-operand domain plus outside-domain behavior |
| `PowSign::Magnitude/SignedMagnitude` | Explicit alternate power meaning |
| Input subnormal flushing | Per-operand transformation |
| NaN/output zero/subnormal encoding | Shared boundary vocabulary |
| `portable`, `relaxed`, `unfused` | Preserved named recipe surfaces and convenience adapters |

Initially keep builders in Zenmath and reexport them from Magemath. Magetypes
keeps its explicitly named arithmetic primitives; it need not depend on this
proposal. Zenmath and Magetypes can compile independently, with Magemath joining
them. If shared policies later belong on Magetypes primitives, extract a tiny
dependency-free contract crate after validating the need and compile cost.
Do not introduce that extra crate merely to complete a diagram.

Keep [correct-rounding, normal and configurable availability](FEATURES.md)
separate. Cargo feature unification must never silently change the numerical
meaning of an existing call. Private accurate rescue can remain available when
its public family is disabled. A policy does not imply an Archmage dependency.

## Evidence required before implementation graduates

1. Maintain an eligibility table: operation, domain, recipe, evaluation graph,
   output encoding, proved accuracy, target coverage and reference identity.
   Unsupported requests fail or select a stronger verified implementation.
2. Test against independent MPFR/exact-integer references. Relaxed operations
   require membership in the allowed result set, not equality to one machine's
   std result. Test identities, signed zeros, NaNs, subnormals, overflow and roots.
3. Preserve raw and canonical-NaN digests separately for exhaustive unary f32
   checks. Fixed-exponent power sweeps are not exhaustive coverage of all pairs.
4. Check all widths, mixed lanes, tails, compile-time evaluation where supported,
   and strict/relaxed Wasm with fusing and non-fusing behavior. Wasmtime exposes
   [relaxed-SIMD determinism](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#method.relaxed_simd_deterministic)
   separately from [NaN canonicalization](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#method.cranelift_nan_canonicalization);
   do not treat one switch as proof of both properties.
5. Use zenbench paired/interleaved comparisons for complete consumer loops and
   scalar/vector entry points. Include check placement, rescue frequency, short
   buffers and dispatch cost. Measure cold compile time and binary size as well
   as runtime: const-policy combinations can multiply monomorphizations.
6. Inspect emitted code before attributing a win to a permission. Domain checks
   do not automatically teach LLVM an algebraic fact. Select an eligible kernel
   explicitly where needed; do not introduce unsafe assumptions to force it.

No new performance measurements accompany this proposal. Existing evidence is
linked from the [math matrix](../../benchmarks/math_matrix_2026-11-06/README.md),
[rounding comparison](../../benchmarks/magemath_rounding_cost_2026-11-05/README.md)
and [power optimization report](../../benchmarks/zenmath_pow_opt_2026-11-06/RESULTS.md).
Start implementation with shared boundary policy plus the existing power
adapter, then validate reuse on log/exp and madd before expanding the schema.
