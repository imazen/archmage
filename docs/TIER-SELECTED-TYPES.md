# Tier-selected type aliases

`use(f32xN)` selects a body-local contextual vector alias for the declared tier.
`use(f32x8)` keeps eight lanes. Both forms work on all seven archmage attribute
entry points listed below. The [constructor migration](TOKEN-CONTEXT-MIGRATION.md)
describes `_with_token` alternatives and compatibility with `define(...)`.

```rust
use archmage::{autoversion, rite};

#[rite(v3, neon, wasm128, default, use(f32xN))]
fn add_row(values: &mut [f32]) {
    let (chunks, tail) = f32xN::partition_slice_mut(values);
    for chunk in chunks {
        (f32xN::load(chunk) + f32xN::splat(1.0)).store(chunk);
    }
    for value in tail { *value += 1.0; }
}

#[autoversion(v3, neon, wasm128, default, use(f32xN))]
fn lanes() -> usize { f32xN::LANES }
```

No token argument is needed for the short constructors inside the matching
feature context. A nested ordinary function does not inherit that context;
use `_with_token` there or give the helper its own `rite` attribute.

## Existing mechanisms and history

Adaptive widths are not a new concept here. The initial review omitted surviving
APIs; `use(f32xN)` extends their convention.

- [`2ef97cd`](https://github.com/imazen/archmage/commit/2ef97cd8f2e82bb4d7789b9b87b3b1c3ad3b79d5)
  introduced module-level `#[multiwidth]` with width-specific namespaces and
  dispatchers only for signatures independent of width. It was removed in
  [`88428cd`](https://github.com/imazen/archmage/commit/88428cd8d76f7c78f38ef5f80f7ec5cfcfde0e62).
- [`0d97db9`](https://github.com/imazen/archmage/commit/0d97db914d57c49efdc94b85cc1d97ef3d02961a)
  introduced function-level `#[magetypes]` substitution of `f32xN`, other vector
  families, and `LANES` constants. Those substitutions were removed on February
  5, 2026 in [`36c8caf`](https://github.com/imazen/archmage/commit/36c8caf56274959a5ac5ea9cf5c23d5f47c4ace1),
  which retained `Token` substitution and moved toward explicit fixed widths.
- The platform-selected prelude `F32Vec`, `RecommendedToken`, and lane constants
  were separately removed in 0.9.27 by
  [`9ac462d`](https://github.com/imazen/archmage/commit/9ac462d32b3d52f17f9fc1d13ad55ca5a68ca850).
  That selection used target architecture and Cargo features, not each runtime
  dispatched variant's tier.
- **Still present:** `magetypes::simd::{v3,v4,v4x,neon,wasm128}::f32xN`
  and the other `xN` families. Their widths are 8/16/16/4/4 f32 lanes.
  These are ordinary explicit-token aliases generated in
  `magetypes/src/simd/generated/mod.rs` by `xtask/src/simd_types/mod.rs`.
  `rite(import_magetypes)` and `arcane(import_magetypes)` import the resolved
  tier namespace through `generate_imports` in `archmage-macros/src/common.rs`.
- **Also present:** `magetypes::SimdTypes` in `magetypes/src/types.rs`, associating
  tokens with vector types and lane constants. Its scalar mapping uses the
  standalone x1 wrappers, unlike the generator's shadowed scalar-x4 aliases. Its
  associated types have no operation bounds, so `T: SimdTypes` alone does not
  expose a uniform generic constructor API. Its V2 mapping currently refers to
  V3-backed vectors; it must not be blindly reused as a capability resolver.

The generator also emits scalar-x4 natural aliases, but its private
`generated::scalar` module is shadowed by the public standalone `simd::scalar`
module. Those generated scalar aliases are **not** publicly reachable as
`simd::scalar::f32xN`. Contextual `use(f32xN)` now selects scalar x4 directly;
it does not depend on that shadowed module.

## Select existing types at macro expansion

Use the variant's declared tier, not the build host or runtime slice length.
Resolve an `xN` family name to a concrete existing contextual alias:

| Variant | `f32xN` | Logical width |
|---|---|---|
| V3 | `generic::local::f32x8<X64V3Token>` | 256 bits |
| V4 / V4x | `generic::local::f32x16<that token>` | 512 bits |
| NEON | `generic::local::f32x4<NeonToken>` | 128 bits |
| WASM SIMD128 | `generic::local::f32x4<Wasm128Token>` | 128 bits |
| Scalar / default | `generic::local::f32x4<ScalarToken>` | Four scalar lanes |

A scalar-x1 policy is possible but is additional API work: today's standalone
`simd::scalar::f32x1` is a different token-taking wrapper, has no
`partition_slice[_mut]`, and does not implement the generic constructor-mode
surface. It is not a drop-in contextual alias. Starting with scalar x4 reuses
existing APIs and correctness coverage; it must be documented as four lanes.
A later x1 backend needs generated API parity and measurements of its own.

`use(f32x8)` keeps its fixed eight lanes everywhere, including polyfills. Adaptive
selection does not imply widest-is-fastest for every workload or CPU. Users can
keep an explicit width. V4 needs the consumer's feature gate forwarded correctly;
the probe declares its `avx512` feature as well as enabling the dependencies.

Adaptive aliases require a complete implemented backend: V3, V4, V4x, NEON,
WASM SIMD128, or ScalarToken. V1/V2, crypto/extended ARM tokens, relaxed WASM,
and FP16 tokens produce a diagnostic for `xN`. There is no implicit downgrade.
Fixed-width aliases retain their concrete token; Rust reports missing backend
traits where a particular shape is unsupported.

`vector_aliases.rs` shares parsing and alias emission across macros. Its shape
roster is generated from `all_simd_types()`; natural widths come from explicit
`magetypes_width` entries in `token-registry.toml`. Namespace imports are not
used as capability evidence. This adds no public vector family, backend trait,
constructor implementation, or forwarding layer.

## Partitions, rows, and tails

The existing `partition_slice` returns `(&[[f32; N]], &[f32])`; its mutable form
returns the corresponding disjoint mutable views. Selecting the type selects
`N`, and `f32xN::LANES` is a concrete associated constant in each expansion.
`load` accepts exactly the partition's fixed-size array. No extra iterator,
allocation, bounds policy, or unsafe block is required for width selection.

Use one bulk loop and a scalar remainder loop. The remainder length is always
less than `LANES`; never call `from_slice` on it unless a full vector is present.
The current `from_slice` panics for a short input; it is not a masked tail load.
If a vector-tail helper is later added, define its fill value, valid lane count,
and bounded store explicitly. Zero-padding alone is wrong for operations such
as minima, division, or reductions with a nonzero neutral element.

For paired buffers, check matching logical lengths before partitioning; blindly
zipping unequal chunk counts discards work. For images, accept width, row count,
and stride and partition each logical row independently. Padding remains
untouched. A contiguous path can process multiple rows together only when the
operation permits it. The compile probe checks all widths 0 through 65, offsets
0 through 3, and three rows with padding for each selected backend it executes.

Changing lane count can change floating-point reduction order. Do not promise
bit-identical reductions from width selection. Pixel algorithms needing exact
output must keep a specified arithmetic order or validate a width-independent
algorithm. The probe uses pointwise addition and exact output comparisons;
it is not evidence about reductions or transcendental accuracy.

Related aliases should use a consistent bit width: `f32xN` and `i32xN` have equal
lane counts; `f64xN` has half as many, and `u8xN` four times as many. Fixed-width
names remain necessary for fixed-format blocks, shuffles with literal indices,
width-specific method names, and conversions whose source/destination lane
counts differ. Do not rename those operations implicitly.

## All seven attribute entry points

| Attribute | `use(...)` resolution |
|---|---|
| `magetypes` | Each generated tier; `Token` substitution remains unchanged |
| `rite` | Explicit single/multiple tiers, or a recognized concrete token parameter |
| `token_target_features` | Same parser and implementation as `rite` |
| `arcane` | Concrete token parameter; aliases enter the feature-enabled body |
| `token_target_features_boundary` | Same implementation as `arcane` |
| `simd_fn` | Legacy alias of `arcane` |
| `autoversion` | Each generated tier, including scalar/default and the `cfg(feature)` fallback |

Both sibling and nested `arcane` forms support aliases. `arcane` still needs a
capability proof; importing a vector alias does not supply one. With
`magetypes(rite, ...)`, the generated attribute now receives the known tier,
so tokenless bodies work. `rite` itself does not substitute a `Token` placeholder;
use its tier-based tokenless form or a concrete token parameter.

A feature bound is a minimum capability, not a unique backend identity.
Without an explicit tier or recognized concrete token, `use(...)` produces a
diagnostic. Genuine `T: Backend` helpers retain `T` and use `_with_token`.

Aliases are local to the body. Short names in parameters/results and automatic
signature rewriting are **not implemented**. Use explicit types in signatures.
A runtime dispatcher must have a tier-independent signature: scalar values,
slices, fixed arrays independent of the tier, or another uniform representation.
Rust's built-in attributes are not extended by this crate.

## Capability matrix and unification review (2026-09-27)

Token-free constructors remove the need to thread a token value through a kernel.
They do not remove the concrete backend type or the proof needed to enter SIMD
code. A tier annotation specifies which features a function requires; it does
not test the executing CPU. Today the four attributes divide that work as follows.

| Capability | `autoversion` | `rite` | `arcane` | `magetypes` |
|---|---|---|---|---|
| Main job | Runtime dispatcher plus private variants | Feature-annotated helper(s) | Token-proven entry wrapper | Variant generation; dispatch supplied separately |
| Select tier explicitly in attribute | Yes, tier list | Yes, one or several | No; reads token parameter / feature bound | Yes, tier list |
| `use(f32x8)` / `use(f32xN)` | Yes | Yes | Yes, recognized concrete token | Yes |
| Short constructors without a token argument | Yes | Yes | Yes inside the wrapper's feature-enabled body | Yes |
| Omit a token parameter in user function | Yes; generator injects internal proof parameter | Yes with explicit tier | No, ordinary form needs proof parameter | With `rite`; ordinary mode still needs proof parameter |
| Safe call from an ordinary caller | Yes; detects and dispatches | No for SIMD tiers; caller must enable sufficient features | Yes with valid token | Ordinary variants: yes with token; `rite` variants: matching context only |
| Built-in runtime dispatcher | Yes | No | No | No; use `incant!` with ordinary variants |
| Multiple generated tiers | Yes | Yes when multiple tiers requested | No | Yes |
| Naming / visibility | Named dispatcher; suffixed private variants | One tier keeps name; several get suffixes; original visibility | Named wrapper; implementation helper | Suffixed variants; original visibility |
| `define(...)` option | No | No | No | Yes, legacy token-taking aliases |
| Substitute `Token` in signature/body | No; special token parameter handling only | No | No | Yes, except tokenless `default` |
| `use` alias usable in signature | No | No | No | No; explicit generic paths can use `Token` |
| Other type/const parameters | Preserved and forwarded | Preserved | Preserved and forwarded | Preserved |
| Per-tier `v4(cfg(avx512))` syntax | Yes | No; only whole-function `cfg(...)` option | No tier list; whole-function `cfg(...)` option | Yes |
| Automatic Cargo gate on default V4 | No | No automatic per-tier gate | No; configure caller/build explicitly | Yes |

In the `rite` column, a token argument does **not** waive the caller-feature
requirement: the function itself has `target_feature`. `arcane` is the wrapper
that uses a valid proof to enter such a function from ordinary code.

The same operation surface is available once a `use` alias resolves: construction,
loads/stores, arithmetic, conversions, `LANES`, partitions, and `_with_token`
alternatives. Aliasing does not add missing backend operations. Ten adaptive
families are supported for V3/V4/V4x/NEON/WASM SIMD128/scalar. Partitions choose the
selected width, but none of the four attributes automatically processes tails or
strided rows, or makes floating-point reductions width-independent.

A genuine generic backend `T` is separate from ordinary algorithm generics.
`arcane` and `rite` can derive feature requirements from suitable token bounds,
but `use(...)` cannot choose a backend from a feature bound alone. Explicitly
typed generic helpers can retain `T` and call `_with_token`; a concrete tier can
select a fixed backend independently of unrelated type/const parameters.

Inherent methods are supported. For a single method in an ordinary trait impl,
`arcane`'s nested form is the existing route; direct safe trait methods cannot
carry `rite`'s target-feature contract, and variant-producing attributes add
methods the trait did not declare. An inherent helper plus delegation works.

### Confirmed default/gating mismatch

On x86_64 without the `avx512` Cargo feature:

```text
#[autoversion(use(f32xN))]             // rejected: no F32x16Backend for X64V4Token
#[autoversion(v4(cfg(avx512)), v3, neon, wasm128, scalar, use(f32xN))] // accepted
#[magetypes(use(f32xN))]               // default V4 is Cargo-gated; still needs Token parameter
#[rite(v4(cfg(avx512)), v3, use(f32xN))] // rejected: per-tier gates not parsed
```

`autoversion`'s ungated V4 default suited scalar auto-vectorization: enabling
AVX-512 compiler features does not itself require the magetypes AVX-512 backend.
Contextual vectors expose that mismatch. Enabling `avx512` makes the first form
compile; an explicit gated list keeps the fallback-only build working too.
These are current limitations, not changes implemented by this review.

The [compile-only inventory](../tests/design-probes/macro-matrix/README.md)
records nine accepted/rejected cases on Rust 1.98.1. Source references:
[alias resolver](../archmage-macros/src/vector_aliases.rs),
[autoversion generation](../archmage-macros/src/autoversion.rs),
[rite generation](../archmage-macros/src/rite.rs),
[arcane proof/wrapping](../archmage-macros/src/arcane.rs),
[magetypes substitution](../archmage-macros/src/magetypes.rs), and
[tier defaults](../archmage-macros/src/tiers.rs).

### What can be unified next (proposals, not implemented)

1. **One tier resolver and variant generator.** Share selected tier, feature set,
   backend type, Cargo gate, suffix, alias emission, and boundary mode. Keep the
   four public spellings compatible while removing independent plumbing.
   `magetypes(rite, ...)` already substantially overlaps multi-tier `rite`.
   `autoversion` can reuse the same variants and add its dispatcher.

2. **One tier/gate grammar and coherent defaults.** Add per-tier gates to `rite`
   and decide how vector-using `autoversion` gates V4. This is the first user-visible
   inconsistency to fix. A named/shared tier set could then be used by dispatcher,
   helper variants, and `incant!`. Preserve existing scalar-code dispatch behavior
   deliberately rather than silently changing every `autoversion` default.

3. **Tokenless source signatures for generated kernels.** Ordinary `magetypes`
   can inject an internal boundary proof parameter, as `autoversion` already does.
   Keep existing explicit-token signatures and `define(...)` working. Align helper
   call conventions so users do not need `without token` solely to compensate for
   which generator created the callee. Macros need declared calling conventions;
   an attribute on one function cannot inspect another function's expanded ABI.

4. **Signature aliases in type positions.** Resolve `f32xN` in parameters/results
   using the same per-tier mapping, preserving qualified paths and real generic
   bindings. This would allow tokenless vector-in/vector-out helpers. It cannot
   give a runtime dispatcher different public Rust return types per CPU: its
   input/output representation must stay uniform. That remains a real constraint.

5. **A scope that owns tier propagation.** A future module/impl-level generator
   could declare the tier set and aliases once, generate each scope per tier, and
   annotate each helper with that tier. Scope-level aliases could also be visible
   in signatures. Ordinary nested functions do not inherit target features;
   macros do not discover a parent's tier from a separately expanded attribute.
   Explicit scope ownership is how “specify a tier somewhere” can work reliably.

6. **Separate feature tier from vector backend policy.** A stronger feature tier
   can use a covered existing backend rather than requiring a duplicate backend
   impl for every crypto/ARM extension token. Generate an explicit verified mapping
   and preserve the distinction between the original proof type and the selected
   vector type. This is also a possible policy for feature-bound generics; it must
   not silently replace an arbitrary `T` or assume that a namespace proves features.

Retain three safety roles even if implementation is shared: runtime detection
(`autoversion`), token-proven entry (`arcane`), and a caller-required feature region
(`rite`). Tier-only, freely callable `arcane(v3)` would need detection and defined
failure/fallback behavior; an annotation alone cannot supply its runtime proof.
`magetypes` can remain a compatible variant-generation facade over that common
engine. No new public vector trait hierarchy is needed for these proposals.

The existing compile measurements cover the current alias implementation only.
Signature rewriting and scope generation need their own measurements. Parse type
positions where possible, reuse registry metadata, and generate only requested
variants; do not assume an expanded scope is free at compile time.

## Verification

`magetypes/tests/adaptive_use.rs` checks all ten families, shared f32/i32 lane
counts, all seven attribute entry points, fixed-width and legacy coexistence,
scalar/default fallbacks, tokenless helpers, and all slice widths 0..65 with
offsets 0..3 across three padded rows. `ARCHMAGE_ADAPTIVE_TEST_TIER` selects a
mandatory backend for execution (`scalar` by default); an unavailable requested
token fails. CI selects V3/V4/V4x in its SDE lanes. Compiler rejection tests in
`tests/soundness/adaptive_*.rs` cover missing/weaker/nested feature contexts and
ambiguous generic backend selection; matching/superset contexts are accepted.

## Compile-time cost

The [actual implementation comparison](../benchmarks/adaptive_use_compile_2026-09-27.md)
measures manual natural-width aliases before/after and implemented
`rite(use(f32xN))`: six-run cold release medians were 2.834 → 2.839 s (default)
and 3.040 → 3.053 s (AVX-512), with overlapping ranges. This adds no public trait
family or vector implementation. The measurements are specific to that consumer.


See the [measured expansion-shape comparison](../benchmarks/tier_width_compile_2026-09-27.md).
The implementation adds name parsing and a per-tier lookup to existing
variant generation. It does not need a new family of vector implementations,
new trait monomorphizations, or another layer of forwarding. Consumer cost still
depends on the kernel and the number of generated variants and used widths.
Adding fallback widths or generating variants users did not request costs work;
select exactly one width per requested family per variant.

That earlier measurement compares fixed-x8 and tier-selected existing aliases. It does
not quantify parser changes, signature rewriting, scalar-x1 parity, or adding
support to every attribute. The actual implementation comparison above covers
the current parser change; signature rewriting and scalar-x1 parity remain
unimplemented. Do not extrapolate a universal overhead from this small kernel.
