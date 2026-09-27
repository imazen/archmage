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
  standalone x1 wrappers, unlike the scalar namespace's x4 aliases. Its
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

See the [measured expansion-shape comparison](../benchmarks/tier_width_compile_2026-09-27.md).
The implementation adds name parsing and a per-tier lookup to existing
variant generation. It does not need a new family of vector implementations,
new trait monomorphizations, or another layer of forwarding. Consumer cost still
depends on the kernel and the number of generated variants and used widths.
Adding fallback widths or generating variants users did not request costs work;
select exactly one width per requested family per variant.

That earlier measurement compares fixed-x8 and tier-selected existing aliases. It does
not quantify parser changes, signature rewriting, scalar-x1 parity, or adding
support to every attribute. Measure those changes when implemented; do not
extrapolate a universal compile-time overhead from this small kernel.
