# Tier-selected type aliases

`#[magetypes(use(f32x8), ...)]` is implemented. Width-adaptive `use(f32x)` and
`use(...)` on the other attributes below are proposals; the current parser does
not implement them. The [constructor migration](TOKEN-CONTEXT-MIGRATION.md)
describes the implemented fixed-width API.

## Select existing types at macro expansion

Use the variant's declared tier, not the build host or runtime slice length.
Resolve an unsized family name to a concrete existing contextual alias:

| Variant | Proposed `f32x` | Logical width |
|---|---|---|
| V3 | `generic::local::f32x8<X64V3Token>` | 256 bits |
| V4 / V4x | `generic::local::f32x16<that token>` | 512 bits |
| NEON | `generic::local::f32x4<NeonToken>` | 128 bits |
| WASM SIMD128 | `generic::local::f32x4<Wasm128Token>` | 128 bits |
| Scalar, first implementation | `generic::local::f32x4<ScalarToken>` | Four scalar lanes |

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

Some registered tiers, including V1/V2 and extended ARM/crypto tokens, do not
have direct implementations of all the vector backend traits. A shared resolver
must either map a known tier explicitly to a covered canonical backend or reject
it. Do not manufacture an impl or silently select an unsupported token. A
canonical mapping also needs an explicit token downgrade for `_with_token` calls.

Use one resolver and alias emitter across macros, with backend/feature mappings
from the registry. No new public `f32x<T>` trait family or associated-vector-type
hierarchy is needed. A concrete alias reuses the existing vector implementation.

## Partitions, rows, and tails

The existing `partition_slice` returns `(&[[f32; N]], &[f32])`; its mutable form
returns the corresponding disjoint mutable views. Selecting the type selects
`N`, and `f32x::LANES` is a concrete associated constant in each expansion.
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

Related aliases should use a consistent bit width: `f32x` and `i32x` have equal
lane counts; `f64x` has half as many, and `u8x` four times as many. Fixed-width
names remain necessary for fixed-format blocks, shuffles with literal indices,
width-specific method names, and conversions whose source/destination lane
counts differ. Do not rename those operations implicitly.

## Feasibility across all seven attribute entry points

| Attribute | Fixed-width `use(...)` today | Proposed implementation / limit |
|---|---|---|
| `magetypes` | Yes | Extend per-tier alias emission with adaptive names |
| `rite` | No | Resolve from an explicit tier, generated tier, or concrete token parameter; token value is optional with a tier |
| `token_target_features` | No | Alias of `rite`; share its parser and emitter |
| `arcane` | No | Resolve from its concrete token parameter and insert aliases in the feature-enabled inner function |
| `token_target_features_boundary` | No | Alias of `arcane`; same implementation |
| `simd_fn` | No | Legacy alias of `arcane`; same implementation |
| `autoversion` | No | Per-variant aliases, including scalar/default and `cfg(feature)` fallback; public dispatcher signature stays tier-independent |

The `arcane` sibling and nested forms must both use the same resolution, including
methods with `self`/`Self`. Tokenless `arcane` would be a separate dispatch/proof
API change: importing a vector alias does not supply the entry-point capability
proof. The current `magetypes(rite, ...)` wrapper emits `rite(import_intrinsics)`
without the tier argument, so it still depends on a token parameter. Supporting
tokenless generated bodies would require forwarding the tier it already knows.

For genuine `T: Backend` helpers, preserve `T` and use `_with_token`. A feature
bound gives a minimum capability, not a unique backend identity. Initially,
require an explicit tier or recognized concrete token for alias selection;
do not silently replace generic parameters or rewrite their input/output types.

Body aliases are inexpensive and avoid rewriting variable names. To use the
short names in parameters or results, rewrite only type positions in the
signature, preserving qualified paths and generic bindings. Tier-specific
internal helpers may have tier-specific signatures and be called by matching
variants. A runtime dispatcher cannot return different Rust vector types or
array lengths from different arms: its public signature must use scalar values,
slices, fixed arrays independent of the tier, or another uniform representation.

Rust's built-in `#[target_feature]`, `#[inline]`, and `#[cfg]` parsers cannot be
extended by this crate. The feasibility table concerns archmage's attributes.

## Compile-time cost

See the [measured expansion-shape comparison](../benchmarks/tier_width_compile_2026-09-27.md).
The proposed implementation adds name parsing and a per-tier lookup to existing
variant generation. It does not need a new family of vector implementations,
new trait monomorphizations, or another layer of forwarding. Consumer cost still
depends on the kernel and the number of generated variants and used widths.
Adding fallback widths or generating variants users did not request costs work;
select exactly one width per requested family per variant.

The measurement compares fixed-x8 and tier-selected existing aliases. It does
not quantify parser changes, signature rewriting, scalar-x1 parity, or adding
support to every attribute. Measure those changes when implemented; do not
extrapolate a universal compile-time overhead from this small kernel.
