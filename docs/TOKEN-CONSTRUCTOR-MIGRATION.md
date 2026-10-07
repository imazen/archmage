# Token constructor migration

magetypes 0.9.30 adds `_t` methods alongside the existing token-first
methods. Both spellings take the token first and have the same behavior:

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x8;

let old = f32x8::splat(ScalarToken, 2.0);
let prepared = f32x8::splat_t(ScalarToken, 2.0);
assert_eq!(old.to_array(), prepared.to_array());
```

Existing token-taking names remain callable but are deprecated, with diagnostics
pointing to the corresponding `_t` method. Projects using `deny(deprecated)` or
`deny(warnings)` must migrate those calls or explicitly allow the warnings.
Vector types stay
generic over one token parameter. No constructor modes or new function-attribute
syntax are part of this change.

## Migrate names before changing contexts

Rename token-taking calls while preserving their arguments and vector widths:

| Existing spelling | Migration spelling |
|---|---|
| `splat(token, value)` | `splat_t(token, value)` |
| `zero(token)` | `zero_t(token)` |
| `load(token, data)` | `load_t(token, data)` |
| `from_array(token, values)` | `from_array_t(token, values)` |
| `partition_slice_mut(token, data)` | `partition_slice_mut_t(token, data)` |
| `from_halves(token, lo, hi)` | `from_halves_t(token, lo, hi)` |

The aliases cover all public inherent methods with an explicit first token,
including conversions, byte loads, block loads, borrowed slice views, native raw
constructors, and the single-lane scalar types. Trait methods whose receiver is
the token do not gain aliases. Existing target and Cargo feature gates still
apply.

Native raw values have two spellings on every native-backend type:
`from_raw_t(token, raw)` for callers that hold a token, generic code included,
and `from_raw(raw)`, which takes no token and requires a matching
target-feature context that Rust checks. The x86 platform-named constructors
0.9.29 shipped (`from_m128`, `from_m128d`, `from_m128i`, `from_m256`,
`from_m256d`, `from_m256i`) are deprecated forwarders to `from_raw_t`, removed
in 0.10; they have no `_t` form, and no other type gets a platform name.

Ordinary functions and backend-generic helpers can call `_t` methods without
target-feature annotations:

```rust
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8;

fn broadcast<T: F32x8Backend>(token: T, value: f32) -> f32x8<T> {
    f32x8::splat_t(token, value)
}
```

Code written for the concrete vector types of 0.9.26 and earlier may call
one-argument conversions such as `f32x4::from_i32x4(v)`. The generic types take
the token first, like the constructors: `f32x4::from_i32x4_t(token, v)`.

Keep function token parameters, dispatch calls, and public vector signatures
unchanged during this rename. Existing `#[magetypes(define(...), ...)]` aliases
refer to the same types and support both method spellings.

## Planned 0.10 boundary

The proposed magetypes 0.10 design retains `_t(token, ...)` and gives the short
constructor names to compiler-checked feature-context construction. That change
is not implemented by this migration release. The new tokenless short names
will not be deprecated; the explicit-token `_t` methods will remain supported. Consumers can adopt `_t` on 0.9,
upgrade later, and then simplify calls inside concrete feature contexts as a
separate step. A generic backend bound alone does not enable target features.

The published archmage 0.9 macro contract uses
`magetypes::simd::generic::TYPE<Token>` for `define(...)`, and tier/backends
namespaces for `import_magetypes`. Preserving those paths avoids requiring a
macro syntax migration for the constructor change.

The [complete signature inventory](constructors/README.md) lists every old and
new constructor, bound, and platform gate for all 40 vector types.

## Maintenance

`xtask/src/simd_types/generic_gen/token_aliases.rs` derives deprecated legacy forwarders from the canonical `_t`
implementation signatures, preserving argument order, bounds, lifetimes,
attributes, and feature gates. It handles generated vectors and the handwritten
cross-width and scalar modules. Regenerate with `cargo run -p xtask -- generate`;
do not maintain separate constructor lists or hand-edit the generated aliases.

`just check-packages` builds the crate archives together without publishing and
asserts that the normalized manifests keep the exact archmage→archmage-macros pin
and the ordinary magetypes→archmage version requirement. Pass `--target` to
`python3 xtask/check_packages.py`, repeatedly, to check other targets.

## Compatibility checks

Against the published 0.9.29 API snapshots, the x86-64, AArch64, and WASM
surfaces retain every existing public line. `cargo-semver-checks` (0.50.0, patch
comparison against the 0.9.29 packages) reports only
`type_method_marked_deprecated`, the intended deprecations; no signature was
removed. The tool cannot check proc-macro crates, so archmage-macros is covered
by expansion snapshots and downstream compilation fixtures instead.

The migration tests exercise both spellings, baseline-callable function
pointers, generic token bounds, floating point bit preservation, mutable slice
views and tails, and existing `define` macro syntax. Default, no-default-feature,
and AVX-512 configurations pass them; AArch64 tests also pass under QEMU, and
WASM and i686 test targets compile. The calling-convention matrix runs on x86,
ARM/QEMU, and WASM/Wasmtime and covers scalar/default signatures, tokenful and
tokenless composition, nested dispatch, and const generics. These checks cover
the additive 0.9 change, not the future 0.10 API, and source compatibility, not
numerical equivalence.

## Downstream compilation after deprecation

The [2026-09-27 consumer audit](DOWNSTREAM-COMPATIBILITY.md) covers 20 published
zen-prefixed consumers, linear-srgb, garb, jxl-encoder-simd, and 14 local
packages. It records exact versions, baseline comparisons, coverage limits,
and cargo-copter workarounds for yanked-version selection and inherited features.
