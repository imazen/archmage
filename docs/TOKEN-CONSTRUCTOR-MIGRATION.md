# Token constructor migration

The next magetypes 0.9 patch adds `_t` methods alongside the existing token-first
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

Native raw values have a uniform `from_raw_t(token, raw)` entry point. Platform
names that already end in `_t`, such as `from_float32x4_t(token, raw)`,
remain available without deprecation or a redundant `_t_t` alias. Prefer `from_raw_t` for
new raw interchange code.
The separate `from_raw(raw)` method requires a matching target-feature context
and is not deprecated.

Ordinary functions and backend-generic helpers can call `_t` methods without
target-feature annotations:

```rust
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8;

fn broadcast<T: F32x8Backend>(token: T, value: f32) -> f32x8<T> {
    f32x8::splat_t(token, value)
}
```

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
macro syntax migration for the constructor change. Compatibility with an actual
0.10 package must be compiled before that release.

The [complete signature inventory](constructors/README.md) lists every old and
new constructor, bound, and platform gate for all 40 vector types.

## Maintenance

`xtask/src/simd_types/generic_gen/token_aliases.rs` derives deprecated legacy forwarders from the canonical `_t`
implementation signatures, preserving argument order, bounds, lifetimes,
attributes, and feature gates. It handles generated vectors and the handwritten
cross-width and scalar modules. Regenerate with `cargo run -p xtask -- generate`;
do not maintain separate constructor lists or hand-edit the generated aliases.

## Compatibility checks

Against the published 0.9.29 API snapshots, the x86-64, AArch64, and WASM
surfaces retain every existing public line. The migration tests exercise both
spellings, baseline-callable function pointers, generic token bounds, floating
point bit preservation, mutable slice views and tails, and existing `define`
macro syntax. Default, no-default-feature, and AVX-512 configurations pass the
focused tests; AArch64 tests also pass under QEMU. WASM and i686 test targets
compile. These checks cover the additive 0.9 change, not the future 0.10 API.

## Release verification (2026-09-27)

The baseline is the published 0.9.29 packages, not the preserved constructor-mode
or `use(...)` draft. `cargo-semver-checks 0.50.0` found no breaking API changes in
archmage or magetypes on x86-64, nor in magetypes on AArch64 or WASM, using
a patch-release comparison. That result predates the deprecation step. With legacy-name deprecations enabled,
the x86-64 patch comparison flags only `type_method_marked_deprecated` (222
checks pass); no signatures were removed. This is the intentional warning
change described above, not an unconditional clean semver-check result.
The tool excludes proc-macro crates: selecting
archmage-macros alone reports no checkable library target. Its compatibility
is covered by expansion snapshots and downstream compilation fixtures, not
by semver-checks. The API check addresses source compatibility;
it does not establish numerical equivalence or future 0.10 compatibility.

For WASM, the tool's automatic rustdoc generation hit
[cargo-semver-checks #1068](https://github.com/obi1kenobi/cargo-semver-checks/issues/1068).
Generating both rustdoc JSON inputs with `cargo rustdoc` (without
`--cap-lints=allow`) and supplying `--current-rustdoc` / `--baseline-rustdoc`
completed the comparison successfully. The ordinary per-target API snapshots
remain the CI check.

`just check-packages` builds actual crate archives together without publishing.
It asserts the normalized manifests retain the exact archmage→archmage-macros
pin and the ordinary compatible magetypes→archmage version requirement. The
archives were also checked with `--target x86_64-unknown-linux-gnu`,
`--target aarch64-unknown-linux-gnu`, and `--target wasm32-wasip1` passed to
`python3 xtask/check_packages.py` (repeat `--target` to check several).

Published jxl-encoder-simd 0.3.0 still calls `f32x4::from_i32x4(vector)` on ARM
and WASM, so it does not compile there
([#117](https://github.com/imazen/archmage/issues/117)). The fix belongs in that
crate: pass the token, `f32x4::from_i32x4(token, vector)`. The calling-convention matrix runs on x86,
ARM/QEMU, and WASM/Wasmtime and covers scalar/default signatures, tokenful and
tokenless composition, nested dispatch, and const generics.

No package version was bumped and nothing was published by these checks.

## Downstream compilation after deprecation

The [2026-09-27 consumer audit](DOWNSTREAM-COMPATIBILITY.md) covers 20 published
zen-prefixed consumers, linear-srgb, garb, jxl-encoder-simd, and 14 local
packages. It records exact versions, baseline comparisons, coverage limits,
and cargo-copter workarounds for yanked-version selection and inherited features.
