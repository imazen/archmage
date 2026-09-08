+++
title = "The #[arcane] Macro"
weight = 3
+++

`#[arcane]` creates a safe entry into a function compiled with the features
proved by its token parameter. For portable vector kernels, normally use
`#[magetypes]`, which generates these entries for the selected tiers.

Start with [the complete portable kernel](@/archmage/getting-started/first-simd.md)
or [the direct-intrinsic entry and helper](@/archmage/concepts/rite.md).
Neither needs an explicit unsafe block.

## What the macro establishes

The wrapper accepts a token, then enters a target-feature function. The wrapper
contains the audited feature-boundary machinery; your body can safely use
intrinsics whose requirements are covered. Safe memory wrappers still enforce
Rust reference size and validity. A CPU token does not justify arbitrary pointer
loads, unchecked indices, or aliasing mutable references.

| Signature/option | Meaning |
|---|---|
| Concrete token such as `X64V3Token` | Uses that token's feature set |
| Recognized tier bound such as `T: HasX64V2` | Uses the declared bound's feature set, not a stronger caller's extra features |
| `import_intrinsics` | Imports the matching combined intrinsic namespace |
| `nested` | Places the feature-enabled function inside the wrapper, useful for trait implementations |
| `_self = Type` | Nested receiver handling; use `_self` inside the extracted body |
| `suppress_const_test` | Opts out of the extra compile-time token-trait check; not a routine application option |

The ordinary expansion uses a sibling feature function and an outer wrapper.
Rust generics and where clauses are forwarded. Methods need the receiver rules
in [methods](@/archmage/advanced/methods.md); a sibling cannot be added to a trait
implementation unless it is a declared trait member, so use nested mode there.

Wrong-architecture definitions are omitted; `stub` has been removed. `incant!` handles
call-site cfg guards; [manual callers](@/archmage/dispatch/manual.md) must supply
them explicitly. A runtime `summon().is_some()` does not cfg-out an invalid name.

For internal helpers use matched `#[rite]` functions or inline backend-generic
helpers. Enter around the whole row/strip/batch, rather than once per tiny vector.
See [target-feature boundaries](@/archmage/concepts/target-feature-boundaries.md).
