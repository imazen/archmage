+++
title = "Prelude"
weight = 4
+++

Use `use archmage::prelude::*;` in kernels to import the macros and token names.
Import generic vector types and backend traits explicitly when using those
spellings. `define(...)` supplies body-local aliases without vector imports.

The [first kernel](@/archmage/getting-started/first-simd.md) shows the minimal
imports with `define(f32x8)`. The [generic helper](@/magetypes/examples/pixel-blending.md)
adds `magetypes::simd::{backends::F32x4Backend, generic::f32x4}`.

An import does not enable CPU features. `#[magetypes]`, `#[arcane]`, and
`#[rite]` establish the relevant function contexts.

For direct intrinsics, `import_intrinsics` on `#[arcane]` or `#[rite]` imports
the appropriate combined intrinsic namespace. Reference-based memory wrappers
are available for supported operations; not every raw pointer intrinsic has a
safe wrapper. Consult the [intrinsics browser](https://imazen.github.io/archmage/intrinsics/).
