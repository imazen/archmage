+++
title = "Trait Reference"
weight = 2
+++

Traits describe static capabilities; they do not turn generic calls into dynamic
dispatch. Ordinary `<T: Trait>` code is monomorphized. `dyn Trait` is a separate
Rust mechanism and is not the normal magetypes kernel pattern.

| Trait family | Use |
|---|---|
| `SimdToken` | Common token operations such as detection and naming |
| `HasX64V2`, `HasX64V3`, etc. | Proof of a tier's CPU capabilities; understood by archmage's feature macros |
| `F32x8Backend`, `I32x4Backend`, etc. | Operations and storage for a particular logical vector shape |
| Published conversion traits such as `F32x8Convert` | Additional cross-type conversions required by a generic body |
| `IntoConcreteToken` | Exact token-type identity checks, not feature upgrades |

Add bounds for the operations your helper actually calls. Source-backend traits
now carry widening, narrowing, pairwise, and terminal byte operations; some
cross-type methods have destination bounds. Follow compiler requirements and
the method's API reference instead of copying every trait into every signature.

The complete [zenblend helper](@/magetypes/examples/pixel-blending.md) needs only
`T: F32x4Backend`. The [zenfilters blur walkthrough](@/magetypes/examples/convolution.md)
uses additional conversions. Neither needs hand-written wrappers for every tier.

A backend bound alone does not enable target features. Call the helper from a
`#[magetypes]` context and verify inlining. A tier bound used with `#[arcane]`
enables that declared tier, not every capability of a stronger concrete caller.

Width capability traits are compatibility APIs, not the preferred way to select
an ISA: “256 bits” alone does not mean AVX2 plus FMA. Prefer tier names for
feature selection and backend traits for vector operations.
