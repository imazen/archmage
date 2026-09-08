+++
title = "Cross-Platform Behavior"
weight = 5
+++

Start with one portable kernel and one explicit tier list, as in
[Your First SIMD Function](@/archmage/getting-started/first-simd.md).
`#[magetypes]` cfg-gates generated definitions and `incant!` cfg-gates their
call sites. You do not need hand-written platform wrappers for that pattern.

Token names exist on all targets. An unsupported token's `summon()` returns
`None`, but that runtime result does not remove a reference to a nonexistent
function from Rust's name resolution. `#[arcane]` and `#[rite]` normally omit
wrong-architecture definitions. A direct manual caller therefore needs an
architecture guard; [manual dispatch](@/archmage/dispatch/manual.md) shows it.

Choose logical shapes supported by every selected backend. [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) has eight
lanes on every backend; NEON and WASM implement it with multiple native vectors.
Do not assume a stronger hardware token implements every narrower backend
trait. Use explicit token extraction or the documented backend matrix.

Portable API names do not imply identical exceptional floating-point results.
[ISA quirks and fixups](@/magetypes/isa-quirks.md) records those contracts.
