+++
title = "IntoConcreteToken Trait"
weight = 4
+++

`IntoConcreteToken` checks the **identity of a token's type**. It does not detect
new CPU features or convert a higher token into a lower token.

```rust
use archmage::{IntoConcreteToken, SimdToken, X64V4Token};
#[cfg(target_arch = "x86_64")]
if let Some(v4) = X64V4Token::summon() {
    assert!(v4.as_x64v4().is_some());
    assert!(v4.as_x64v3().is_none());
    let v3 = v4.v3();
    assert!(v3.as_x64v3().is_some());
}
```

Use `.v3()`, `.v2()`, or `.neon()` when the held token implies that lower tier.
These extraction methods are infallible and do not repeat CPU detection.
Use `summon()` when you need evidence for additional features.

For ordinary portable kernels, use a generated entry and a backend-generic
helper. For a family of concrete variants, use [incant!](@/archmage/dispatch/incant.md).
Manual branching on exact token types must handle unrecognized token types and cfg-gate
architecture-specific calls. Falling through without doing the work is not a
valid scalar fallback.

This is a low-level reference facility. The reviewed zen kernel patterns do not
justify introducing exact-token-type dispatch into the beginner call chain.

With `incant!(... with token)`, a `scalar` tier only matches an actual
`ScalarToken`; an unrecognized held token can panic. Use a tokenless `default`
fallback if every unrecognized token must still run the operation.
