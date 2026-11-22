+++
title = "IntoConcreteToken Trait"
weight = 4
+++

`IntoConcreteToken` lets generic code recover a concrete CPU proof. The `as_*`
methods check the token's exact type. The `try_*` methods also accept descendants
in the registry's parent hierarchy. Both use the caller's existing proof.

```rust
use archmage::{IntoConcreteToken, ScalarToken, SimdToken, X64V4Token};

fn has_v3(token: impl IntoConcreteToken) -> bool {
    token.try_v3().is_some()
}

assert!(!has_v3(ScalarToken));
if let Some(v4) = X64V4Token::summon() {
    assert!(v4.as_x64v4().is_some());
    assert!(v4.as_x64v3().is_none());
    assert!(has_v3(v4));
}
```

Use `.v3()`, `.v2()`, or `.neon()` when the held token implies that lower tier.
These extraction methods are infallible and do not repeat CPU detection.
Use `summon()` when you need evidence for additional features.

For ordinary portable kernels, use a generated entry and a backend-generic
helper. For a family of concrete variants, use [attuned!](@/archmage/dispatch/incant.md).
When branching on `try_*`, check stronger tiers first, cfg-gate
architecture-specific calls, and provide a scalar fallback.

With `attuned!(work(data), [_v3, _scalar], using(token))`, non-scalar tiers
check `IntoConcreteToken` without probing the CPU, and any unrecognized token
type falls through to the guaranteed `_scalar` arm (`work_scalar_t(ScalarToken,
data)`).
