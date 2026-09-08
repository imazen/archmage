+++
title = "Understanding Tokens"
weight = 3
+++

A token is a small proof value: its construction establishes that the required
CPU features are available. Copying it does not repeat detection. It does not
enable features in an ordinary function by being passed as an argument.

Normally [incant!](@/archmage/dispatch/incant.md) summons the token and calls a
`#[magetypes]` variant for you. Explicit summoning is useful for a chosen tier:

```rust
use archmage::{SimdToken, X64V3Token};
if X64V3Token::summon().is_some() {
    // This machine can provide V3 proof. SIMD work still belongs in
    // a #[magetypes] or #[arcane] context, not this baseline block.
}
```

`None` is normal on another architecture or a CPU missing a required feature.
Use the fallback rather than unwrapping unconditionally. Safe code cannot forge
a token from a model name or from the presence of just one instruction feature.

## Reuse and extract proof

A higher token can provide lower-tier tokens through explicit extraction:

```rust
use archmage::{SimdToken, X64V4Token, X64V3Token};
#[cfg(target_arch = "x86_64")]
if let Some(v4) = X64V4Token::summon() {
    let _v3: X64V3Token = v4.v3();
    let _v2 = v4.v2();
}
```

Rust does not implicitly coerce V4 to V3 in an ordinary function call. Use
`.v3()`. `IntoConcreteToken::as_x64v3()` instead tests exact identity, so it
returns `None` for a V4 token. Hardware implication also does not automatically
supply every magetypes backend implementation for that token type.

## Construct from an existing feature context

Repository addition after 0.9.28: `from_context()` uses a target-feature function
as the constructor. Rust requires a caller with matching or superset features:

```rust
use archmage::prelude::*;
#[rite(v3)]
fn context_helper() -> bool {
    let _token = X64V3Token::from_context();
    true
}
#[arcane]
fn entry(_token: X64V3Token) -> bool { context_helper() }
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() { assert!(entry(token)); }
```

This reference example covers an already feature-enabled caller. Most reviewed
zen kernels instead receive and thread the token explicitly. `from_context()`
is not an unchecked constructor callable from an ordinary baseline function.
The dedicated compile-fail tests verify that restriction.
