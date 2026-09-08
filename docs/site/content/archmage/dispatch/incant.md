+++
title = "incant! Macro"
weight = 2
aliases = ["archmage/dispatch/tiered-fallback/"]
+++

`incant!` connects a public function to a family of suffixed implementations.
The [first gain kernel](@/archmage/getting-started/first-simd.md) shows the full
portable chain; [type and const generics](@/magetypes/dispatch/types-and-dispatch.md)
shows turbofish forwarding. Dispatch belongs outside the hot loop.

## Variants and fallback

For `incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])`, provide:

| Variant | First argument |
|---|---|
| `gain_impl_v3` | `X64V3Token` |
| `gain_impl_neon` | `NeonToken` |
| `gain_impl_wasm128` | `Wasm128Token` |
| `gain_impl_scalar` | `ScalarToken` |

The remaining arguments match the call. The macro cfg-gates architecture-specific
references and supplies the selected token. A `default` fallback instead names
`gain_impl_default` and takes no token. `#[magetypes]` vector bodies naturally
use `scalar`; ordinary scalar fallback functions can use `default`.

List tiers deliberately, keeping generation and dispatch in sync. Include a
fallback explicitly rather than relying on the current auto-append behavior.
`v4(cfg(avx512))` gates a variant on the caller's feature; forward that feature
to the dependencies. See [features](@/archmage/getting-started/installation.md).

## Calls inside a tier

Inside a macro-managed feature context, nested `incant!` can be rewritten into
a direct matching-tier call, keeping helpers in that context rather than
repeating runtime detection. With token-first functions it forwards the token.
`without token` is for an existing tokenless variant, not a way to erase a
required token argument. Use `#[rite]` helpers only when the caller's features
cover them; a baseline dispatcher cannot safely enter a bare rite function.

The options `with token` (dispatch using a held token's identity), `without token`,
and tier-list modifiers are reference facilities. They are not prerequisites
for the primary zen patterns. Do not introduce them merely to avoid writing the
simple public/generated/helper chain. [Coverage](@/magetypes/examples/coverage.md)
distinguishes observed production patterns from reference-only ones.

## Hand-tuned variants

A generated family can omit one tier, which you implement with `#[arcane]`
using the expected suffix. Keep its input/output and numerical contract aligned
with the other variants. `zenfilters`, `zenresize`, and `zenblend` use portable
helpers alongside specialized x86 implementations. See
[direct intrinsic helpers](@/archmage/concepts/rite.md).

Use [dispatch testing](@/archmage/testing/dispatch-testing.md) to exercise fallback
paths on the current machine, and architecture-specific CI for other ISAs.

## Keep a separate scalar algorithm

`aom-rs` uses `#[magetypes(..., -scalar)]` when its `_scalar` function must call
the transcribed reference implementation. Generation and dispatch then have
intentionally different lists: generate the SIMD tiers, dispatch to those plus
the separately defined scalar tier.

This reduced coefficient-level kernel follows
[`aom-txb/src/simd.rs`](https://github.com/imazen/aom-rs/blob/028c5e131c0eaa338df931fb252a8b234ba700fb/crates/aom-txb/src/simd.rs).
It retains the full-i32-domain absolute-value/clamp proof; codec column padding
is omitted, and the public input is a flat coefficient slice.

```rust
use archmage::prelude::*;
#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
fn levels_impl(token: Token, input: &[i32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let zero = i32x8::zero(token);
    let cap = i32x8::splat(token, 127);
    let (chunks, tail) = input.as_chunks::<8>();
    let (dst, dst_tail) = output.as_chunks_mut::<8>();
    for (chunk, out) in chunks.iter().zip(dst) {
        let x = i32x8::load(token, chunk);
        let sign = x.shr_arithmetic_const::<31>();
        let abs = (x ^ sign) - sign;
        // Only MIN remains negative after wrapping absolute value.
        let clamped = i32x8::blend(abs.simd_lt(zero), cap, abs.min(cap));
        *out = clamped.to_array().map(|v| v as u8);
    }
    levels_impl_scalar(ScalarToken, tail, dst_tail);
}
fn levels_impl_scalar(_token: ScalarToken, input: &[i32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    for (&v, out) in input.iter().zip(output) {
        *out = v.unsigned_abs().min(127) as u8;
    }
}
pub fn levels(input: &[i32], output: &mut [u8]) {
    incant!(levels_impl(input, output), [v3, neon, wasm128, scalar])
}
let input = [i32::MIN, i32::MAX, -128, -127, -1, 0, 1, 127, 128];
let mut output = [0; 9];
levels(&input, &mut output);
assert_eq!(output, input.map(|v| v.unsigned_abs().min(127) as u8));
```

Here SIMD subtraction wraps as specified by the vector API. The negative
`i32::MIN` sentinel is explicitly repaired to 127. The final byte cast cannot
truncate a value outside 0..127. Tests must include that sentinel, not just
normal positive coefficients.

## Existing-token and tokenless reference forms

`with token` uses the **exact type** of a held token. It does not detect an
upgrade or extract a lower proof. Use `default` when unmatched token types must
still execute a fallback; a `scalar` arm only matches an actual `ScalarToken`.

```rust
use archmage::prelude::*;
use archmage::IntoConcreteToken;
#[arcane]
fn twice_v3(_token: X64V3Token, x: u32) -> u32 { x * 2 }
fn twice_default(x: u32) -> u32 { x * 2 }
fn twice<T: IntoConcreteToken>(token: T, x: u32) -> u32 {
    incant!(twice(x) with token, [v3, default])
}
assert_eq!(twice(ScalarToken, 4), 8);
if let Some(token) = X64V3Token::summon() { assert_eq!(twice(token, 4), 8); }
```

`without token` calls a tokenless callee in the surrounding macro-managed tier.
It takes no tier list, does not detect CPU features, and is rejected in baseline
code. [The complete rite composition example](@/archmage/concepts/rite.md)
shows both directions: an entry calls a tokenless helper using `without token`,
and that helper calls a token-first function using `from_context()` implicitly.

The descriptive dispatch macro alias is `dispatch_variant!`; the primary spelling
in these examples is `incant!`. There is no archmage macro literally named
`dispatch!`.
