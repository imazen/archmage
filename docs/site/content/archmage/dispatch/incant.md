+++
title = "attuned! and reattune! Macros"
description = "Context-aware SIMD family calls and runtime tier reselection (replaces incant!)"
weight = 2
aliases = ["archmage/dispatch/tiered-fallback/"]
+++

`attuned!` and `reattune!` replace `incant!` (and its alias `dispatch_variant!`),
which is superseded and queued for deprecation in the next release.

`attuned!` connects a caller to a family of tier-suffixed implementations,
probing the CPU from ordinary code and staying zero-probe inside an already
attuned context. The [first gain kernel](@/archmage/getting-started/first-simd.md)
shows the portable call chain;
[type and const generics](@/magetypes/dispatch/types-and-dispatch.md) shows
turbofish forwarding (`attuned!(kernel::<3, Pixel>(args))`). Keep dispatch
outside the hot loop.

## Variants and fallback

Outside an `#[attune]` feature body, `attuned!` calls `_t` proof wrappers and
prepends each tier's summoned token (or substitutes a `Token` argument marker
when the token is not first). Elided `attuned!(gain_impl(plane, gain))` uses the
default portable tiers (`[_v3, _neon, _wasm128, _scalar]`):

| Variant | First argument |
|---|---|
| `gain_impl_v3_t` | `X64V3Token` |
| `gain_impl_neon_t` | `NeonToken` |
| `gain_impl_wasm128_t` | `Wasm128Token` |
| `gain_impl_scalar_t` | `ScalarToken` |

`attuned!` cfg-gates architecture-specific arms automatically and sorts
candidates by registry priority rather than textual list order.

Default `attuned!(...)` never includes `_v4` or `_v4x`, even when `avx512` is
enabled. Pass an explicit list when adding AVX-512 or calling a sparse family:
`attuned!(gain_impl(plane, gain), [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar])`.
Explicit lists are exhaustive: no `_scalar` fallback is appended implicitly, and
every call must retain a guaranteed fallback after architecture and Cargo
feature gates are applied. Forward `avx512` to dependencies as shown in
[installation](@/archmage/getting-started/installation.md).

## Calls inside a tier (`attuned!` vs `reattune!`)

Inside an `#[attune]` body, `attuned!` checks whether each candidate is covered
by the enclosing target-feature context:

- Covered candidates emit a direct call (`helper_v3(args)`) with no CPU probe
  and no token argument (unless forced to proof form via `_v3_t` or given a
  `Token` argument marker).
- Uncovered candidates can use one eligible enclosing proof parameter (for
  example, calling a lower or generic `IntoConcreteToken` target).
- `reattune!(work(args), [_v4(cfg(avx512)), _v3, _scalar])` skips enclosing-proof
  retention and explicitly probes at runtime for a stronger tier such as V4 from
  inside a V3 context.

## Hand-tuned variants

A generated family can omit one tier with `-_<tier>`, which you supply by hand
with `#[attune]` and the matching `_<tier>_t` suffix. Here `#[attune]` generates
the NEON, WASM, and scalar proof wrappers of the gain kernel, and the V3 entry
is written with x86 intrinsics:

```rust
use archmage::prelude::*;

#[attune(_*_t, -_v3, define(f32x8))]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail { *value *= gain; }
}

// The _v3_t suffix and token-first signature supply the family's V3 proof entry.
#[attune(import_intrinsics)]
fn gain_impl_v3_t(_token: X64V3Token, plane: &mut [f32], gain: f32) {
    let factor = _mm256_set1_ps(gain);
    let (chunks, tail) = plane.as_chunks_mut::<8>();
    for chunk in chunks {
        _mm256_storeu_ps(chunk, _mm256_mul_ps(_mm256_loadu_ps(chunk), factor));
    }
    for value in tail { *value *= gain; }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    attuned!(gain_impl(plane, gain))
}

let mut plane = [2.0; 11];
apply_gain(&mut plane, 0.5);
assert_eq!(plane, [1.0; 11]);
```

This adapts the gain kernel from
[Reusable generic kernels](@/magetypes/examples/generic-kernels.md), where
`zenfilters` splits generated portable variants from specialized x86 entries.
Keep hand-written variants aligned in input/output behavior and numerical
contract with the rest of the family; see
[direct intrinsic helpers](@/archmage/concepts/rite.md) and
[dispatch testing](@/archmage/testing/dispatch-testing.md).

## Keep a separate scalar algorithm

Use `-_scalar` when the scalar fallback calls a separate reference
implementation rather than running the vector body under `ScalarToken`. This
reduced coefficient-level kernel follows `aom-rs`
([`aom-txb/src/simd.rs`](https://github.com/imazen/aom-rs/blob/028c5e131c0eaa338df931fb252a8b234ba700fb/crates/aom-txb/src/simd.rs)),
retaining the full-`i32`-domain absolute-value and clamp proof:

```rust
use archmage::prelude::*;

#[attune(_*_t, -_scalar, define(i32x8))]
fn levels_impl(token: Token, input: &[i32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let zero = i32x8::zero_t(token);
    let cap = i32x8::splat_t(token, 127);
    let (chunks, tail) = input.as_chunks::<8>();
    let (dst, dst_tail) = output.as_chunks_mut::<8>();
    for (chunk, out) in chunks.iter().zip(dst) {
        let x = i32x8::load_t(token, chunk);
        let sign = x.shr_arithmetic_const::<31>();
        let abs = (x ^ sign) - sign;
        // Only MIN remains negative after wrapping absolute value.
        let clamped = i32x8::blend(abs.simd_lt(zero), cap, abs.min(cap));
        *out = clamped.to_array().map(|v| v as u8);
    }
    levels_impl_scalar_t(ScalarToken, tail, dst_tail);
}

fn levels_impl_scalar_t(_token: ScalarToken, input: &[i32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    for (&v, out) in input.iter().zip(output) {
        *out = v.unsigned_abs().min(127) as u8;
    }
}

pub fn levels(input: &[i32], output: &mut [u8]) {
    attuned!(levels_impl(input, output))
}

let input = [i32::MIN, i32::MAX, -128, -127, -1, 0, 1, 127, 128];
let mut output = [0; 9];
levels(&input, &mut output);
assert_eq!(output, input.map(|v| v.unsigned_abs().min(127) as u8));
```

SIMD subtraction wraps as specified by the vector API, leaving only `i32::MIN`
negative after `(x ^ sign) - sign`; the `simd_lt(zero)` blend clamps it to 127
before casting to `u8`. Always include `i32::MIN` in test inputs.

## Held tokens (`using(...)`), renaming (`names(...)`), and `reattune!`

`using(token)` dispatches on the exact type of an already-held token via
`IntoConcreteToken`, evaluating the expression once and never probing the CPU.
Any unrecognized token falls through to the guaranteed `_scalar` arm. `names(...)`
maps tiers to nonstandard function paths when calling external families.

```rust
use archmage::prelude::*;
use archmage::IntoConcreteToken;

#[attune]
fn twice_v3_t(_token: X64V3Token, x: u32) -> u32 { x * 2 }
fn twice_scalar_t(_token: ScalarToken, x: u32) -> u32 { x * 2 }

fn twice<T: IntoConcreteToken>(token: T, x: u32) -> u32 {
    attuned!(twice(x), [_v3, _scalar], using(token))
}

#[attune(all, +v4(cfg(avx512)))]
fn bump(x: u32) -> u32 { x + 1 }

#[attune]
fn widen_v3_t(_token: X64V3Token, x: u32) -> u32 {
    reattune!(bump(x), [_v4(cfg(avx512)), _v3, _scalar])
}

assert_eq!(twice(ScalarToken, 4), 8);
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    assert_eq!(twice(token, 4), 8);
    assert_eq!(widen_v3_t(token, 4), 5);
}
```

## Legacy `incant!` spelling (queued for deprecation)

| Legacy spelling | `attuned!` / `reattune!` replacement | Notes |
|---|---|---|
| `incant!(gain(plane, k), [v3, neon, wasm128, scalar])` | `attuned!(gain(plane, k))` or `attuned!(gain(plane, k), [_v3, _neon, _wasm128, _scalar])` | Tier names use leading `_`; outside a feature body, `attuned!` calls `_t` proof wrappers |
| `incant!(gain(plane, k), [v4(cfg(avx512)), v3, neon, wasm128, scalar])` | `attuned!(gain(plane, k), [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar])` | AVX-512 is always opt-in on `attuned!` |
| `incant!(helper(x) without token)` | `attuned!(helper(x))` | Inside an `#[attune]` body, covered `_*` helpers are called directly without a token |
| `incant!(work(x) with token, [v3, default])` | `attuned!(work(x), [_v3, _scalar], using(token))` | Checks exact token type without probing; unmatched tokens run `_scalar` instead of panicking |
| Manual runtime upgrade inside a tier body | `reattune!(work(x), [_v4(cfg(avx512)), _v3, _scalar])` | Explicitly probes for stronger tiers from an existing feature context |
