+++
title = "The #[rite] Macro"
weight = 4
+++

`#[rite]` applies target features directly to an internal helper. `#[arcane]`
provides the safe entry from ordinary code. Put the batch loop behind the entry,
then call matched helpers inside it.

This small x86 specialization follows the multiply/load/store/tail structure of
`zenfilters/src/simd/x86.rs`. It is an adaptation for illustrating the boundary,
not a replacement for the portable [gain tutorial](@/archmage/getting-started/first-simd.md).

```rust
use archmage::prelude::*;

#[rite(import_intrinsics)]
fn scale_chunk(_token: X64V3Token, chunk: &mut [f32; 8], factor: f32) {
    let v = _mm256_loadu_ps(chunk);
    let scaled = _mm256_mul_ps(v, _mm256_set1_ps(factor));
    _mm256_storeu_ps(chunk, scaled);
}

#[arcane(import_intrinsics)]
fn scale_v3(token: X64V3Token, data: &mut [f32], factor: f32) {
    let (chunks, tail) = data.as_chunks_mut::<8>();
    for chunk in chunks { scale_chunk(token, chunk, factor); }
    for value in tail { *value *= factor; }
}

fn scale_scalar(_token: ScalarToken, data: &mut [f32], factor: f32) {
    for value in data { *value *= factor; }
}

pub fn scale(data: &mut [f32], factor: f32) {
    incant!(scale(data, factor), [v3, scalar])
}
let mut data = [2.0; 11];
scale(&mut data, 3.0);
assert_eq!(data, [6.0; 11]);
```

The safe unaligned wrappers accept array references. Both x86 functions receive
V3 features; the helper can inline into the loop. `incant!` references the x86
entry only on x86. Other targets use the scalar implementation.

## Forms

| Form | Use |
|---|---|
| `#[rite]` with a concrete token parameter | Features inferred from that token |
| `#[rite(v3)]` | Explicit tier, no token parameter required; function keeps its name |
| `#[rite(v3, neon)]` | Multiple suffixed, cfg-gated variants |
| `#[magetypes(rite, ...)]` | Per-tier `Token` substitution plus direct feature attributes |
| `import_intrinsics` | Combined intrinsic namespace with available reference-based wrappers |

A tokenless `#[rite(v3)]` body can obtain `X64V3Token::from_context()` when it
needs a token. Rust checks that the caller context covers the constructor's
features. This is a reference option; the primary zen examples thread tokens.

Rite has no baseline-safe outer wrapper. Summoning a token in an ordinary caller
does not make a direct rite call safe to rustc. Use the default `#[magetypes]`
or `#[arcane]` entry. Attributes permit optimization; they do not establish a
universal zero-call or zero-overhead result. Inspect your optimized loop.

## Tokenless helpers in codec call chains

`rav1d-safe` uses `#[rite(neon)]` for `cfl_row_8bpc` calling `cfl_lane4` in
[`safe_simd/ipred_arm.rs`](https://github.com/imazen/rav1d-safe/blob/e73811f5d4dad81b75195ca18554fd8a5df19515/src/safe_simd/ipred_arm.rs).
Both helpers have explicit features and no token parameter. The surrounding
codec entry supplies the feature context. Ordinary calls between matching
helpers work; `without token` is not required for this production pattern.

In the runnable x86 example above, the corresponding refactor is to annotate
`scale_chunk` with `#[rite(v3, import_intrinsics)]`, remove its token argument,
and call `scale_chunk(chunk, factor)` from `scale_v3`. The feature-enabled
entry still receives the proof token. This is useful when inner helpers operate
only on intrinsic values and references and have no need to construct magetypes.

## Call a token-first helper from a tokenless context

Repository addition: a tokenless `#[rite]` body can use ordinary `incant!` to
call a token-first helper. The rewriter selects a covered tier and supplies
`CalleeToken::from_context()`. It does not summon or attempt a stronger tier.

```rust
use archmage::prelude::*;
#[magetypes(rite, v3, neon, wasm128, scalar)]
fn sum<const N: usize>(_token: Token, values: &[u32; N]) -> u32 {
    values.iter().sum()
}
#[rite(v3, neon, wasm128, scalar)]
fn helper(values: &[u32; 3]) -> u32 {
    incant!(sum::<3>(values), [v3, neon, wasm128, scalar])
}
#[magetypes(v3, neon, wasm128, scalar)]
fn entry(_token: Token, values: &[u32; 3]) -> u32 {
    incant!(helper(values) without token)
}
pub fn total(values: &[u32; 3]) -> u32 {
    incant!(entry(values), [v3, neon, wasm128, scalar])
}
assert_eq!(total(&[1, 2, 3]), 6);
```

The V3 helper contains the equivalent of
`sum_v3::<3>(X64V3Token::from_context(), values)`. The registry's implication
relationships restrict selection to the caller's architecture and covered
features. Rust independently checks the emitted constructor and helper calls.
A cfg-disabled preferred tier falls through to another covered tier or the
explicit scalar/default fallback. No covered tier and no fallback is a compile
error. A fallback is selected statically, not by a CPU probe.

An ordinary baseline caller cannot construct feature proof:

```compile_fail,E0133
use archmage::X64V3Token;
fn main() { let _token = X64V3Token::from_context(); }
```

The example above is a reference composition test, not a claim that zen already
uses this new spelling. It shortens the explicit `from_context()` bridge. Plain
calls between existing tokenless helpers remain appropriate too.
