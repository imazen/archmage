+++
title = "Direct Feature Helpers (#[attune(<tier>)] and #[attune(_*)])"
description = "Zero-wrapper #[target_feature] helpers with #[attune(v3)], #[attune] fn name_v3, and #[attune(_*)] (replaces #[rite])"
weight = 4
+++

`#[attune(<tier>)]`, `#[attune] fn name_<tier>`, and `#[attune(_*)]` replace
`#[rite]`, which is superseded and queued for deprecation in the next release.

Direct `#[attune]` helpers apply `#[target_feature]` and `#[inline]` directly to
an internal function with no outer wrapper. Proof wrappers (`#[attune(wrap)]` or
`_<tier>_t`) and `#[attune(dispatch)]` provide the safe entry from ordinary
code. Put the batch loop behind the entry, then call matched helpers inside it.

This x86 specialization follows the multiply/load/store/tail structure of
`zenfilters/src/simd/x86.rs`. It illustrates the entry/helper boundary; for
portable kernels across ISAs, start with the
[gain tutorial](@/archmage/getting-started/first-simd.md).

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[attune(v3, import_intrinsics)]
fn scale_chunk(chunk: &mut [f32; 8], factor: f32) {
    let v = _mm256_loadu_ps(chunk);
    let scaled = _mm256_mul_ps(v, _mm256_set1_ps(factor));
    _mm256_storeu_ps(chunk, scaled);
}

#[attune]
fn scale_v3_t(_token: X64V3Token, data: &mut [f32], factor: f32) {
    let (chunks, tail) = data.as_chunks_mut::<8>();
    for chunk in chunks { scale_chunk(chunk, factor); }
    for value in tail { *value *= factor; }
}

fn scale_scalar_t(_token: ScalarToken, data: &mut [f32], factor: f32) {
    for value in data { *value *= factor; }
}

pub fn scale(data: &mut [f32], factor: f32) {
    attuned!(scale(data, factor), [_v3, _scalar])
}

let mut data = [2.0; 11];
scale(&mut data, 3.0);
assert_eq!(data, [6.0; 11]);
```

The safe unaligned wrappers accept array references. Both x86 functions compile
with V3 features, so `scale_chunk` can inline into the loop without a token
parameter or wrapper call. `attuned!` references `scale_v3_t` only on x86-64;
other targets run `scale_scalar_t`.

## Forms

| Form | Meaning |
|---|---|
| `#[attune(v3)] fn helper(...)` | Explicit tier, no token parameter required; keeps function name |
| `#[attune] fn helper_v3(...)` | Infers direct V3 features from the `_v3` suffix (even if a token parameter is present) |
| `#[attune(_*)] fn helper(...)` | Portable direct variants (`helper_v3`, `helper_neon`, `helper_wasm128`, `helper_scalar`) |
| `#[attune(_*, define(f32x8))] fn helper(token: Token, ...)` | Per-tier `Token` substitution and vector alias on direct helpers |
| `import_intrinsics` | Imports `archmage::intrinsics::{arch}::*` including reference-based memory wrappers |

A tokenless `#[attune(v3)]` body can call `X64V3Token::from_context()` (or
`Token::from_context()` in a `#[attune(_*)]` family) when it needs a token for
Magetypes constructors. Rust checks that the caller context covers the
constructor's features.

Direct helpers have no baseline-safe outer wrapper. Summoning a token in an
ordinary caller does not make a direct call safe to rustc; enter through
`#[attune(dispatch)]`, `#[attune(_*_t)]` + `attuned!`, or `#[attune(wrap)]`.
Attributes permit optimization; they do not guarantee zero-call or
zero-overhead codegen. Inspect your optimized loop.

## Tokenless helpers in codec call chains

In codec kernels where inner helpers operate purely on intrinsic values and
array references (such as row and lane helpers in `rav1d-safe`'s
[`safe_simd/ipred_arm.rs`](https://github.com/imazen/rav1d-safe/blob/e73811f5d4dad81b75195ca18554fd8a5df19515/src/safe_simd/ipred_arm.rs)),
annotate each helper with `#[attune(neon)]` (or `#[attune(v3)]`) and call
matching helpers directly as ordinary functions. The surrounding entry supplies
the feature context; no token threading is required unless inner code constructs
Magetypes vectors.

## Multi-tier helpers and `attuned!` context forwarding

Inside an attuned caller, `attuned!(helper(values))` resolves to the covered
direct variant (`helper_v3`, `helper_neon`, `helper_wasm128`, `helper_scalar`)
with no runtime probe. When a direct helper family also takes a `Token`
parameter, pass `Token` at the call site and `attuned!` supplies
`CalleeToken::from_context()` automatically:

```rust
use archmage::prelude::*;

#[attune(_*)]
fn sum<const N: usize>(token: Token, values: &[u32; N]) -> u32 {
    let _ = token;
    values.iter().sum()
}

#[attune(_*)]
fn helper(values: &[u32; 3]) -> u32 {
    attuned!(sum::<3>(Token, values))
}

#[attune(dispatch)]
pub fn total(values: &[u32; 3]) -> u32 {
    attuned!(helper(values))
}

assert_eq!(total(&[1, 2, 3]), 6);
```

In `helper_v3`, `attuned!(sum::<3>(Token, values))` expands to
`sum_v3::<3>(::archmage::X64V3Token::from_context(), values)`. Selection follows
the caller's architecture and covered features statically, never probing the
CPU. Rust independently verifies both the `from_context()` call and the direct
target-feature call.

An ordinary baseline caller cannot construct feature proof:

```compile_fail,E0133
use archmage::X64V3Token;
fn main() { let _token = X64V3Token::from_context(); }
```

## Legacy `#[rite]` spelling (queued for deprecation)

| Legacy spelling | `#[attune]` replacement | Notes |
|---|---|---|
| `#[rite(v3)] fn helper(x: f32)` | `#[attune(v3)] fn helper(x: f32)` | Same name, visibility, and `#[inline]` default; `#[attune]` preserves an explicit `#[inline(...)]` |
| `#[rite] fn helper(token: X64V3Token, x: f32)` | `#[attune] fn helper_v3(token: X64V3Token, x: f32)` or `#[attune(v3)] fn helper(token: X64V3Token, x: f32)` | Token parameter alone does not infer a tier; use a `_v3` suffix or explicit `v3` |
| `#[rite(v3, neon, wasm128, scalar)] fn helper(x: f32)` | `#[attune(_*)] fn helper(x: f32)` | Same `helper_<tier>` names; dispatch from attuned callers via `attuned!(helper(x))` |
| `#[rite(v3, v4, neon)] fn helper(x: f32)` | `#[attune(_v3, _v4(cfg(avx512)), _neon)] fn helper(x: f32)` | Explicit tier outputs take leading `_`; gate V4 on `cfg(avx512)` |
| `#[magetypes(rite, v3, neon, wasm128, scalar)]` | `#[attune(_*, define(...))]` | Specializes `Token` and vector aliases across direct tier helpers |
