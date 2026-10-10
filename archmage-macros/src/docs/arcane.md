Mark a function as an arcane SIMD function.

This macro generates a safe wrapper around a `#[target_feature]` function.
The token parameter type determines which CPU features are enabled.

# Expansion Modes

## Sibling (default)

Generates two functions at the same scope: a safe `#[target_feature]` sibling
and a safe wrapper. `self`/`Self` work naturally since both functions share scope.
Compatible with `#![forbid(unsafe_code)]`.

```ignore
#[arcane]
fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
```

Methods work naturally:

```ignore
impl MyType {
    #[arcane]
    fn compute(&self, token: X64V3Token) -> f32 {
        self.data.iter().sum()  // self/Self just work!
    }
}
```

## Associated functions in an inherent impl (`in_impl`)

An associated function without a receiver cannot call its sibling by bare
name from inside an `impl` block, and the macro cannot see the block. Say
`in_impl`, and the wrapper calls `Self::__arcane_fn(...)`:

```ignore
impl Table {
    #[arcane(in_impl)]
    fn build(token: X64V3Token, n: usize) -> Self { Self::with_capacity(n) }
}
```

Methods with a receiver need no flag.

## Trait impls (`in_trait`, `nested` or `_self = Type`)

A trait impl cannot hold the extra sibling, so the inner function nests
inside the method. `in_trait` (alias `nested`) selects that expansion. A
method with a receiver also needs `_self = Type`, because the nested
function has no `self` and no `Self`: the receiver becomes `_self`, `Self`
becomes the named type, and `self` in the body becomes `_self`, so the body
reads as written:

```ignore
impl SimdOps for MyType {
    #[arcane(in_trait, _self = MyType)]
    fn compute(&self, token: X64V3Token) -> Self {
        Self::new(self.data.iter().sum())
    }
}
```

`_self = Type` implies `in_trait`. `in_impl` and `in_trait` describe
different places and are rejected together.

# Cross-Architecture Behavior

**Default (cfg-out):** On the wrong architecture, the function is not emitted
at all — no stub, no dead code. Code that references it must be cfg-gated.

`stub` has been removed. Use `incant!` or explicit call-site cfg guards.

# Token Parameter Forms

```ignore
// Concrete token
#[arcane]
fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { ... }

// impl Trait bound
#[arcane]
fn process(token: impl HasX64V2, data: &[f32; 8]) -> [f32; 8] { ... }

// Generic with inline or where-clause bounds
#[arcane]
fn process<T: HasX64V2>(token: T, data: &[f32; 8]) -> [f32; 8] { ... }

// Wildcard
#[arcane]
fn process(_: X64V3Token, data: &[f32; 8]) -> [f32; 8] { ... }
```

Exactly one parameter may be a token; two token parameters are a compile
error, because the wrapper would not know which proof to assert. Wildcard
and tuple patterns on other parameters are renamed to `__archmage_arg_N` in
the wrapper's signature so they can be forwarded.

# Options

| Option | Effect |
|--------|--------|
| `in_impl` | Receiver-less associated function in an inherent impl: the wrapper calls `Self::__arcane_fn` |
| `in_trait` (alias `nested`) | Trait impl: the inner function nests inside the method |
| `_self = Type` | Implies `in_trait`; the receiver becomes `_self`, `Self` and `self` are rewritten |
| `inline_always` | Use `#[inline(always)]` (requires nightly) |
| `import_intrinsics` | Auto-import `archmage::intrinsics::{arch}::*` (includes safe memory ops) |
| `import_magetypes` | Auto-import `magetypes::simd::{ns}::*` and `magetypes::simd::backends::*` |

## Auto-Imports

`import_intrinsics` and `import_magetypes` inject `use` statements into the
function body, eliminating boilerplate. The macro derives the architecture and
namespace from the token type:

```ignore
// Without auto-imports — lots of boilerplate:
use std::arch::x86_64::*;
use magetypes::simd::v3::*;

#[arcane]
fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load_t(token, data);
    let zero = _mm256_setzero_ps();
    // ...
}

// With auto-imports — clean:
#[arcane(import_intrinsics, import_magetypes)]
fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load_t(token, data);
    let zero = _mm256_setzero_ps();
    // ...
}
```

The namespace mapping is token-driven:

| Token | `import_intrinsics` | `import_magetypes` |
|-------|--------------------|--------------------|
| `X64V1..V3Token` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v3::*` |
| `X64V4Token` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v4::*` |
| `X64V4xToken` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v4x::*` |
| `NeonToken` / ARM | `archmage::intrinsics::aarch64::*` | `magetypes::simd::neon::*` |
| `Wasm128Token` | `archmage::intrinsics::wasm32::*` | `magetypes::simd::wasm128::*` |

Works with concrete tokens, `impl Trait` bounds, and generic parameters.

# Supported Tokens

- **x86_64**: `X64V2Token`, `X64V3Token`/`Desktop64`, `X64V4Token`/`Avx512Token`/`Server64`,
  `X64V4xToken`, `Avx512Fp16Token`, `X64CryptoToken`, `X64V3CryptoToken`,
  `X64V3GfniCryptoToken`
- **ARM**: `NeonToken`/`Arm64`, `Arm64V2Token`, `Arm64V3Token`,
  `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken`
- **WASM**: `Wasm128Token`

# Supported Trait Bounds

`HasX64V2`, `HasX64V4`, `HasNeon`, `HasNeonAes`, `HasNeonSha3`, `HasArm64V2`, `HasArm64V3`

```ignore
#![feature(target_feature_inline_always)]

#[arcane(inline_always)]
fn fast_kernel(token: Avx2Token, data: &mut [f32]) {
    // Inner function will use #[inline(always)]
}
```

Concrete tokens are checked through a shared, tier-specific associated constant.
This rejects accidental token-name aliases without reevaluating a tag comparison
in every expansion. The matching archmage release pins this macro crate exactly.
Getting past the check takes deliberately shadowing archmage's type names and
copying their hidden constants, which gives undefined behavior on CPUs without
the features; see the [safety model](https://imazen.github.io/archmage/archmage/concepts/safety/).

`#[arcane(suppress_const_test)]` omits this accidental-misuse check for trusted
generators. The caller must ensure the actual token proves the tier selected
by its name. Intrinsic target-feature checking remains enabled, but does not
authenticate that token: it checks instructions against the generated feature
context. Ordinary callers should keep the default check enabled. This option
is intended for generators whose registry already establishes that match.
