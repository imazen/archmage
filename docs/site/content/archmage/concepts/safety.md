+++
title = "Safety Model"
description = "Why safe code can call SIMD intrinsics through archmage, and what that relies on"
weight = 6
+++

archmage lets safe Rust call SIMD intrinsics, and a crate that uses it can keep
`#![forbid(unsafe_code)]`. What you need to know as a user:

- A token such as `X64V3Token` proves the CPU has its features. Get one from
  `summon()`, or let `incant!` and `#[magetypes]` get it for you.
- Enter feature-enabled code through `#[arcane]`, `#[magetypes]` or `incant!`,
  and call `#[rite]` helpers from there.
- Don't use `unsafe` to make a token. An `unsafe` block is where you take over
  the proof yourself.

The rest of this page explains why that is enough, what the guarantee relies
on, and how it is checked. For an audit, start from
[SOUNDNESS.md](https://github.com/imazen/archmage/blob/main/docs/SOUNDNESS.md),
which inventories every `unsafe` in archmage and magetypes, and
[AUDITING.md](https://github.com/imazen/archmage/blob/main/AUDITING.md). The
[magetypes safety model](@/magetypes/safety.md) covers the vector types.

## The invariant

Every intrinsic that needs a CPU feature runs where that feature is proven
present. A proof is a token value or a `#[target_feature]` function.

Running an instruction the CPU lacks is undefined behavior. Usually the CPU
raises an invalid-opcode fault and the process dies with `SIGILL`. A few
instructions, such as `lzcnt` on CPUs without it, decode as an older
instruction and return a different result instead.

## Tokens

A token such as `X64V3Token` is a zero-sized struct with a private field, so
code outside archmage cannot build one with a struct literal. The `SimdToken`
trait is sealed, so no other type can implement it. Safe code gets a token in
three ways:

- `X64V3Token::summon()` checks the CPU and returns `None` when a feature is
  missing. On x86 that includes checking that the operating system saves AVX
  state. When the build already guarantees the features, for example with
  `-Ctarget-cpu=x86-64-v3`, the check compiles away.
- A stronger token converts to a weaker one: `v4.v3()`.
- `X64V3Token::from_context()` is itself a `#[target_feature]` function, so
  rustc allows a safe call only from a function whose own `#[target_feature]`
  attribute covers V3, such as an `#[arcane]` or `#[rite]` body. Global flags
  such as `-Ctarget-cpu=native` do not count.

From an ordinary function the call needs `unsafe`, so this does not compile:

```rust,compile_fail
#![forbid(unsafe_code)]
use archmage::X64V3Token;

fn forge() -> X64V3Token {
    X64V3Token::from_context()
}
```

Inside a matching context the same call is safe:

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[rite(v3)]
fn reprove() -> X64V3Token {
    X64V3Token::from_context()
}

#[arcane]
fn entry(_token: X64V3Token) -> X64V3Token {
    reprove()
}

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let _same: X64V3Token = entry(token);
}
```

## What `#[arcane]` generates

On x86-64, this entry point:

```rust
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[arcane]
fn process(token: X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    assert_eq!(process(token, 1.0, 2.0), 3.0);
}
```

expands to a `#[target_feature]` function and a wrapper. This is the committed
expansion snapshot,
[`tests/expand/arcane/sibling.expanded.rs`](https://github.com/imazen/archmage/blob/main/tests/expand/arcane/sibling.expanded.rs):

```text
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}
#[inline(always)]
fn process(token: X64V3Token, a: f32, b: f32) -> f32 {
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_process(token, a, b) }
}
```

Calling a `#[target_feature]` function from a function without those features
is `unsafe` because the CPU might lack them. The wrapper takes an
`X64V3Token` by value, and safe code gets archmage's `X64V3Token` only where
the features are present, so the `unsafe` block holds. The wrapper's first
line checks the parameter's type for archmage's V3 tier constant, so a local
struct named `X64V3Token`, or a weaker token imported under that name, fails to
compile.

The inner function is a safe `fn`. Inside it, value intrinsics such as
`_mm256_add_ps` are safe to call (Rust 1.87 and later). Intrinsics that take
raw pointers still need `unsafe`, so `import_intrinsics` brings in versions
that take references: `_mm256_loadu_ps(&[f32; 8])` instead of a
`*const f32`. A token proves CPU features and nothing about memory; bounds and
aliasing come from Rust references.

`#[rite]` emits no wrapper. It puts `#[target_feature]` and `#[inline]` on the
function itself, and rustc allows safe calls to it only from functions with the
same or more features. The caller's attribute is the proof.

## `#![forbid(unsafe_code)]` in your crate

The `unsafe_code` lint skips code generated by external procedural macros, so
the wrapper's `unsafe` block does not count against your crate. A crate that
uses `#[arcane]`, `#[rite]`, `incant!`, magetypes and the reference-taking
memory intrinsics can forbid `unsafe` entirely, as the examples on this page do.

## What the guarantee relies on

- **No `unsafe` of your own around tokens.** `unsafe {
  X64V3Token::from_context() }` in an ordinary function, or the deprecated
  `forge_token_dangerously()`, makes you the proof.
- **Detection is trusted.** x86 uses Rust's `std_detect` (CPUID, plus the check
  that the OS saves AVX and AVX-512 state); without the `std` feature, only
  compile-time features count. AArch64 uses `std_detect`, register decoding
  through `winarm-cpufeatures` on Windows, and a fixed Apple Silicon baseline
  on macOS, Mac Catalyst and simulators. A wrong answer from any of them is a
  wrong token.
- **WASM** has no runtime detection. The engine validates a module's
  instructions when it loads, so a module that runs has its features.
- **`from_context()` skips detection**, so it also ignores tiers disabled for
  testing with `testable_dispatch`. Use `summon()` where dispatch has to follow
  runtime state.
- **`suppress_const_test`** removes the type check. It exists for code
  generators that emit real type paths, such as `#[magetypes]`.
- **No deliberate evasion of the check.** Accidental collisions fail to
  compile: a local type named `X64V3Token`, or a weaker token imported under
  that name, lacks the hidden tier constant the wrapper checks. Getting past
  the check without `unsafe` means deliberately shadowing archmage's type names
  and copying its hidden constants onto your own types. What you get is
  undefined behavior in your own functions on CPUs without the features.
  Usually that's a `SIGILL` crash, but not always: on a CPU without LZCNT or
  BMI1, `lzcnt` and `tzcnt` run as older instructions and return wrong values
  instead of faulting.

## How it's checked

- `cargo xtask soundness` checks every intrinsic call in archmage, magetypes and
  the macro expansion snapshots against the features of its enclosing function
  or token. Its reference is 16,371 intrinsics (17,312 signatures) extracted from Rust's
  `stdarch` sources. In 0.9.30 it verifies 5,493 calls in 11 files, scanning
  274. It fails if the count drops below 4,000, so it cannot pass by seeing
  nothing.
- [`tests/soundness/`](https://github.com/imazen/archmage/tree/main/tests/soundness)
  holds a compile-fail case for each bypass the macros block:
  - shadowed and aliased token and trait names
  - `from_context()` called from a missing, weaker or foreign-architecture
    context
  - `from_context` coerced to a safe function pointer
- `cargo xtask validate` checks that each `summon()` tests exactly the features
  that `token-registry.toml` declares, the file every feature list is
  generated from.
- Exercise tests call each token's intrinsics on real hardware, under QEMU for
  AArch64, under wasmtime, and under Intel SDE for AVX-512 FP16.
