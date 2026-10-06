# Auditing Guide

Files that human reviewers should examine for safety-critical code.

## Safety Model

- **`src/tokens/mod.rs`** — `SimdToken` trait (the core safety contract), `ScalarToken`'s constructors, `CompileTimeGuaranteedError`, `IntoConcreteToken` dispatch

## Generated Token Implementations

All generated from `token-registry.toml` via `cargo xtask generate`.

- **`src/tokens/generated/x86.rs`** — every x86 token, V1 through Avx512Fp16 (the AVX-512 tokens are not feature-gated): runtime detection via `is_x86_feature_detected!`, atomic caching, the `from_context()` constructor and its deprecated alias, disable mechanism with cascading
- **`src/tokens/generated/arm.rs`** — AArch64 tokens: runtime detection via `is_aarch64_feature_detected!`, same caching/disable pattern
- **`src/tokens/generated/wasm.rs`** — WASM token: compile-time only (no runtime detection on wasm32)

## Generated Stubs

Cross-platform stubs where `summon()` always returns `None`.

- **`src/tokens/generated/x86_stubs.rs`** — x86 stubs, AVX-512 tokens included (used on ARM/WASM)
- **`src/tokens/generated/arm_stubs.rs`** — ARM stubs (used on x86/WASM)
- **`src/tokens/generated/wasm_stubs.rs`** — WASM stubs (used on x86/ARM)

## Macro Safety

- **`archmage-macros/src/arcane.rs`** — `#[arcane]` generates a `#[target_feature(enable = "...")]` sibling (or nested) function and a wrapper whose `unsafe` call is sound if and only if the token proves the features. The wrapper also asserts that the token is archmage's own type (`__ARCHMAGE_ASSERT_TIER_*`), and `common.rs` restates tier-trait bounds through `::archmage::` paths.
- **`archmage-macros/src/rite.rs`** — `#[rite]` adds `#[target_feature]` + `#[inline]` directly and emits no `unsafe`; rustc checks its callers. `lib.rs` holds the macro entry points.

## magetypes

- **`magetypes/src/simd_storage.rs`** — every `unsafe` in magetypes: the `Pod` and `TokenStorage` contracts, the byte copies and views, the AVX-512 gather/scatter helpers with their lane bounds, and the public `Upcast` trait, which declares an `unsafe fn`. `TokenStorage` is implemented only through this file's `impl_token_storage!`, whose expansion checks each wrapper's layout at compile time. The crate root denies `unsafe_code` and allows it for this module alone; `cargo xtask soundness` also rejects the `unsafe` keyword, any other `allow(unsafe_code)`, and gather/scatter intrinsics anywhere else in magetypes, including code cfg'd out for the host.
- **`magetypes/src/simd/generic/generated/*_impl.rs`** — one `impl_token_storage!` invocation per vector type, next to its `#[repr(C)]` struct.
- **`magetypes/src/simd/impls/*.rs`** — generated backends with no `unsafe`; every intrinsic call sits in an `#[arcane]` region, for the receiver's token or, for SSE2-only operations, `X64V1Token`.

The full `unsafe` inventory, with the verification behind each entry, is in [`docs/SOUNDNESS.md`](docs/SOUNDNESS.md).

## Code Generators

- **`xtask/src/token_gen.rs`** — Generates all token structs, `SimdToken` impls, detection logic, disable mechanism, and cascading. Wrong code here = wrong safety guarantees everywhere.
- **`xtask/src/registry.rs`** — Parses and validates `token-registry.toml`. Validates feature set consistency.

## Source of Truth

- **`token-registry.toml`** — Defines every token's feature set, trait memberships, and hierarchy. If a feature is missing here, the generated `#[target_feature]` won't include it, and LLVM won't use the corresponding instructions. If an extra feature is listed, the token will demand more than necessary (safe but overly restrictive).

## Verification Tests

- **`tests/miri_safe.rs`** — Miri-clean tests for token creation, disable, and basic operations
- **`magetypes/tests/miri_boundary_tests.rs`** — Miri tests for SIMD type boundary conditions
- **`tests/token_infrastructure.rs`** — Comprehensive token system tests: `compiled_with()`, `summon()`, disable/cascading, `CompileTimeGuaranteedError`, feature flag strings
- **`tests/soundness_exploits.rs`** with **`tests/soundness/*.rs`** — 26 attacks (token shadowing and aliasing, sealed-trait bypass, tier-trait shadowing, `from_context()`/`from_raw()` outside a covering context, storage size/alignment/token misuse), each compiled and asserted to fail with a specific error code (two cases are asserted to compile)
- **`magetypes/src/bypass_adversarial.rs`** — 20 `compile_fail` doctests calling backend methods through UFCS without a token
- **`xtask/src/soundness.rs`** and **`xtask/src/soundness/raw_context.rs`** — the intrinsic-feature scanner and the `from_raw` context check

- **`tests/from_context.rs`** — the safe forge path under `#![forbid(unsafe_code)]`; **`tests/soundness/from_context_*.rs` (driven by `tests/soundness_exploits.rs`)** — the four rejected forges (no context, weaker context, fn-pointer coercion, foreign architecture)
- **`tests/token_downcast.rs`** — extraction methods and `IntoConcreteToken` for every token, including forged stubs

## What to Look For

1. **Every `unsafe` block** in generated token code is an `unsafe` call to `from_context()` from a feature-free function. `from_context()` is a safe `#[target_feature]` fn, so a caller *inside* a matching region needs no `unsafe` and rustc does the checking; these call sites (`summon()`, the cold detect fns, the extraction methods, `IntoConcreteToken`) are not in such a region, so they discharge the obligation by hand. Verify each against the adjacent `// SAFETY:` comment: detection just succeeded, the features are compile-time guaranteed, or the source token's feature set is a registry-verified superset.
2. **Feature lists** in `token-registry.toml` must match LLVM's x86-64 microarchitecture levels. Missing features = potential UB (code uses instructions the CPU might not have).
3. **`#[target_feature]` strings** generated by `#[arcane]` macro must match the token's feature set. The macro reads from `archmage-macros/src/generated/registry.rs`.
4. **Disable cascading** must go parent → all descendants. If V3 is disabled but V4 isn't, code could summon a V4 token that implies V3 features on a CPU that lacks them. (Note that `from_context()` deliberately ignores disabling — it runs no detection, and a `#[target_feature]` region cannot be made to stop having its features.)
5. **Stubs** must never return `Some` from `summon()`. A stub returning a token on the wrong architecture = instant UB.
