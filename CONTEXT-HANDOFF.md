# Context Handoff: Token Registry Consolidation

## Current State

Working tree is clean. All tests pass.

Recent commits:
```
421621a fix: arcane macro token/trait recognition, exhaustive tests
757c8cc chore: cargo fmt on generated SIMD files and polyfill codegen
02e66bf feat: add bitcast methods and auto-generated polyfills for magetypes
cc2491c chore: bump archmage and archmage-macros to 0.3.1
826845c task: add arcane macro trait-name recognition tests
```

The `TASK-arcane-macro-compat-tests.md` task is complete — all 5 items done.

## Problem: 10+ Independent Copies of Token-Feature Mappings

The codebase has **no single source of truth** for which tokens provide which CPU features. Instead, the same data is independently maintained in 10+ locations. The soundness-critical invariant — that the macro's `#[target_feature]` generation matches what `try_new()` actually verifies — is checked by nothing.

### All locations (with exact file:line refs)

**Soundness-critical (these MUST agree or you get UB):**

1. `archmage-macros/src/lib.rs:79-139` — `token_to_features()`: maps token name → features the macro enables via `#[target_feature]`
2. `archmage-macros/src/lib.rs:147-237` — `trait_to_features()`: same but for trait bounds, with cumulative features
3. `src/tokens/x86.rs`, `src/tokens/x86_avx512.rs`, `src/tokens/arm.rs`, `src/tokens/wasm.rs` — each token's `try_new()` checks specific features via `is_x86_feature_available!()` / `is_aarch64_feature_available!()`

**Data integrity (wrong = wrong behavior but not UB):**

4. `xtask/src/main.rs:27-127` — `token_provides_features()`: 3rd copy of the same mapping, used to validate generated intrinsic code is safe under its gating token
5. `xtask/src/main.rs:220-246` — `MAGETYPES_FILE_MAPPINGS`: assigns tokens to generated files (x86/w128→X64V3Token, etc.)
6. `xtask/src/main.rs:580-597` — `map_features_to_token()`: reverse mapping for doc generation
7. `xtask/src/simd_types/mod.rs:161-300` — `generate_width_namespaces()`: hardcoded `pub type Token = archmage::X64V3Token` strings for sse/avx2/avx512/neon/simd128 modules
8. `xtask/src/simd_types/structure_polyfill.rs:74-96` — `PLATFORMS` const: token assignments for polyfill generation (sse→X64V3Token, neon→NeonToken, simd128→Simd128Token)
9. `xtask/src/simd_types/structure_polyfill.rs:112-118` — `W512_PLATFORMS`: token for W512 polyfill (avx2→X64V3Token)
10. `xtask/src/simd_types/arch/x86.rs:66-79` — `required_token()` and `required_feature()`: width→token mapping
11. `xtask/src/simd_types/arch/arm.rs:67-74` — same for ARM
12. `xtask/src/simd_types/arch/wasm.rs:53-60` — same for WASM

**Hierarchy (wrong = type system lies):**

13. `src/tokens/mod.rs:132-191` — trait definitions (Has128BitSimd, HasX64V2, etc.) and their inheritance relationships
14. `src/tokens/x86.rs:700-774` + `src/tokens/arm.rs:279-299` — `impl HasX64V2 for X64V3Token` etc.
15. `src/tokens/x86_stubs.rs:108-188` + `src/tokens/arm_stubs.rs:44-63` — same for cross-platform stubs

**Test/doc copies (must match but aren't authoritative):**

16. `archmage-macros/src/lib.rs:1395-1440` — `ALL_CONCRETE_TOKENS` and `ALL_TRAIT_NAMES` test lists
17. `archmage-macros/src/lib.rs:833-875` — multiwidth `WidthConfig` structs with token names and target_features arrays
18. `CLAUDE.md` — documentation tables listing tokens, traits, and their feature sets

### The specific soundness gap

If `token_to_features("X64V3Token")` returns `["avx2", "fma", "bmi1", "bmi2"]` but `X64V3Token::try_new()` only checks `avx2` and `fma` (forgetting bmi1/bmi2), then `#[arcane]` generates `#[target_feature(enable = "bmi1")]` for a function that might run on hardware without BMI1. That's UB. **No test catches this class of bug.**

### Granular token gap

The macro recognizes `Avx512fVlToken`, `Avx512bwVlToken`, `Avx512Vbmi2Token`, `Avx512Vbmi2VlToken` in `token_to_features()`, but `xtask/src/main.rs:token_provides_features()` does NOT list these — so xtask can't validate generated code gated by these tokens. The `map_features_to_token()` function for doc generation also doesn't know about the granular tokens.

## Proposed Solution: Single Registry

Create one authoritative data source that defines each token once:

```rust
struct TokenDef {
    name: &'static str,
    aliases: &'static [&'static str],       // e.g., ["Desktop64"] for X64V3Token
    try_new_features: &'static [&'static str], // features try_new() checks
    target_features: &'static [&'static str],  // features for #[target_feature] (concrete position)
    cumulative_features: &'static [&'static str], // all implied features (trait position)
    traits: &'static [&'static str],          // HasX64V2, Has256BitSimd, etc.
    arch: &'static str,                        // "x86_64", "aarch64", "wasm32"
    cfg_feature: Option<&'static str>,         // Some("avx512") for avx512 tokens
}
```

From this, generate:
- `token_to_features()` and `trait_to_features()` in the macro crate
- `token_provides_features()` and `map_features_to_token()` in xtask
- The token struct definitions with `try_new()`
- The marker trait impls
- The width namespace token assignments
- The polyfill platform configs
- The test completeness lists
- The multiwidth WidthConfig arrays

### Where to put the registry

Three options (decision needed from user):

**A. `archmage-registry` data crate** — both `archmage-macros` and `xtask` depend on it. Cleanest, but adds a crate.

**B. TOML data file** — `token-registry.toml` parsed at build time. Flexible but adds parsing deps.

**C. `include!()` shared .rs file** — e.g., `shared/token_registry.rs` with const data, `include!`'d by macro crate and xtask. Zero deps, auditable, slightly unusual.

### Audit surface after consolidation

The entire soundness argument becomes:
1. Read `token_registry.rs` (or equivalent) — one file, ~200 lines
2. Verify `try_new_features` matches what each `try_new()` actually checks
3. Verify `target_features ⊆ try_new_features` (the critical invariant)
4. Everything else is generated — can't drift

## Pre-existing Issues Found

- `magetypes/tests/generated_simd_types.rs:313` — `test_cast_slice_512` panics on `unwrap()` when AVX-512 token is available but `cast_slice` returns None. Pre-existing, not a regression.
- `xtask/src/main.rs:token_provides_features()` is missing the granular AVX-512 tokens (Avx512fVlToken, Avx512bwVlToken, Avx512Vbmi2Token, Avx512Vbmi2VlToken) that exist in the macro and runtime crate.

## How to Continue

1. Delete this file after reading
2. Run `git log --oneline -6` and `cargo test -p archmage -p archmage-macros --features avx512` to verify state
3. Decide which registry approach (A/B/C) to use
4. Implement the single registry and generate all derived copies from it
