# Task: Add Macro Trait-Name Recognition Tests

## Problem

archmage 0.3.0 broke all downstream `#[arcane]` functions that used generic trait bounds (`HasAvx`, `HasAvx2`, `HasFma`, `HasAvx512f`). These traits were removed from the runtime crate AND from the proc macro's `trait_to_features()` recognition map, but:

1. No test caught the breakage before publish
2. The `#[arcane]` error message still shows `HasAvx2` as a valid example
3. No migration guide documented the breaking change
4. Downstream code got 15+ cryptic "arcane requires a token parameter" errors

**Root cause:** The proc macro (`archmage-macros/src/lib.rs`) has a `trait_to_features()` function that maps trait bound names to target features. When traits were removed from the runtime crate in 0.3.0, nobody updated the macro's recognition map to keep them as aliases, AND nobody added tests verifying which trait names the macro accepts.

## What Needs to Change

### 1. Fix the misleading error message

In `archmage-macros/src/lib.rs:430-435`, the error message shows:
```
- impl Trait: `token: impl HasAvx2`
- Generic: `fn foo<T: HasAvx2>(token: T, ...)`
```

But `HasAvx2` is NOT a valid trait bound in 0.3.0. The error should show actual supported forms:
```
- Concrete: `token: X64V3Token`
- impl Trait: `token: impl Has256BitSimd`
- Generic: `fn foo<T: HasX64V2>(token: T, ...)`
```

### 2. Add exhaustive trait-name recognition tests

Create `tests/arcane_trait_recognition.rs` that tests every form the macro should accept. For each supported trait/token name, define an `#[arcane]` function and verify it compiles and runs.

**All token names** (from `token_to_features()`):
- `Sse41Token`, `Sse42Token`, `AvxToken`, `Avx2Token`, `FmaToken`, `Avx2FmaToken`
- `X64V2Token`, `X64V3Token`, `Desktop64`
- `Avx512fToken`, `Avx512bwToken`, `Avx512fVlToken`, `Avx512bwVlToken`
- `Avx512Vbmi2Token`, `Avx512Vbmi2VlToken`
- `X64V4Token`, `Avx512Token`, `Server64`
- `Avx512ModernToken`, `X64V4ModernToken`
- `Avx512Fp16Token`
- `NeonToken`, `Arm64`, `NeonAesToken`, `NeonSha3Token`
- `ArmCryptoToken`, `ArmCrypto3Token`
- `Wasm128Token`

**All trait names** (from `trait_to_features()`):
- `HasX64V2`, `HasX64V4`
- `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`
- `HasNeon`, `HasNeonAes`, `HasNeonSha3`

Test patterns for each:
```rust
// Concrete token
#[arcane]
fn with_concrete(token: X64V3Token, ...) { ... }

// impl Trait
#[arcane]
fn with_impl(token: impl Has256BitSimd, ...) { ... }

// Generic inline bounds
#[arcane]
fn with_generic<T: HasX64V2>(token: T, ...) { ... }

// Generic where clause
#[arcane]
fn with_where<T>(token: T, ...) where T: HasX64V2 { ... }
```

### 3. Add compile-fail tests for rejected forms

Create `tests/compile_fail/arcane_rejects_unknown_traits.rs`:
```rust
// These should fail with a clear error, not compile silently
#[arcane]
fn bad_trait(token: impl HasAvx2, ...) { ... }  // HasAvx2 not a real trait

#[arcane]
fn bad_trait2<T: HasFma>(token: T, ...) { ... }  // HasFma not a real trait
```

### 4. Add a `token_to_features` ↔ `trait_to_features` consistency check

Add a unit test in `archmage-macros` (or integration test) that verifies:
- Every concrete token struct exported from `archmage` has an entry in `token_to_features()`
- Every trait exported from `archmage` has an entry in `trait_to_features()`
- The feature sets are consistent (e.g., X64V3Token and Avx2FmaToken map to the same features)

### 5. Either add backward-compat aliases OR document migration

**Option A (preferred):** Add the old trait names as aliases in `trait_to_features()`:
```rust
// Backward compat - these are recognized by the macro even though
// the traits don't exist in the runtime crate anymore.
// They map to the closest modern equivalent.
"HasAvx" | "HasAvx2" => Some(&["avx", "avx2"]),  // closest: Has256BitSimd
"HasFma" => Some(&["fma"]),
"HasAvx512f" => Some(&["avx512f"]),  // closest: Has512BitSimd
```

The macro would generate `#[target_feature]` correctly, and the trait bounds would fail at the Rust type-checker level (since the traits don't exist), giving users a clear "trait not found" error instead of the cryptic "arcane requires a token parameter" error.

**Option B:** Add a migration guide to README.md and CHANGELOG.md documenting:
- `HasAvx` → use `Has256BitSimd` or concrete `AvxToken`
- `HasAvx2 + HasFma` → use concrete `X64V3Token` or `Avx2FmaToken`
- `HasAvx512f` → use `Has512BitSimd` or concrete `X64V4Token`
- `archmage::mem::avx` → use `safe_unaligned_simd::x86_64`

## Priority

High - this broke ALL downstream code using generic `#[arcane]` functions on the 0.2→0.3 upgrade. The fix took significant debugging time because the error message was misleading.
