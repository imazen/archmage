# Safety-Critical Code Audit

This document identifies non-generated code that is safety-critical. Changes to these
areas require extra scrutiny as they can lead to undefined behavior if incorrect.

Run `cargo xtask audit` to scan for safety-critical code and verify invariants.

## Criticality Levels

- **CRITICAL**: Incorrect code causes UB directly. Requires proof of correctness.
- **FRAGILE**: Incorrect code can cause UB indirectly or silently break safety invariants.
- **SENSITIVE**: Affects security properties but may not cause UB.

## Safety-Critical Areas

### 1. Token Forge Functions (CRITICAL)

**File**: `src/tokens/mod.rs`

The `forge_token_dangerously()` method creates tokens without feature verification.
If called when CPU features are not available, subsequent SIMD operations cause UB.

**Invariant**: Every `forge_token_dangerously()` call must be inside:
- A `summon()` that verified features, OR
- A function with `#[target_feature]` matching the token's requirements, OR
- Generated code inside `#[arcane]`/`#[multiwidth]` macros

**Audit command**: `cargo xtask audit --forge-calls`

---

### 2. Macro Token Extraction (CRITICAL)

**File**: `archmage-macros/src/lib.rs`

The `#[arcane]` macro extracts the token type from function parameters and generates
`#[target_feature(enable = "...")]`. If it extracts wrong features, generated code
will execute SIMD instructions without proper CPU support.

**Invariant**: Token type → feature mapping must match `token-registry.toml` exactly.

**Verified by**: `cargo xtask validate` (checks registry against macro registry)

---

### 3. summon() Feature Checks (CRITICAL)

**File**: `xtask/src/token_gen.rs` → generates `src/tokens/generated/*.rs`

Each token's `summon()` must check ALL required CPU features before calling
`forge_token_dangerously()`. Missing a feature check allows token creation on
unsupported hardware.

**Invariant**: Features checked in `summon()` must match `token-registry.toml`.

**Verified by**: `cargo xtask validate` (step 4 of CI)

---

### 4. Unsafe Memory Operations (CRITICAL)

**Files**:
- `xtask/src/simd_types/structure.rs` (x86)
- `xtask/src/simd_types/structure_arm.rs` (ARM)
- `xtask/src/simd_types/structure_wasm.rs` (WASM)

These generate `cast_slice`, `from_bytes`, `as_bytes`, and bitcast operations that
use `unsafe` transmutes. Incorrect size/alignment assumptions cause UB.

**Invariants**:
- `cast_slice`: Must verify alignment AND length divisibility
- `from_bytes`: Array size must equal `size_of::<Self>()`
- Bitcasts: Both types must have identical size and compatible alignment

**Verified by**: Miri boundary tests (`magetypes/tests/miri_boundary_tests.rs`)

---

### 5. Intrinsic-Token Mapping (CRITICAL)

**File**: `archmage-macros/src/generated/registry.rs`

Maps token names to their required target features. If wrong, `#[arcane]` generates
incorrect `#[target_feature]` attributes.

**Invariant**: Must match `token-registry.toml` exactly.

**Verified by**:
- `cargo xtask validate` (regenerates and compares)
- `cargo xtask soundness` (validates 3409 intrinsic calls against stdarch)

---

### 6. Intrinsic Selection in Codegen (FRAGILE)

**Files**: `xtask/src/simd_types/*.rs`

When generating SIMD type methods, we select intrinsics based on the gating token.
Using an intrinsic that requires higher features than the token provides causes UB.

**Invariant**: All intrinsics used in a method must be available with the method's token.

**Verified by**: `cargo xtask soundness` (static analysis against stdarch database)

---

### 7. Polyfill Implementations (FRAGILE)

**Files**: `xtask/src/simd_types/structure*.rs` (polyfill sections)

Polyfills emulate missing intrinsics using available ones. Incorrect polyfills
silently produce wrong results.

**Invariant**: Polyfill output must match native intrinsic output for all inputs.

**Verified by**: `magetypes/tests/polyfill_parity.rs`

---

## Audit Markers

Use these comments to mark safety-critical code:

```rust
// CRITICAL: <explanation of why this is critical>
// FRAGILE: <explanation of what could break>
// SAFETY: <explanation of why this unsafe block is sound>
```

The `cargo xtask audit` command scans for these markers and reports:
- CRITICAL/FRAGILE without corresponding SAFETY justification
- Unsafe blocks without SAFETY comments
- forge_token_dangerously calls outside approved contexts

---

## Adding New Safety-Critical Code

1. Add `// CRITICAL:` or `// FRAGILE:` comment explaining the risk
2. Add `// SAFETY:` comment justifying correctness
3. Add to this document if it's a new category
4. Ensure CI verification exists (test, Miri, or static analysis)
5. Get review from maintainer

---

## Verification Matrix

| Area | Static Check | Runtime Test | Miri | Fuzzing |
|------|-------------|--------------|------|---------|
| Token forge | `xtask validate` | feature_consistency | N/A | N/A |
| Macro extraction | `xtask validate` | compile_fail tests | N/A | N/A |
| summon() checks | `xtask validate` | feature_consistency | N/A | N/A |
| Memory ops | `xtask soundness` | boundary tests | ✓ | TODO |
| Intrinsic mapping | `xtask soundness` | N/A | N/A | N/A |
| Polyfills | N/A | polyfill_parity | ✓ | TODO |
