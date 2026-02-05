# Polyfill Architecture Design

This document outlines the architecture for cross-platform SIMD polyfills in archmage.

## Current State

### Platforms with Tokens (capability proofs)
| Platform | Token | SIMD Types |
|----------|-------|------------|
| x86_64 SSE4.1 | `Sse41Token` | ✅ f32x4, i32x4, etc. |
| x86_64 AVX2+FMA | `X64V3Token` | ✅ f32x8, i32x8, etc. |
| x86_64 AVX-512 | `X64V4Token` | ✅ f32x16, i32x16, etc. |
| AArch64 NEON | `NeonToken` | ❌ Not yet |
| WebAssembly SIMD128 | `Wasm128Token` | ❌ Not yet |

### Current Polyfill (hand-written)
- `polyfill::v3::f32x8` - emulates AVX2 f32x8 using two SSE f32x4
- Works but has maintainability concerns

## Design Goals

1. **Consistent API** - Same operations available on all platforms
2. **Automatic polyfills** - Write once, run everywhere (at varying speeds)
3. **Maintainability** - Single source of truth for operations
4. **Verifiable correctness** - Automated tests comparing polyfill to native

## Architecture

### 1. Platform Matrix

```
Target Code Width     Native Backends
─────────────────────────────────────────────────────────────
f32x4 (128-bit)  →  SSE4.1, NEON, SIMD128
f32x8 (256-bit)  →  AVX2 native, OR 2×f32x4 polyfill
f32x16 (512-bit) →  AVX-512 native, OR 2×f32x8, OR 4×f32x4
```

### 2. Polyfill Generation Strategy

Instead of hand-writing polyfills, we should **generate them** from the native type definitions.

```rust
// In xtask/src/simd_types/polyfill.rs

/// Defines how to emulate a wider type using narrower ones
struct PolyfillConfig {
    /// Target type to emulate (e.g., f32x8)
    target: SimdType,
    /// Base type to use (e.g., f32x4)
    base: SimdType,
    /// Number of base vectors needed
    count: usize,  // 2 for f32x8 → 2×f32x4
}

/// Generate polyfill implementation
fn generate_polyfill(config: &PolyfillConfig) -> String {
    // Auto-generate based on base type's operations
}
```

### 3. Operation Categories

| Category | Polyfill Strategy | Example |
|----------|------------------|---------|
| Element-wise | Apply to each part | `add`, `mul`, `sqrt` |
| Horizontal | Reduce parts, then combine | `reduce_add`, `reduce_max` |
| Shuffle/Permute | Complex - may need special handling | `blend`, `permute` |
| Comparison | Apply to each part | `simd_eq`, `simd_lt` |
| Load/Store | Split/combine arrays | `load`, `store` |

### 4. Cross-Platform Type System

```rust
// NOTE: This was an early design concept. The actual implementation uses
// per-platform generated types + polyfills, not a single portable struct.
// See magetypes/src/simd/generated/ for the real implementation.
```

## ARM NEON Support

### Token Hierarchy (already exists)
```
NeonToken (baseline, always available on AArch64)
├── NeonAesToken (+ AES)
├── NeonSha3Token (+ SHA3)
├── ArmCryptoToken (AES + SHA2 + CRC)
└── ArmCrypto3Token (+ SHA3)
```

### NEON Type Generation

Add to `xtask/src/simd_types/arch/`:

```rust
// xtask/src/simd_types/arch/arm.rs

pub fn neon_inner_type(ty: &SimdType) -> &'static str {
    match (ty.elem, ty.width) {
        (ElementType::F32, SimdWidth::W128) => "float32x4_t",
        (ElementType::F64, SimdWidth::W128) => "float64x2_t",
        (ElementType::I32, SimdWidth::W128) => "int32x4_t",
        (ElementType::I16, SimdWidth::W128) => "int16x8_t",
        (ElementType::I8, SimdWidth::W128) => "int8x16_t",
        // ...
    }
}

pub fn neon_intrinsic(op: &str, ty: &SimdType) -> String {
    // vaddq_f32, vmulq_f32, etc.
}
```

### NEON Polyfill (256-bit on 128-bit NEON)

Since NEON is 128-bit only, f32x8 would use the same 2×f32x4 pattern:

```rust
// polyfill::neon::f32x8
pub struct f32x8 {
    lo: neon::f32x4,
    hi: neon::f32x4,
}
```

## WebAssembly SIMD128 Support

### Token (already exists)
```rust
Wasm128Token // 128-bit SIMD, always compile-time known
```

### WASM Type Generation

```rust
// xtask/src/simd_types/arch/wasm.rs

pub fn wasm_inner_type(ty: &SimdType) -> &'static str {
    // WASM SIMD uses v128 for everything
    "v128"
}

pub fn wasm_intrinsic(op: &str, ty: &SimdType) -> String {
    match (op, ty.elem) {
        ("add", ElementType::F32) => "f32x4_add",
        ("mul", ElementType::F32) => "f32x4_mul",
        ("add", ElementType::I32) => "i32x4_add",
        // ...
    }
}
```

## Verification Strategy

### 1. Unit Tests (Exact Match)

```rust
#[test]
fn polyfill_matches_native_add() {
    if let Some(avx_token) = X64V3Token::summon() {
        if let Some(sse_token) = Sse41Token::summon() {
            let data_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let data_b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

            // Native AVX2
            let native_a = native::f32x8::load(avx_token, &data_a);
            let native_b = native::f32x8::load(avx_token, &data_b);
            let native_result = (native_a + native_b).to_array();

            // Polyfill (2×SSE)
            let poly_a = polyfill::v3::f32x8::load(sse_token, &data_a);
            let poly_b = polyfill::v3::f32x8::load(sse_token, &data_b);
            let poly_result = (poly_a + poly_b).to_array();

            assert_eq!(native_result, poly_result);
        }
    }
}
```

### 2. Property-Based Testing (Fuzzing)

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn polyfill_add_commutative(
        a in prop::array::uniform8(-1e6f32..1e6f32),
        b in prop::array::uniform8(-1e6f32..1e6f32),
    ) {
        if let Some(token) = Sse41Token::summon() {
            let va = polyfill::v3::f32x8::load(token, &a);
            let vb = polyfill::v3::f32x8::load(token, &b);

            let result1 = (va + vb).to_array();
            let result2 = (vb + va).to_array();

            for i in 0..8 {
                prop_assert!((result1[i] - result2[i]).abs() < 1e-6);
            }
        }
    }
}
```

### 3. Cross-Platform CI

```yaml
# .github/workflows/cross-platform.yml
jobs:
  test-x86:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test --features polyfill-verification

  test-arm:
    runs-on: ubuntu-24.04-arm
    steps:
      - run: cargo test --features polyfill-verification

  test-wasm:
    runs-on: ubuntu-latest
    steps:
      - run: cargo install wasm-pack
      - run: wasm-pack test --node
```

### 4. Accuracy Tests for Transcendentals

```rust
#[test]
fn polyfill_exp_accuracy() {
    if let Some(token) = Sse41Token::summon() {
        // Test against std::f32::exp
        let inputs = [-10.0, -1.0, 0.0, 1.0, 10.0, 88.0];
        for input in inputs {
            let v = polyfill::v3::f32x8::splat(token, input);
            let result = v.exp_lowp().to_array();
            let expected = input.exp();

            for &r in &result {
                let rel_err = ((r - expected) / expected).abs();
                assert!(rel_err < 0.001, "exp({}) = {} vs {}", input, r, expected);
            }
        }
    }
}
```

## Maintainability Approach

### 1. Single Source of Truth

Operations are defined once in `xtask/src/simd_types/ops.rs` as **abstract operations**:

```rust
enum SimdOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Sqrt,
    Abs,
    // ...
}

impl SimdOp {
    /// How this op polyfills (element-wise, horizontal, etc.)
    fn polyfill_strategy(&self) -> PolyfillStrategy {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => PolyfillStrategy::Elementwise,
            Self::Min | Self::Max => PolyfillStrategy::Elementwise,
            Self::ReduceAdd | Self::ReduceMax => PolyfillStrategy::Horizontal,
            // ...
        }
    }

    /// Generate code for this op on a given platform
    fn codegen(&self, platform: Platform, ty: &SimdType) -> String;
}
```

### 2. Generated Tests

Tests are generated from the same operation definitions:

```rust
fn generate_polyfill_tests() -> String {
    let mut code = String::new();

    for op in all_ops() {
        code.push_str(&generate_test_for_op(op));
    }

    code
}
```

### 3. API Contract

All SIMD types must implement a trait that guarantees API consistency:

```rust
/// Common operations all SIMD float types must provide
pub trait SimdFloat: Sized + Copy {
    type Token: SimdToken;
    const LANES: usize;
    type Element;

    fn load(token: Self::Token, data: &[Self::Element]) -> Self;
    fn splat(token: Self::Token, v: Self::Element) -> Self;
    fn zero(token: Self::Token) -> Self;
    fn store(self, out: &mut [Self::Element]);
    fn to_array(self) -> Vec<Self::Element>;

    // Math
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;

    // Horizontal
    fn reduce_add(self) -> Self::Element;
    fn reduce_max(self) -> Self::Element;
    fn reduce_min(self) -> Self::Element;
}
```

### 4. Compile-Time Verification

Use const assertions to verify polyfill correctness:

```rust
// Verify lane counts match
const _: () = assert!(polyfill::v3::f32x8::LANES == native::f32x8::LANES);

// Verify size matches
const _: () = assert!(
    core::mem::size_of::<polyfill::v3::f32x8>() ==
    core::mem::size_of::<native::f32x8>()
);
```

## Implementation Phases

### Phase 1: Refactor Polyfill Generation (Current)
- [ ] Move polyfill to generated code
- [ ] Add polyfill verification tests
- [ ] Ensure SSE↔AVX2 parity

### Phase 2: ARM NEON Types
- [ ] Add `xtask/src/simd_types/arch/arm.rs`
- [ ] Generate `src/simd/arm/` types
- [ ] Add NEON polyfill for 256-bit

### Phase 3: WebAssembly SIMD128
- [ ] Add `xtask/src/simd_types/arch/wasm.rs`
- [ ] Generate `src/simd/wasm/` types
- [ ] Add WASM polyfill for 256-bit

### Phase 4: Unified Portable API
- [ ] Create `src/simd/portable.rs`
- [ ] Runtime dispatch based on detected features
- [ ] Platform-agnostic benchmarks

## Open Questions

1. **FMA availability**: NEON always has FMA, SSE doesn't. How to handle `mul_add` in polyfill?
   - Option A: Polyfill mul_add as mul+add on SSE
   - Option B: Require FMA token for mul_add
   - Current: Option A (documented in `mul_add` as non-fused on SSE)

2. **Denormals**: Different platforms handle denormals differently. Test edge cases.

3. **NaN semantics**: IEEE 754 vs platform-specific NaN handling.

4. **Alignment**: WASM SIMD has different alignment requirements.
