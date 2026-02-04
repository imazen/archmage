# ARM NEON SIMD Type Generation

This document describes the design for generating ARM NEON SIMD types in archmage.

## Overview

ARM NEON (Advanced SIMD) provides 128-bit vector registers on AArch64. Unlike x86 which has multiple width levels (SSE→AVX→AVX-512), NEON is exclusively 128-bit. For 256-bit operations, we use polyfills (2×128-bit).

## NEON vs x86 Comparison

| Aspect | x86 SSE | ARM NEON |
|--------|---------|----------|
| Width | 128-bit | 128-bit |
| Registers | 16 (XMM0-15) | 32 (V0-31) |
| FMA | Separate (requires FMA flag) | Always available |
| Float types | `__m128`, `__m128d` | `float32x4_t`, `float64x2_t` |
| Int types | `__m128i` (all sizes) | Separate per size |
| Prefix | `_mm_` | `v` |
| Suffix | `_ps`, `_epi32` | `q_f32`, `q_s32` |

## NEON Intrinsic Naming Convention

```
v[operation][modifier]q_[type]

v         = vector prefix (always)
operation = add, sub, mul, div, min, max, etc.
modifier  = optional (a=accumulate, l=long, n=narrow, etc.)
q         = quadword (128-bit) - use q for all our types
type      = f32, f64, s8, u8, s16, u16, s32, u32, s64, u64
```

### Examples

| Operation | x86 SSE | ARM NEON |
|-----------|---------|----------|
| Add f32 | `_mm_add_ps(a, b)` | `vaddq_f32(a, b)` |
| Mul f32 | `_mm_mul_ps(a, b)` | `vmulq_f32(a, b)` |
| Min f32 | `_mm_min_ps(a, b)` | `vminq_f32(a, b)` |
| Add i32 | `_mm_add_epi32(a, b)` | `vaddq_s32(a, b)` |
| Min u8 | `_mm_min_epu8(a, b)` | `vminq_u8(a, b)` |
| FMA | `_mm_fmadd_ps(a, b, c)` | `vfmaq_f32(c, a, b)` |
| Load | `_mm_loadu_ps(ptr)` | `vld1q_f32(ptr)` |
| Store | `_mm_storeu_ps(ptr, v)` | `vst1q_f32(ptr, v)` |
| Splat | `_mm_set1_ps(x)` | `vdupq_n_f32(x)` |
| Zero | `_mm_setzero_ps()` | `vdupq_n_f32(0.0)` |

## NEON Type Mapping

### Intrinsic Types

| Element | Lanes | NEON Type |
|---------|-------|-----------|
| f32 | 4 | `float32x4_t` |
| f64 | 2 | `float64x2_t` |
| i8 | 16 | `int8x16_t` |
| u8 | 16 | `uint8x16_t` |
| i16 | 8 | `int16x8_t` |
| u16 | 8 | `uint16x8_t` |
| i32 | 4 | `int32x4_t` |
| u32 | 4 | `uint32x4_t` |
| i64 | 2 | `int64x2_t` |
| u64 | 2 | `uint64x2_t` |

### Token Requirements

| Width | Token | Notes |
|-------|-------|-------|
| 128-bit | `NeonToken` | Baseline (always available) |
| 256-bit | `NeonToken` + polyfill | Use 2×128-bit |

## Implementation Plan

### 1. Architecture Module (`xtask/src/simd_types/arch/arm.rs`)

```rust
pub struct Arm;

impl Arch for Arm {
    fn target_arch() -> &'static str { "aarch64" }

    fn intrinsic_type(elem: ElementType, width: SimdWidth) -> &'static str {
        // Only W128 is native
        match (elem, width) {
            (ElementType::F32, SimdWidth::W128) => "float32x4_t",
            (ElementType::F64, SimdWidth::W128) => "float64x2_t",
            (ElementType::I8, SimdWidth::W128) => "int8x16_t",
            (ElementType::U8, SimdWidth::W128) => "uint8x16_t",
            // ...
        }
    }

    fn prefix(_width: SimdWidth) -> &'static str { "v" }

    fn suffix(elem: ElementType) -> &'static str {
        match elem {
            ElementType::F32 => "q_f32",
            ElementType::F64 => "q_f64",
            ElementType::I8 => "q_s8",
            ElementType::U8 => "q_u8",
            // ...
        }
    }

    fn required_token(width: SimdWidth, _: bool) -> &'static str {
        match width {
            SimdWidth::W128 => "NeonToken",
            _ => panic!("NEON only supports 128-bit"),
        }
    }

    fn supports_width(width: SimdWidth) -> bool {
        width == SimdWidth::W128
    }
}
```

### 2. Operation Mapping

| Category | x86 Pattern | NEON Pattern |
|----------|-------------|--------------|
| Arithmetic | `_mm_add_ps` | `vaddq_f32` |
| Comparison | `_mm_cmpeq_ps` | `vceqq_f32` |
| Min/Max | `_mm_min_ps` | `vminq_f32` |
| Bitwise | `_mm_and_ps` | `vandq_f32` (cast needed) |
| Shuffle | Complex | `vzip`, `vuzp`, `vtrn`, `vtbl` |
| FMA | `_mm_fmadd_ps(a,b,c)` | `vfmaq_f32(c,a,b)` (note: arg order!) |
| Sqrt | `_mm_sqrt_ps` | `vsqrtq_f32` |
| Reciprocal | `_mm_rcp_ps` | `vrecpeq_f32` (estimate) |
| Floor/Ceil | `_mm_floor_ps` | `vrndmq_f32` / `vrndpq_f32` |
| Round | `_mm_round_ps` | `vrndnq_f32` |

### 3. Special Considerations

#### FMA Argument Order
x86 FMA: `a * b + c` → `_mm_fmadd_ps(a, b, c)`
NEON FMA: `a + b * c` → `vfmaq_f32(a, b, c)`

So our `mul_add(self, a, b)` meaning `self * a + b`:
- x86: `_mm_fmadd_ps(self, a, b)`
- NEON: `vfmaq_f32(b, self, a)`

#### Bitwise Operations on Floats
NEON requires casting float vectors to integer vectors for bitwise ops:
```rust
// NEON and for floats
fn and(self, other: Self) -> Self {
    let a = vreinterpretq_u32_f32(self.0);
    let b = vreinterpretq_u32_f32(other.0);
    Self(vreinterpretq_f32_u32(vandq_u32(a, b)))
}
```

#### Horizontal Operations
NEON lacks efficient horizontal ops. Use pairwise operations:
```rust
// reduce_add for f32x4
fn reduce_add(self) -> f32 {
    let sum = vpaddq_f32(self.0, self.0);  // [a+b, c+d, a+b, c+d]
    let sum = vpaddq_f32(sum, sum);         // [a+b+c+d, ...]
    vgetq_lane_f32(sum, 0)
}
```

### 4. Generated File Structure

```
src/simd/
├── mod.rs              # Re-exports, width namespaces
├── polyfill.rs         # Platform-agnostic polyfills
├── x86/                # x86-64 types
│   ├── mod.rs
│   ├── w128.rs
│   ├── w256.rs
│   └── w512.rs
└── arm/                # AArch64 types (NEW)
    ├── mod.rs
    └── w128.rs         # NEON 128-bit types
```

### 5. Cross-Platform Namespace

```rust
// src/simd/mod.rs additions

#[cfg(target_arch = "aarch64")]
pub mod neon {
    //! NEON width aliases (128-bit SIMD)

    pub use super::arm::w128::*;
    pub type Token = crate::NeonToken;

    pub const LANES_F32: usize = 4;
    pub const LANES_F64: usize = 2;
    pub const LANES_32: usize = 4;
    pub const LANES_16: usize = 8;
    pub const LANES_8: usize = 16;
}
```

## Testing Strategy

### 1. Cross-Compilation Verification

```bash
# Build for AArch64
cargo build --target aarch64-unknown-linux-gnu

# Run tests on ARM hardware or QEMU
cargo test --target aarch64-unknown-linux-gnu
```

### 2. Value Verification Tests

Compare NEON results to scalar reference implementations:

```rust
#[test]
#[cfg(target_arch = "aarch64")]
fn verify_neon_add() {
    let token = NeonToken::summon().unwrap();
    let a = f32x4::load(token, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::load(token, &[4.0, 3.0, 2.0, 1.0]);
    let result = (a + b).to_array();
    assert_eq!(result, [5.0, 5.0, 5.0, 5.0]);
}
```

### 3. Cross-Platform Parity Tests

Same algorithm, different platforms, same results:

```rust
// Run on both x86 and ARM, compare outputs
fn test_algorithm() -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        let token = Sse41Token::summon().unwrap();
        // ... x86 implementation
    }
    #[cfg(target_arch = "aarch64")]
    {
        let token = NeonToken::summon().unwrap();
        // ... NEON implementation
    }
}
```

## Polyfill for 256-bit

Since NEON only has 128-bit registers, f32x8 uses polyfill:

```rust
// src/simd/polyfill/neon.rs
pub struct f32x8 {
    lo: neon::f32x4,
    hi: neon::f32x4,
}

impl f32x8 {
    pub fn load(token: NeonToken, data: &[f32; 8]) -> Self {
        Self {
            lo: neon::f32x4::load(token, data[0..4].try_into().unwrap()),
            hi: neon::f32x4::load(token, data[4..8].try_into().unwrap()),
        }
    }
    // ... same pattern as SSE polyfill
}
```

## Implementation Order

1. ✅ Design document (this file)
2. [ ] Create `xtask/src/simd_types/arch/arm.rs`
3. [ ] Update `xtask/src/simd_types/arch/mod.rs`
4. [ ] Update structure.rs for ARM code generation
5. [ ] Update ops.rs for ARM intrinsics
6. [ ] Generate `src/simd/arm/w128.rs`
7. [ ] Add NEON namespace to `src/simd/mod.rs`
8. [ ] Add NEON polyfill for 256-bit
9. [ ] Cross-compile and test
