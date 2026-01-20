# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

## Quick Start

```bash
cargo test                    # Run tests
cargo test --all-features     # Test with all integrations
cargo clippy --all-features   # Lint
```

## Core Insight: Rust 1.85+ Changed Everything

As of Rust 1.85, **value-based intrinsics are safe inside `#[target_feature]` functions**:

```rust
#[target_feature(enable = "avx2")]
unsafe fn example() {
    let a = _mm256_setzero_ps();           // SAFE!
    let b = _mm256_add_ps(a, a);           // SAFE!
    let c = _mm256_fmadd_ps(a, a, a);      // SAFE!

    // Only memory ops remain unsafe (raw pointers)
    let v = unsafe { _mm256_loadu_ps(ptr) };  // Still needs unsafe
}
```

This means we **don't need to wrap** arithmetic, shuffle, compare, bitwise, or other value-based intrinsics. Only:
1. **Tokens** - Prove CPU features are available
2. **Safe load/store** - Reference-based memory operations
3. **`#[simd_fn]` macro** - Enable `#[target_feature]` via token proof
4. **Composite operations** - Higher-level algorithms using `#[simd_fn]`

## What archmage Provides

### 1. Capability Tokens

Zero-sized proof types for runtime dispatch:

```rust
if let Some(token) = Avx2Token::try_new() {
    process_avx2(token, data);  // Safe to call
}
```

### 2. The `#[simd_fn]` Macro (Core Feature)

Makes raw intrinsics safe via token proof:

```rust
#[simd_fn]
fn kernel(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    // Memory ops still need unsafe (raw pointers)
    let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

    // ALL value-based intrinsics are SAFE here!
    let doubled = _mm256_add_ps(v, v);

    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
    out
}
```

The macro expands to an inner `#[target_feature]` function, making intrinsics safe.

**Important limitation**: The token is NOT passed to the inner function. This means you cannot call other token-taking functions from inside `#[simd_fn]`. Use raw intrinsics directly.

### 3. Safe Load/Store

Reference-based memory operations (in `ops::x86`):

```rust
let v = load_f32x8(token, &data);   // Safe! Uses reference
store_f32x8(token, &mut out, v);    // Safe! Uses reference
```

### 4. Composite Operations

Higher-level SIMD algorithms using `#[simd_fn]`:

- `transpose_8x8` - 8x8 matrix transpose (critical for DCT)
- `dot_product_f32` - Dot product with FMA
- `hsum_f32x8`, `hmax_f32x8`, `hmin_f32x8` - Horizontal reductions
- `sum_f32_slice`, `max_f32_slice`, `min_f32_slice` - Slice operations

## Directory Structure

```
src/
├── lib.rs              # Main exports, #[simd_fn] macro re-export
├── tokens/
│   ├── mod.rs          # SimdToken trait
│   ├── x86.rs          # Sse2, Avx2, Fma, X64V2/V3/V4 tokens
│   ├── arm.rs          # Neon, Sve, Sve2 tokens
│   └── wasm.rs         # Simd128Token
├── ops/
│   └── x86/mod.rs      # Safe load/store only (reference-based)
├── integrate/
│   ├── safe_simd.rs    # safe_unaligned_simd integration
│   └── wide_ops.rs     # wide crate integration
└── composite/
    ├── transpose.rs    # 8x8 transpose
    ├── dot_product.rs  # Dot product with FMA
    └── horizontal.rs   # Horizontal sum/max/min
```

## Token Hierarchy

### x86_64

**Feature Tokens:**
- `Sse2Token` - SSE2 (baseline)
- `Sse41Token` → `Sse42Token` → `AvxToken` → `Avx2Token` → `Avx512fToken`
- `FmaToken` (independent)
- `Avx2FmaToken` (combined)

**Profile Tokens (x86-64 psABI levels):**
- `X64V2Token` - SSE4.2 + POPCNT (Nehalem 2008+)
- `X64V3Token` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
- `X64V4Token` - AVX-512 (Xeon 2017+, Zen 4+)

### AArch64

- `NeonToken` - NEON (baseline, always available)
- `SveToken` - SVE (Graviton 3, A64FX)
- `Sve2Token` - SVE2 (ARMv9, Graviton 4)

See `~/work/multiversed` for feature set details used by the multiversion integration.

## Key Design Decisions

### Why Not Wrap Value-Based Intrinsics?

Rust 1.85+ made them safe inside `#[target_feature]`. Wrapping them adds no safety value and creates API bloat. The `#[simd_fn]` macro enables `#[target_feature]` based on the token parameter.

### Why `#[simd_fn]` Creates an Inner Function?

The outer function takes the token (proving features are available), then calls an inner `#[target_feature]` function. This makes the call to the inner function safe because we've proven the features exist.

The token is NOT passed to the inner function because:
1. The inner function already has `#[target_feature]` - it doesn't need runtime proof
2. Passing it would complicate the macro and add parameter overhead
3. Inside `#[simd_fn]`, you should use raw intrinsics directly

### Why Reference-Based Load/Store?

Memory operations are still unsafe with raw pointers. By using `&[f32; 8]` instead of `*const f32`, we get:
- Memory validity guaranteed by Rust's borrow checker
- Correct size guaranteed by the array type
- Token proves feature availability
- Zero overhead - compiles to same assembly

## Future Work

- [ ] Make `#[simd_fn]` optionally pass token through for composability
- [ ] Generic `SimdArch` trait for architecture-independent composites
- [ ] NEON/SVE composite operations for aarch64
- [ ] DCT, color conversion, and other JPEG primitives

## License

MIT OR Apache-2.0
