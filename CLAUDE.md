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

The token IS passed through to the inner function, so you can call other token-taking functions:

```rust
#[simd_fn]
fn outer(token: Avx2Token, data: &mut [f32; 64]) {
    // Can call other token-taking functions
    transpose_8x8(token, data);
    let v = load_f32x8(token, (&data[0..8]).try_into().unwrap());
}
```

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
│   ├── x86.rs          # Sse2, Avx2, Fma, X64V2/V3/V4, AVX-512 tokens
│   ├── arm.rs          # Neon, Sve, Sve2 tokens
│   └── wasm.rs         # Simd128Token
├── ops/
│   └── x86/mod.rs      # Safe load/store only (reference-based)
├── integrate/
│   ├── safe_simd.rs    # safe_unaligned_simd integration
│   └── wide_ops.rs     # wide crate integration
├── generated/          # AUTO-GENERATED - token-gated safe_unaligned_simd wrappers
│   ├── mod.rs
│   ├── VERSION         # Tracks source version
│   └── x86/
│       ├── sse.rs      # SSE functions (6)
│       ├── sse2.rs     # SSE2 functions (20)
│       ├── avx.rs      # AVX functions (17)
│       ├── avx512f.rs  # AVX-512F functions (49)
│       ├── avx512f_vl.rs   # AVX-512F+VL functions (86)
│       ├── avx512bw.rs     # AVX-512BW functions (13)
│       ├── avx512bw_vl.rs  # AVX-512BW+VL functions (26)
│       ├── avx512vbmi2.rs  # AVX-512VBMI2 functions (6)
│       └── avx512vbmi2_vl.rs # AVX-512VBMI2+VL functions (12)
├── composite/
│   ├── transpose.rs    # 8x8 transpose
│   ├── dot_product.rs  # Dot product with FMA
│   └── horizontal.rs   # Horizontal sum/max/min
xtask/
└── src/main.rs         # Generator for safe_unaligned_simd wrappers
```

## Generated Wrappers (safe-simd feature)

The `src/generated/` directory contains **auto-generated** token-gated wrappers for all
`safe_unaligned_simd` functions. These wrappers make the functions truly safe by requiring
a capability token as the first parameter.

**To regenerate after updating safe_unaligned_simd:**
```bash
cargo run -p xtask -- generate
```

**CI automatically checks for updates** weekly and creates PRs when new versions are available.

The generator in `xtask/src/main.rs`:
1. Parses safe_unaligned_simd source from cargo cache
2. Extracts function signatures and `#[target_feature]` attributes
3. Generates wrapper functions that take a token + call the original
4. Groups by feature set (sse, sse2, avx, avx512f, etc.)

## Token Hierarchy

### x86_64

**Feature Tokens:**
- `SseToken` - SSE (rarely needed, SSE2 is baseline)
- `Sse2Token` - SSE2 (baseline on x86_64)
- `Sse41Token` → `Sse42Token` → `AvxToken` → `Avx2Token` → `Avx512fToken`
- `FmaToken` (independent)
- `Avx2FmaToken` (combined)

**AVX-512 Tokens (for safe-simd feature):**
- `Avx512fToken` - AVX-512 Foundation (512-bit only)
- `Avx512fVlToken` - AVX-512F + VL (128/256/512-bit operations)
- `Avx512bwToken` - AVX-512 Byte/Word
- `Avx512bwVlToken` - AVX-512BW + VL
- `Avx512Vbmi2Token` - AVX-512 VBMI2 (Ice Lake+, Zen 4+)
- `Avx512Vbmi2VlToken` - AVX-512 VBMI2 + VL

**Profile Tokens (x86-64 psABI levels):**
- `X64V2Token` - SSE4.2 + POPCNT (Nehalem 2008+)
- `X64V3Token` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
- `X64V4Token` - AVX-512F/BW/CD/DQ/VL (Xeon 2017+, Zen 4+)

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

The token IS passed through to the inner function, enabling composability:
- You can call other token-taking functions from inside `#[simd_fn]`
- The safe load/store functions require the token
- Composite operations can call other composites

### Why Reference-Based Load/Store?

Memory operations are still unsafe with raw pointers. By using `&[f32; 8]` instead of `*const f32`, we get:
- Memory validity guaranteed by Rust's borrow checker
- Correct size guaranteed by the array type
- Token proves feature availability
- Zero overhead - compiles to same assembly

## Capability Marker Traits

The following marker traits enable generic code to constrain which tokens are accepted:

- `Has128BitSimd` - SSE2, NEON, WASM SIMD128
- `Has256BitSimd` - AVX, AVX2, X64V3 (implies Has128BitSimd)
- `Has512BitSimd` - AVX-512, X64V4 (implies Has256BitSimd)
- `HasFma` - FMA, AVX2+FMA, X64V3, X64V4, NEON
- `HasScalableVectors` - SVE, SVE2

Example:
```rust
fn requires_fma<T: HasFma>(token: T) { ... }
fn requires_256<T: Has256BitSimd>(token: T) { ... }
```

## Operation Traits (Generic API)

Operation traits provide a generic interface with specialized implementations per token:

- `Transpose8x8` - 8x8 matrix transpose (critical for DCT)
- `DotProduct` - dot product, norm_squared, norm
- `HorizontalOps` - sum, max, min reductions

Usage:
```rust
fn process<T: Transpose8x8 + DotProduct>(token: T, block: &mut [f32; 64], data: &[f32]) {
    token.transpose_8x8(block);        // Uses optimized impl for T
    let dot = token.dot_product_f32(data, data);
}

// Works with any implementing token
if let Some(token) = X64V3Token::try_new() {
    process(token, &mut block, &data);
}
```

Each token provides its own optimized implementation - the compiler selects the right one at compile time.

## Future Work

- [x] Make `#[simd_fn]` pass token through for composability
- [x] Capability marker traits for generic bounds
- [x] aarch64 `mem` wrappers via macro-based generation (160 NEON functions)
- [ ] NEON/SVE composite operations for aarch64
- [ ] DCT, color conversion, and other JPEG primitives (see halide-kernels)

## License

MIT OR Apache-2.0
