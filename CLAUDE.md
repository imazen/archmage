# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

## Quick Start

```bash
cargo test                    # Run tests
cargo test --all-features     # Test with all integrations
cargo clippy --all-features   # Lint
cargo run -p xtask -- generate # Regenerate safe_unaligned_simd wrappers
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
2. **`#[arcane]` macro** - Enable `#[target_feature]` via token proof
3. **Safe load/store** - Reference-based memory operations (optional)

## How `#[arcane]` Works

The macro generates an inner function with `#[target_feature]`:

```rust
// You write:
#[arcane]
fn my_kernel(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();  // Safe!
    // ...
}

// Macro generates:
fn my_kernel(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_setzero_ps();  // Safe inside #[target_feature]!
        // ...
    }
    // SAFETY: Token proves CPU support was verified via try_new()
    unsafe { inner(data) }
}
```

**Why is this safe?**
1. `inner()` has `#[target_feature]`, so intrinsics are safe inside
2. Calling `inner()` is unsafe, but valid because:
   - The function requires a token parameter
   - Tokens can only be created via `try_new()` which checks CPUID
   - If you have a token, the CPU supports the features

## Generic Token Bounds

Use trait bounds to accept any compatible token:

```rust
#[arcane]
fn process(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    // Works with Avx2Token, X64V3Token, X64V4Token, etc.
}

#[arcane]
fn fma_kernel<T: HasAvx2 + HasFma>(token: T, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Requires both AVX2 and FMA
}
```

**Recommended starting point:** `X64V3Token` (AVX2 + FMA + BMI2, covers Haswell 2013+ and Zen 1+)

## Directory Structure

```
src/
├── lib.rs              # Main exports
├── tokens/
│   ├── mod.rs          # SimdToken trait, marker traits (HasAvx2, etc.)
│   ├── x86.rs          # x86 token types
│   ├── arm.rs          # ARM token types
│   └── wasm.rs         # WASM token types
├── composite/          # Higher-level operations (__composite feature)
│   ├── mod.rs
│   ├── simd_ops.rs     # SIMD operation traits
│   ├── scalar_ops.rs   # Scalar fallback traits
│   ├── x86_impls.rs    # Token trait implementations
│   ├── transpose.rs    # 8x8 matrix transpose
│   ├── dot_product.rs  # Dot product
│   └── horizontal.rs   # Horizontal reductions
├── integrate/
│   └── wide_ops.rs     # wide crate integration (__wide feature)
├── mem.rs              # Re-exports generated wrappers
└── generated/          # AUTO-GENERATED (safe_unaligned_simd feature)
    ├── x86/            # 235 x86_64 functions
    └── aarch64/        # 240 NEON functions
xtask/
└── src/main.rs         # Wrapper generator
```

## Token Hierarchy

**Profile Tokens (recommended):**
- `X64V3Token` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+) **← Start here**
- `X64V4Token` - + AVX-512 (Skylake-X 2017+, Zen 4+)
- `X64V2Token` - SSE4.2 + POPCNT (Nehalem 2008+)

**Feature Tokens:**
- `Sse2Token` → `Sse41Token` → `Sse42Token` → `AvxToken` → `Avx2Token`
- `FmaToken` (independent), `Avx2FmaToken` (combined)
- `Avx512fToken`, `Avx512bwToken`, `Avx512Vbmi2Token` + VL variants

**ARM:**
- `NeonToken` (baseline), `SveToken`, `Sve2Token`

## Marker Traits

Enable generic bounds:

```rust
fn requires_avx2(token: impl HasAvx2) { ... }
fn requires_fma(token: impl HasFma) { ... }
fn requires_both<T: HasAvx2 + HasFma>(token: T) { ... }
```

**Width traits:** `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`
**Feature traits:** `HasSse`, `HasSse2`, `HasAvx`, `HasAvx2`, `HasFma`, `HasAvx512f`, etc.
**ARM traits:** `HasNeon`, `HasSve`, `HasSve2`

## Safe Memory Operations

With `safe_unaligned_simd` feature, the `mem` module provides reference-based load/store:

```rust
use archmage::mem::avx;

if let Some(token) = X64V3Token::try_new() {
    let v = avx::_mm256_loadu_ps(token, &data);  // Safe! Reference, not pointer
    avx::_mm256_storeu_ps(token, &mut out, v);
}
```

## Limitations

**Self receivers not supported:**

```rust
// Won't work - inner functions can't have self
#[arcane]
fn process(&self, token: impl HasAvx2) { ... }

// Workaround: use free function
#[arcane]
fn process_impl(state: &MyState, token: impl HasAvx2) { ... }
```

## Generated Wrappers

The `mem` module wraps `safe_unaligned_simd` with token requirements:

```bash
cargo run -p xtask -- generate  # Regenerate after safe_unaligned_simd updates
```

The generator:
1. Parses safe_unaligned_simd source from cargo cache
2. Extracts function signatures and `#[target_feature]` attributes
3. Generates wrappers with `impl HasXxx` bounds
4. Groups by feature set (sse, sse2, avx, neon, etc.)

## License

MIT OR Apache-2.0
