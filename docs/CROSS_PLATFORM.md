# Cross-Platform SIMD with Archmage

This guide explains how to write portable SIMD code with archmage that works across
x86_64, aarch64, and wasm32.

## Quick Comparison: archmage vs wide + multiversed

| Feature | wide | archmage |
|---------|------|----------|
| Detection | Compile-time (`#[cfg(target_feature)]`) | Runtime (tokens) |
| Binary compatibility | Needs recompile per CPU | Single binary, all CPUs |
| Intrinsic inlining | Always (compile-time) | Inside `#[target_feature]` fns |
| Performance | Baseline | ~1.3x slower without `#[target_feature]` context |
| Safety | Unsafe intrinsic calls | Safe via token proof |

**When to use archmage over wide:**
- You need a single binary that runs on multiple CPU generations
- You want runtime feature detection (like multiversed provides)
- You prefer safe APIs with compile-time guarantees

## Pattern 1: Simple Runtime Dispatch

```rust
pub fn process(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return process_avx2(token, data);
        }
        if let Some(token) = archmage::Sse41Token::summon() {
            return process_sse(token, data);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return process_neon(token, data);
        }
    }

    // Scalar fallback
    process_scalar(data)
}
```

## Pattern 2: Using `#[arcane]` Macro (Recommended)

The `#[arcane]` macro generates `#[target_feature]` wrappers automatically, ensuring
intrinsics inline into proper SIMD instructions:

```rust
use archmage::{arcane, X64V3Token, SimdToken};
use archmage::simd::f32x8;

#[arcane]
fn process_simd(token: X64V3Token, data: &[f32]) -> f32 {
    let mut acc = f32x8::zero(token);

    for chunk in data.chunks_exact(8) {
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let v = f32x8::load(token, arr);
        acc = acc + v;  // This + compiles to a single vaddps instruction!
    }

    acc.reduce_add()
}

// Usage - token proves CPU support at call site
if let Some(token) = X64V3Token::summon() {
    let result = process_simd(token, &data);
}
```

**How `#[arcane]` works:**
1. Reads the token type to determine required target features (e.g., `X64V3Token` → `avx2,fma`)
2. Generates an inner function with `#[target_feature(enable = "...")]`
3. Calls that inner function safely (token proves CPU support)

**Why this matters:**
- Without `#[target_feature]`, operators become function calls (~1.3x slower)
- With `#[arcane]`, operators inline into single SIMD instructions
- Performance matches or exceeds the `wide` crate

## Pattern 4: Multi-Width Code with `#[multiwidth]`

For code that should work across SSE, AVX2, and AVX-512:

```rust
#[multiwidth]
mod kernels {
    use archmage::simd::*;

    pub fn sum(token: Token, data: &[f32]) -> f32 {
        let mut acc = f32xN::zero(token);  // Width-agnostic type!
        let chunks = data.chunks_exact(LANES_F32);

        for chunk in chunks {
            let arr: &[f32; LANES_F32] = chunk.try_into().unwrap();
            let v = f32xN::load(token, arr);
            acc = acc + v;
        }

        acc.reduce_add()
    }
}

// The macro generates:
// - kernels::sse::sum(Sse41Token, &[f32])     - 4-wide
// - kernels::avx2::sum(X64V3Token, &[f32])  - 8-wide
// - kernels::avx512::sum(X64V4Token, &[f32])  - 16-wide (with feature)
// - kernels::sum(&[f32])                       - auto-dispatch
```

## Platform-Specific Types

### x86_64

| Token | Width | SIMD Types |
|-------|-------|------------|
| `Sse41Token` | 128-bit | `f32x4`, `f64x2`, `i32x4`, ... |
| `X64V3Token` | 256-bit | `f32x8`, `f64x4`, `i32x8`, ... |
| `X64V4Token` | 512-bit | `f32x16`, `f64x8`, `i32x16`, ... |

### aarch64

| Token | Width | SIMD Types |
|-------|-------|------------|
| `NeonToken` | 128-bit | `f32x4`, `f64x2`, `i32x4`, ... |

### wasm32

WebAssembly SIMD support is planned. Currently use scalar fallbacks.

## Complete Example

See `examples/cross_platform.rs` for a complete working example that:
- Detects CPU features at runtime
- Dispatches to the best SIMD implementation
- Falls back to scalar code when needed
- Works on x86_64, aarch64, and other platforms

Run it with:
```bash
cargo run --example cross_platform --release
```

## Performance Tips

1. **Use `#[target_feature]` or `#[arcane]`** for hot paths - this enables proper inlining
2. **Minimize token passing** - get the token once, use it for entire batch operations
3. **Process data in chunks** - match chunk size to SIMD width (8 for AVX2, 4 for NEON)
4. **Use FMA when available** - `.mul_add()` is faster than separate `*` and `+`
5. **Reduce at the end** - keep data in SIMD registers, reduce to scalar only at the end

## Comparison with `-C target-cpu=native`

| Scenario | Approach | Performance |
|----------|----------|-------------|
| Dev machine only | `-C target-cpu=native` | Best (all ops inline) |
| Distributed binary | archmage with `#[arcane]` | ~1.3x slower than native |
| Library code | archmage tokens | Safe, portable |

For maximum performance when you control the build environment, use:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

For distributable binaries, use archmage's runtime dispatch with `#[arcane]` functions.
