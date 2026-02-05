# Archmage

Archmage makes SIMD programming in Rust safe and zero-overhead. You prove CPU feature availability once with a **token**, then write safe SIMD code that compiles to raw instructions.

## The Pattern

Here's everything you need to know:

```rust
use archmage::{Desktop64, SimdToken, rite};
use magetypes::f32x8;

// PUBLIC API: Check CPU features once, dispatch to SIMD or scalar
pub fn sum_of_squares(data: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        sum_of_squares_simd(token, data)
    } else {
        data.iter().map(|x| x * x).sum()
    }
}

// SIMD IMPLEMENTATION: Loop goes inside, use #[rite] for all SIMD functions
#[rite]
fn sum_of_squares_simd(token: Desktop64, data: &[f32]) -> f32 {
    let mut total = 0.0;
    for chunk in data.chunks_exact(8) {
        let v = f32x8::from_slice(token, chunk);
        total += (v * v).reduce_add();
    }
    // Handle remainder
    for &x in data.chunks_exact(8).remainder() {
        total += x * x;
    }
    total
}
```

That's it. Three things to remember:

1. **`summon()` once** at your API boundary — returns `Some(token)` if CPU supports the features
2. **Put loops inside** the SIMD function — not outside calling in
3. **Use `#[rite]`** for all SIMD functions — it enables SIMD instructions with zero overhead

## Why This Works

**Tokens** are zero-sized proof types. `Desktop64` proves the CPU has AVX2+FMA (available on Intel Haswell 2013+, AMD Zen 1 2017+, basically any x86-64 from the last decade).

**`#[rite]`** tells the compiler "this function uses SIMD instructions." It adds `#[target_feature(enable = "avx2,fma,...")]` and `#[inline]`. Since Rust 1.85, SIMD intrinsics are safe inside these functions.

**magetypes** provides SIMD vector types (`f32x8`, `i32x4`, etc.) with natural operators. You get `+`, `-`, `*`, `/` that compile to single SIMD instructions.

## Zero Overhead

Archmage generates identical assembly to hand-written unsafe code:

```
Benchmark: 1000 iterations, 8-float operations
  Hand-written unsafe:    570 ns
  #[rite] functions:      572 ns  ← same
```

The abstraction is purely compile-time. At runtime, it's just your SIMD instructions.

## Cross-Platform by Default

Token types exist on all platforms. On unsupported platforms, `summon()` returns `None`:

```rust
// This compiles on ARM, WASM, everywhere
if let Some(token) = Desktop64::summon() {
    // Only runs on x86-64 with AVX2+FMA
    process_avx2(token, data);
} else {
    // Runs on ARM, WASM, older x86, etc.
    process_scalar(data);
}
```

No `#[cfg(target_arch)]` needed for basic dispatch. Write once, compile everywhere.

## Token Quick Reference

| Token | Features | Typical CPUs |
|-------|----------|--------------|
| `Desktop64` | AVX2 + FMA | Intel Haswell+, AMD Zen+ |
| `X64V4Token` | + AVX-512 | Intel Skylake-X+, AMD Zen 4+ |
| `Arm64` | NEON | All 64-bit ARM |
| `Simd128Token` | WASM SIMD128 | Modern browsers |

`Desktop64` is the sweet spot for x86-64 — wide availability, good performance.

## Next Steps

- **[Installation](./getting-started/installation.md)** — Add archmage to your project
- **[Your First SIMD Function](./getting-started/first-simd.md)** — Complete walkthrough
- **[When to Use #\[arcane\]](./concepts/arcane-vs-rite.md)** — Entry points vs internal functions
