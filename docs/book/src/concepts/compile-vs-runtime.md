# Compile-Time vs Runtime

Understanding when things happen is key to writing correct SIMD code.

## Quick Reference

| What | When | Example |
|------|------|---------|
| `#[cfg(target_arch)]` | Compile | Include code for x86 only |
| `#[cfg(target_feature)]` | Compile | True only if compiled with that feature |
| `#[rite]` / `#[arcane]` | Compile | Tell LLVM to use SIMD instructions |
| `Token::summon()` | Runtime | Check if CPU supports the features |

## The Key Insight

**`#[rite]`** (or `#[arcane]`) tells LLVM: "Assume SIMD features are available in this function."

This is what makes intrinsics compile to single instructions. Without it, even if you call `_mm256_add_ps`, LLVM might not use AVX2 instructions.

```rust
#[rite]
fn example(_: Desktop64, a: __m256, b: __m256) -> __m256 {
    // LLVM knows AVX2 is available here
    // _mm256_add_ps compiles to a single vaddps instruction
    _mm256_add_ps(a, b)
}
```

**`summon()`** is the runtime check that proves the CPU actually has those features:

```rust
if let Some(token) = Desktop64::summon() {
    // Runtime: CPUID confirmed AVX2+FMA
    example(token, a, b);  // Safe to call
}
```

## Common Mistakes

### Mistake 1: Thinking `#[cfg(target_feature)]` Is Runtime

```rust
// WRONG: This is compile-time!
#[cfg(target_feature = "avx2")]
fn maybe_avx2() {
    // Only exists if compiled with -Ctarget-cpu=haswell
}

// RIGHT: Use tokens for runtime detection
fn maybe_avx2() {
    if let Some(token) = Desktop64::summon() {
        // Runtime check via CPUID
    }
}
```

### Mistake 2: Calling SIMD Functions Without `#[rite]`

```rust
use safe_unaligned_simd::x86_64 as safe_simd;

// WRONG: No target_feature, LLVM uses scalar code
fn process(data: &[f32; 8]) {
    let v = safe_simd::_mm256_loadu_ps(data);
    // May not even compile to vmovups!
}

// RIGHT: Tell LLVM to use AVX2
#[rite]
fn process(_: Desktop64, data: &[f32; 8]) {
    let v = safe_simd::_mm256_loadu_ps(data);
    // Compiles to vmovups
}
```

### Mistake 3: Summoning in Hot Loops

```rust
// WRONG: CPUID every iteration
for chunk in data.chunks(8) {
    if let Some(token) = Desktop64::summon() {
        process(token, chunk);
    }
}

// RIGHT: Summon once, loop inside
if let Some(token) = Desktop64::summon() {
    for chunk in data.chunks(8) {
        process(token, chunk);
    }
}
```

## When Detection Compiles Away

With `-Ctarget-cpu=native` or `-Ctarget-cpu=haswell`:

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

Now `Desktop64::summon()` becomes a no-op — LLVM knows at compile time that AVX2 is available, so the check compiles to an unconditional `Some(token)`.

Check this programmatically:

```rust
match Desktop64::guaranteed() {
    Some(true) => "Compile-time guaranteed (no runtime check)",
    Some(false) => "Wrong architecture",
    None => "Runtime check needed",
}
```

## Summon Performance

Even when runtime checks are needed, they're fast:

| Operation | Time |
|-----------|------|
| First `summon()` | ~2.6 ns |
| Subsequent calls (cached) | ~1.3 ns |
| With `-Ctarget-cpu=haswell` | 0 ns |

But the reason to hoist `summon()` outside loops isn't because it's slow — it's so LLVM can optimize the entire loop as one unit with consistent assumptions.

## LLVM Optimization Boundaries

Functions with the same `#[target_feature]` inline into each other. Functions with different features create optimization boundaries.

```rust
// GOOD: Both #[rite] with same token, inlines perfectly
#[rite]
fn outer(token: Desktop64, data: &[f32]) -> f32 {
    inner(token, data) * 2.0
}

#[rite]
fn inner(token: Desktop64, data: &[f32]) -> f32 {
    // ...
}
```

```rust
// SLOWER: Different tokens = optimization boundary
#[rite]
fn v4_kernel(token: X64V4Token, data: &[f32]) -> f32 {
    v3_helper(token, data)  // Works, but can't fully inline
}

#[rite]
fn v3_helper(token: X64V3Token, data: &[f32]) -> f32 {
    // Different target_feature set
}
```

For hot paths, use the same token type throughout the call chain.
