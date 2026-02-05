# Compile-Time vs Runtime

Understanding when feature detection happens—and how LLVM optimizes across feature boundaries—is crucial for writing correct and fast SIMD code.

## The Mechanisms

| Mechanism | When | What It Does |
|-----------|------|--------------|
| `#[cfg(target_arch = "...")]` | Compile | Include/exclude code from binary |
| `#[cfg(target_feature = "...")]` | Compile | True only if feature is in target spec |
| `#[cfg(feature = "...")]` | Compile | Cargo feature flag |
| `#[target_feature(enable = "...")]` | Compile | Tell LLVM to use these instructions in this function |
| `-Ctarget-cpu=native` | Compile | LLVM assumes current CPU's features globally |
| `Token::summon()` | Runtime | CPUID instruction, returns `Option<Token>` |

## The Key Insight: `#[target_feature(enable)]`

This is the mechanism that makes SIMD work. It tells LLVM: "Inside this function, assume these CPU features are available."

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn process_avx2(data: &[f32; 8]) -> f32 {
    // LLVM generates AVX2 instructions here
    // _mm256_* intrinsics compile to single instructions
    let v = _mm256_loadu_ps(data.as_ptr());
    // ...
}
```

**Why `unsafe`?** The function uses AVX2 instructions, but LLVM doesn't verify the caller checked for AVX2. If you call this on a CPU without AVX2, you get an illegal instruction fault. The `unsafe` is the contract: "caller must ensure CPU support."

**This is what `#[arcane]` does for you:**

```rust
// You write:
#[arcane]
fn process(token: Desktop64, data: &[f32; 8]) -> f32 { /* ... */ }

// Macro generates:
fn process(token: Desktop64, data: &[f32; 8]) -> f32 {
    #[target_feature(enable = "avx2,fma,bmi1,bmi2")]
    #[inline]
    unsafe fn __inner(token: Desktop64, data: &[f32; 8]) -> f32 { /* ... */ }

    // SAFETY: Token existence proves summon() succeeded
    unsafe { __inner(token, data) }
}
```

The token proves the runtime check happened. The inner function gets LLVM's optimizations.

## LLVM Optimization and Feature Boundaries

**Archmage is never slower than equivalent unsafe code.** When you use the right patterns (`#[rite]` helpers called from `#[arcane]`), the generated assembly is identical to hand-written `#[target_feature]` + `unsafe` code.

Here's why: **LLVM's optimization passes respect `#[target_feature]` boundaries.**

### Same Features = Full Optimization

When caller and callee have the same target features, LLVM can:
- Inline fully
- Propagate constants
- Eliminate redundant loads/stores
- Combine operations across the call boundary

```rust
#[arcane]
fn outer(token: Desktop64, data: &[f32; 8]) -> f32 {
    inner(token, data) * 2.0  // Inlines perfectly!
}

#[arcane]
fn inner(token: Desktop64, data: &[f32; 8]) -> f32 {
    let v = f32x8::from_array(token, *data);
    v.reduce_add()
}
```

Both functions have `#[target_feature(enable = "avx2,fma,...")]`. LLVM sees one optimization region.

### Different Features = Optimization Boundary

When features differ, LLVM must be conservative:

```rust
#[arcane]
fn v4_caller(token: X64V4Token, data: &[f32; 8]) -> f32 {
    // token: X64V4Token → avx512f,avx512bw,...
    v3_helper(token, data)  // Different features!
}

#[arcane]
fn v3_helper(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // token: X64V3Token → avx2,fma,...
    // Different target_feature set = optimization boundary
}
```

This still works—V4 is a superset of V3—but LLVM can't fully inline across the boundary because the `#[target_feature]` annotations differ.

### Generic Bounds = Optimization Boundary

Generics create the same problem:

```rust
#[arcane]
fn generic_impl<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    // LLVM doesn't know what T's features are at compile time
    // Must generate conservative code that works for any HasX64V2
}
```

The compiler generates one version of this function for the trait bound, not specialized versions for each concrete token. This prevents inlining and vectorization across the boundary.

**Rule: Use concrete tokens for hot paths.**

## Downcasting vs Upcasting

### Downcasting: Free

Higher tokens can be used where lower tokens are expected:

```rust
#[arcane]
fn v4_kernel(token: X64V4Token, data: &[f32; 8]) -> f32 {
    // V4 → V3 is free: just passing token, same LLVM features (superset)
    v3_sum(token, data)  // Desktop64 accepts X64V4Token
}

#[arcane]
fn v3_sum(token: Desktop64, data: &[f32; 8]) -> f32 {
    // ...
}
```

This works because `X64V4Token` has all the features of `Desktop64` plus more. LLVM's target features are a superset, so optimization flows freely.

### Upcasting: Safe but Creates Boundary

Going the other direction requires `IntoConcreteToken`:

```rust
fn dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    if let Some(v4) = token.as_x64v4() {
        v4_path(v4, data)  // Uses AVX-512 if available
    } else if let Some(v3) = token.as_x64v3() {
        v3_path(v3, data)  // Falls back to AVX2
    } else {
        scalar_path(data)
    }
}
```

This is **safe**—`as_x64v4()` returns `None` if the token doesn't support V4. But it creates an optimization boundary because the generic `T` becomes a concrete type at the branch point.

**Don't upcast in hot loops.** Upcast once at your dispatch point, then pass concrete tokens through.

## The `#[rite]` Optimization

`#[rite]` exists to eliminate wrapper overhead for inner helpers:

```rust
// #[arcane] creates a wrapper:
fn entry(token: Desktop64, data: &[f32; 8]) -> f32 {
    #[target_feature(enable = "avx2,fma,...")]
    unsafe fn __inner(...) { ... }
    unsafe { __inner(...) }
}

// #[rite] is the function directly:
#[target_feature(enable = "avx2,fma,...")]
#[inline]
fn helper(token: Desktop64, data: &[f32; 8]) -> f32 { ... }
```

Since Rust 1.85+, calling a `#[target_feature]` function from a matching context is safe. So `#[arcane]` can call `#[rite]` helpers without `unsafe`:

```rust
#[arcane]
fn entry(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let prod = mul_vectors(token, a, b);  // Calls #[rite], no unsafe!
    horizontal_sum(token, prod)
}

#[rite]
fn mul_vectors(_: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> __m256 { ... }

#[rite]
fn horizontal_sum(_: Desktop64, v: __m256) -> f32 { ... }
```

All three functions share the same `#[target_feature]` annotation. LLVM sees one optimization region.

## When Detection Compiles Away

With `-Ctarget-cpu=native` or `-Ctarget-cpu=haswell`:

```rust
// When compiled with -Ctarget-cpu=haswell:
// - #[cfg(target_feature = "avx2")] is TRUE
// - X64V3Token::guaranteed() returns Some(true)
// - summon() becomes a no-op
// - LLVM eliminates the branch entirely

if let Some(token) = X64V3Token::summon() {
    // This branch is the only code generated
}
```

Check programmatically:

```rust
match X64V3Token::guaranteed() {
    Some(true) => println!("Compile-time guaranteed"),
    Some(false) => println!("Wrong architecture"),
    None => println!("Runtime check needed"),
}
```

## Common Mistakes

### Mistake 1: Generic Bounds in Hot Paths

```rust
// SLOW: Generic creates optimization boundary
fn process<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    // Called millions of times, can't inline properly
}

// FAST: Concrete token, full optimization
fn process(token: Desktop64, data: &[f32]) -> f32 {
    // LLVM knows exact features, inlines everything
}
```

### Mistake 2: Assuming `#[cfg(target_feature)]` Is Runtime

```rust
// WRONG: This is compile-time, not runtime!
#[cfg(target_feature = "avx2")]
fn maybe_avx2() {
    // This function doesn't exist unless compiled with -Ctarget-cpu=haswell
}

// RIGHT: Use tokens for runtime detection
fn maybe_avx2() {
    if let Some(token) = Desktop64::summon() {
        avx2_impl(token);
    }
}
```

### Mistake 3: Dispatching in Hot Loops

```rust
// WRONG: crosses target-feature boundary every iteration
for chunk in data.chunks(8) {
    if let Some(token) = Desktop64::summon() {
        process(token, chunk);  // #[arcane] wrapper = boundary per call
    }
}

// BETTER: summon once, but still a boundary per iteration
if let Some(token) = Desktop64::summon() {
    for chunk in data.chunks(8) {
        process(token, chunk);  // Still calling #[arcane] each iteration
    }
}

// BEST: loop inside #[arcane], use #[rite] helpers
if let Some(token) = Desktop64::summon() {
    process_all(token, data);  // One boundary crossing
}
```

**The real cost isn't `summon()`** — it's the `#[target_feature]` boundary. `summon()` is a cached atomic load (~1.3 ns). But each `#[arcane]` call transitions between LLVM optimization regions: the caller has baseline features, the callee has AVX2+FMA. LLVM can't optimize across that boundary — no inlining, no constant propagation, no instruction combining.

Put the loop *inside* `#[arcane]` and use `#[rite]` for helpers. Everything inside shares the same target features, so LLVM treats it as one optimization region.

| Operation | Time |
|-----------|------|
| `Desktop64::summon()` (cached) | ~1.3 ns |
| First call (actual detection) | ~2.6 ns |
| With `-Ctarget-cpu=haswell` | 0 ns (compiles to `Some(token)`) |

### Mistake 4: Using `#[cfg(target_arch)]` Unnecessarily

```rust
// UNNECESSARY: Tokens exist everywhere
#[cfg(target_arch = "x86_64")]
{
    if let Some(token) = Desktop64::summon() { ... }
}

// CLEANER: Just use the token
if let Some(token) = Desktop64::summon() {
    // Returns None on non-x86, compiles everywhere
}
```

## Summary

| Question | Answer |
|----------|--------|
| "Does this code exist in the binary?" | `#[cfg(...)]` — compile-time |
| "Can this CPU run AVX2?" | `Token::summon()` — runtime |
| "What instructions can LLVM use here?" | `#[target_feature(enable)]` — per-function |
| "Is runtime check needed?" | `Token::guaranteed()` — tells you |
| "Will these functions inline together?" | Same target features + concrete types = yes |
| "Do generic bounds hurt performance?" | Yes, they create optimization boundaries |
| "Is downcasting (V4→V3) free?" | Yes, features are superset |
| "Is upcasting safe?" | Yes, but creates optimization boundary |
