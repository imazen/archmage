# Compile-Time vs Runtime

Understanding when feature detection happens is crucial for writing correct and fast SIMD code.

## The Mechanisms

| Mechanism | When | What It Does |
|-----------|------|--------------|
| `#[cfg(target_arch = "...")]` | Compile | Include/exclude code from binary |
| `#[cfg(target_feature = "...")]` | Compile | True only if feature is in target spec |
| `#[cfg(feature = "...")]` | Compile | Cargo feature flag |
| `-Ctarget-cpu=native` | Compile | LLVM assumes current CPU's features |
| `Token::summon()` | Runtime | CPUID instruction, returns `Option<Token>` |

## Compile-Time: `#[cfg]` Attributes

These control what code **exists in the binary**:

```rust
// Only compiled for x86-64
#[cfg(target_arch = "x86_64")]
fn x86_only() { }

// Only compiled if AVX2 is in the target spec
// (e.g., -Ctarget-cpu=haswell)
#[cfg(target_feature = "avx2")]
fn avx2_guaranteed() { }

// Only compiled if cargo feature enabled
#[cfg(feature = "avx512")]
fn with_avx512_feature() { }
```

**Key insight**: `#[cfg(target_feature = "avx2")]` is **false** by default on x86-64. The default target only guarantees SSE/SSE2.

## Runtime: Token Detection

`summon()` does actual CPU detection:

```rust
use archmage::{X64V3Token, SimdToken};

fn main() {
    // This is a RUNTIME check (CPUID instruction)
    if let Some(token) = X64V3Token::summon() {
        println!("AVX2+FMA available");
    } else {
        println!("Falling back to scalar");
    }
}
```

## When Detection Compiles Away

With `-Ctarget-cpu=native` or `-Ctarget-cpu=haswell`:

```rust
// When compiled with -Ctarget-cpu=haswell:
// - target_feature = "avx2" is TRUE
// - X64V3Token::guaranteed() returns Some(true)
// - summon() becomes a no-op, returns Some unconditionally
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

### Mistake 1: Assuming `#[cfg(target_feature)]` Is Runtime

```rust
// WRONG: This is compile-time, not runtime!
#[cfg(target_feature = "avx2")]
fn maybe_avx2() {
    // This function doesn't exist unless compiled with -Ctarget-cpu=haswell
}

// RIGHT: Use tokens for runtime detection
fn maybe_avx2() {
    if let Some(token) = X64V3Token::summon() {
        avx2_impl(token);
    }
}
```

### Mistake 2: Gating Types on Cargo Features

```rust
// WRONG: Type doesn't exist at runtime even if CPU supports it
#[cfg(feature = "avx512")]
pub type FastVec = f32x16;

// RIGHT: Types always exist, detection is runtime
pub type FastVec = f32x16;  // Always available

fn use_it() {
    if let Some(token) = X64V4Token::summon() {
        let v = FastVec::splat(token, 1.0);
    }
}
```

### Mistake 3: Mixing Up Cfg and Summon

```rust
// WRONG: Cfg is compile-time, but you're thinking runtime
#[cfg(target_arch = "x86_64")]
fn process(data: &[f32]) {
    // This compiles fine, but panics on old CPUs!
    let token = X64V3Token::summon().unwrap();  // DON'T unwrap blindly
}

// RIGHT: Handle the None case
#[cfg(target_arch = "x86_64")]
fn process(data: &[f32]) {
    if let Some(token) = X64V3Token::summon() {
        process_avx2(token, data);
    } else {
        process_scalar(data);
    }
}
```

## Summary

| Question | Answer |
|----------|--------|
| "Does this code exist in the binary?" | `#[cfg(...)]` — compile-time |
| "Can this CPU run AVX2?" | `Token::summon()` — runtime |
| "Is runtime check needed?" | `Token::guaranteed()` — tells you |
| "How do I skip the check?" | `-Ctarget-cpu=native` or similar |
