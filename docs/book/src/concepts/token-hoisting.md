# Token Hoisting

**This is the most important performance rule in archmage.**

Summon tokens **once** at your API boundary. Pass them through the call chain. Never summon in hot loops.

## The Problem: 42% Performance Regression

```rust
// WRONG: Summoning in inner function
fn distance(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    if let Some(token) = X64V3Token::summon() {  // CPUID every call!
        distance_simd(token, a, b)
    } else {
        distance_scalar(a, b)
    }
}

fn find_closest(points: &[[f32; 8]], query: &[f32; 8]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (i, point) in points.iter().enumerate() {
        let d = distance(point, query);  // summon() called N times!
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}
```

This is **42% slower** than hoisting the token. CPUID is not free.

## The Solution: Hoist to API Boundary

```rust
use archmage::{X64V3Token, SimdToken, arcane};
use magetypes::simd::f32x8;

// RIGHT: Summon once, pass through
fn find_closest(points: &[[f32; 8]], query: &[f32; 8]) -> usize {
    // Summon ONCE at entry
    if let Some(token) = X64V3Token::summon() {
        find_closest_simd(token, points, query)
    } else {
        find_closest_scalar(points, query)
    }
}

#[arcane]
fn find_closest_simd(token: X64V3Token, points: &[[f32; 8]], query: &[f32; 8]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (i, point) in points.iter().enumerate() {
        let d = distance_simd(token, point, query);  // Token passed, no summon!
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

#[arcane]
fn distance_simd(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Just use the token - no detection here
    let va = f32x8::from_array(token, *a);
    let vb = f32x8::from_array(token, *b);
    let diff = va - vb;
    (diff * diff).reduce_add().sqrt()
}
```

## The Rule

```
┌─────────────────────────────────────────────────────────┐
│  Public API boundary                                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │  if let Some(token) = Token::summon() {             ││
│  │      // ONLY place summon() is called               ││
│  │      internal_impl(token, ...);                     ││
│  │  }                                                  ││
│  └─────────────────────────────────────────────────────┘│
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────────┐│
│  │  #[arcane]                                          ││
│  │  fn internal_impl(token: Token, ...) {              ││
│  │      helper(token, ...);  // Pass token through     ││
│  │  }                                                  ││
│  └─────────────────────────────────────────────────────┘│
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────────┐│
│  │  #[arcane]                                          ││
│  │  fn helper(token: Token, ...) {                     ││
│  │      // Use token, never summon                     ││
│  │  }                                                  ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Why Tokens Are Zero-Cost to Pass

```rust
// Tokens are zero-sized
assert_eq!(std::mem::size_of::<X64V3Token>(), 0);

// Passing them costs nothing at runtime
fn f(token: X64V3Token) { }  // No actual parameter in compiled code
```

The token exists only at compile time to prove you did the check. At runtime, it's completely erased.

## When `-Ctarget-cpu=native` Helps

With compile-time feature guarantees, `summon()` becomes a no-op:

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

Now `X64V3Token::summon()` compiles to:

```rust
// Effectively becomes:
fn summon() -> Option<X64V3Token> {
    Some(X64V3Token)  // No CPUID, unconditional
}
```

But even then, **always hoist**. It's good practice, and your code works correctly when compiled without target-cpu.

## Summary

| Pattern | Performance | Correctness |
|---------|-------------|-------------|
| `summon()` in hot loop | 42% slower | Works |
| `summon()` at API boundary | Optimal | Works |
| `summon()` with `-Ctarget-cpu` | Optimal | Works |
