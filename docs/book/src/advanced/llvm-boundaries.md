# LLVM Optimization Boundaries

Understanding when LLVM can and cannot optimize across function calls is crucial for peak SIMD performance.

## The Problem

`#[target_feature]` changes LLVM's target settings for a function. When caller and callee have different settings, LLVM cannot optimize across the boundary.

```rust
// Generic caller - baseline target settings
fn dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) {
    if let Some(t) = token.as_x64v3() {
        process_avx2(t, data);  // Different LLVM target!
    }
}

// AVX2 callee - AVX2 target settings
#[arcane]
fn process_avx2(token: X64V3Token, data: &[f32]) {
    // Can't inline back into dispatch()
}
```

## Why It Matters

SIMD performance depends heavily on:

1. **Inlining** — Avoids function call overhead
2. **Register allocation** — Keeps values in SIMD registers
3. **Instruction scheduling** — Reorders for pipeline efficiency

All of these break at target feature boundaries.

## Good: `#[rite]` Helpers Inside `#[arcane]`

```rust
#[arcane]
fn outer(token: X64V3Token, data: &[f32]) -> f32 {
    let a = step1(token, data);     // #[rite] → inlines
    let b = step2(token, data);     // #[rite] → inlines
    a + b
}

#[rite]
fn step1(token: X64V3Token, data: &[f32]) -> f32 {
    // Same target features as outer → LLVM inlines freely
}

#[rite]
fn step2(token: X64V3Token, data: &[f32]) -> f32 {
    // Same — one optimization region
}
```

## Good: Downcast (Higher → Lower)

```rust
#[arcane]
fn v4_main(token: X64V4Token, data: &[f32]) -> f32 {
    // Calling V3 function with V4 token
    // V4 is superset of V3, LLVM can still optimize
    v3_helper(token, data)
}

#[rite]
fn v3_helper(token: X64V3Token, data: &[f32]) -> f32 {
    // V4's features ⊃ V3's features → inlines properly
}
```

## Bad: Generic Boundary

```rust
// Generic function - no target features
fn generic<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    // This is compiled with baseline settings
    if let Some(t) = token.as_x64v3() {
        concrete(t, data)  // BOUNDARY - can't inline back
    } else {
        0.0
    }
}

#[arcane]
fn concrete(token: X64V3Token, data: &[f32]) -> f32 {
    // This has AVX2 settings
    // LLVM won't inline this into generic()
}
```

## Bad: Upcast Check in Hot Code

```rust
#[arcane]
fn process(token: X64V3Token, data: &[f32]) -> f32 {
    // WRONG: Checking for higher tier inside hot function
    for chunk in data.chunks(8) {
        if let Some(v4) = token.as_x64v4() {  // Wait, this always fails!
            // V3 token can't become V4
        }
    }
}
```

Even when the check makes sense, it's an optimization barrier.

## Pattern: Dispatch at Entry

```rust
// Public API - dispatch happens here
pub fn process(data: &[f32]) -> f32 {
    // Summon and dispatch ONCE
    #[cfg(feature = "avx512")]
    if let Some(token) = X64V4Token::summon() {
        return process_v4(token, data);
    }

    if let Some(token) = X64V3Token::summon() {
        return process_v3(token, data);
    }

    process_scalar(data)
}

// Each implementation is self-contained
#[arcane]
fn process_v4(token: X64V4Token, data: &[f32]) -> f32 {
    // Entry point — one boundary crossing
    let result = step1_v4(token, data);  // #[rite] inlines
    step2_v4(token, result)              // #[rite] inlines
}

#[rite]
fn step1_v4(token: X64V4Token, data: &[f32]) -> f32 { /* ... */ }

#[rite]
fn step2_v4(token: X64V4Token, result: f32) -> f32 { /* ... */ }
```

## Pattern: Trait with Concrete Impls

```rust
trait Processor {
    fn process(&self, data: &[f32]) -> f32;
}

struct V3Processor(X64V3Token);

impl Processor for V3Processor {
    fn process(&self, data: &[f32]) -> f32 {
        // Note: this can't use #[arcane] on trait method
        // Call through to arcane function instead
        process_v3_impl(self.0, data)
    }
}

#[arcane]
fn process_v3_impl(token: X64V3Token, data: &[f32]) -> f32 {
    // Full optimization here
}
```

## Measured Impact

The boundary costs 4x on simple vector adds and up to 6.2x on real workloads like DCT-8 (where LLVM loses FMA fusion and register reuse across the boundary). Benchmark data confirms that feature direction matters for cross-token nesting:

- **Downgrade (V4→V3, V3→V2):** free — caller's superset enables inlining
- **Upgrade (V2→V3, V3→V4):** 4x — callee needs features the caller lacks
- **Same level (V3→V3):** free when called from matching `#[target_feature]` context

All patterns: archmage = bare `#[target_feature]` (zero overhead from the abstraction).

See the [full benchmark data](../../../PERFORMANCE.md) for all patterns and numbers.

## Summary

| Pattern | Inlining | Recommendation |
|---------|----------|----------------|
| `#[rite]` with same token | Full | Default for hot paths |
| Downcast (V4→V3) via `#[rite]` | Full | Safe and fast — superset enables inlining |
| Upgrade (V2→V3, V3→V4) | Boundary (4x) | Dispatch at entry point, not in hot code |
| `#[arcane]` from non-SIMD code | Boundary (4-6x) | Entry point only — one crossing |
| Generic → concrete | Boundary | Entry point only |
