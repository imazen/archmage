# Archmage Safety Model & Idiomatic Usage

Authoritative reference for understanding and teaching archmage. Read before writing docs or examples.

> **`#![forbid(unsafe_code)]` compatible.** Downstream crates can use `#![forbid(unsafe_code)]` when combining archmage tokens + `#[arcane]`/`#[rite]` macros + `safe_unaligned_simd` for memory operations. The `unsafe` lives inside archmage's generated code, not yours.

> **Descriptive aliases.** `#[token_target_features_boundary]` = `#[arcane]`, `#[token_target_features]` = `#[rite]`, `dispatch_variant!` = `incant!`. These help AI tools and newcomers infer what the macros do from the name alone.

> **Source of truth.** All token-to-feature mappings are defined in [`token-registry.toml`](../token-registry.toml). Everything else (generated code, macro registries, docs) is derived from it.

## Terminology

| Term | Definition |
|------|------------|
| **safe** | No `unsafe` keyword required |
| **unsafe** | Requires `unsafe` keyword (Rust can't verify the invariant) |
| **sound** | Cannot cause UB when used as documented |
| **unsound** | CAN cause UB even when used correctly |

These are orthogonal. Archmage's `#[arcane]` generates unsafe code that IS sound — the token proves features exist.

## The Core Safety Model

### 1. Tokens Are Proofs

```rust
#[derive(Clone, Copy)]  // Zero-sized!
pub struct X64V3Token;  // Contains no data

impl SimdToken for X64V3Token {
    fn summon() -> Option<Self> {
        if /* runtime CPUID check */ {
            Some(Self)  // Existence = proof features are available
        } else {
            None
        }
    }
}
```

You can't construct a token except through `summon()`. If you have one, the features are available.

### 2. `#[arcane]` Generates Safe-to-Call Code

```rust
// What you write:
#[arcane]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_setzero_ps();  // Safe inside #[target_feature]!
    // ...
}

// What the macro generates:
fn kernel(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    #[target_feature(enable = "avx2,fma,...")]
    unsafe fn __inner(data: &[f32; 8]) -> f32 {
        let v = _mm256_setzero_ps();
        // ...
    }
    // SAFETY: Token existence proves CPU support
    unsafe { __inner(data) }
}
```

The outer function is safe. The `unsafe` is an implementation detail justified by the token.

### 3. Rust 1.85+ Changed Everything

Value-based intrinsics are safe inside `#[target_feature]` functions:

```rust
#[target_feature(enable = "avx2")]
fn example() {
    let a = _mm256_setzero_ps();       // Safe
    let b = _mm256_add_ps(a, a);       // Safe
    let c = _mm256_fmadd_ps(a, b, c);  // Safe

    // Only memory operations need unsafe:
    let v = unsafe { _mm256_loadu_ps(ptr) };  // Raw pointer
}
```

## Two Crates

**Archmage** provides tokens, macros, and detection. Minimal.

**Magetypes** provides SIMD types (`f32x8`, `i32x4`), operators, methods, transcendentals, and cross-platform polyfills. Maximal.

## Idiomatic Patterns

### `#[rite]` inside, `#[arcane]` at the boundary

`#[rite]` adds `#[target_feature]` + `#[inline]` directly, so LLVM inlines it into callers with matching features. `#[arcane]` generates an inner `#[target_feature]` function called from a safe outer function (needed when transitioning from non-SIMD code — this crossing is the target-feature boundary).

```rust
pub fn public_api(data: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        process_simd(token, data)
    } else {
        data.iter().sum()
    }
}

#[arcane]  // Boundary: called from non-SIMD code
fn process_simd(token: Desktop64, data: &[f32]) -> f32 {
    let mut sum = 0.0;
    for chunk in data.chunks_exact(8) {
        sum += process_chunk(token, chunk.try_into().unwrap());
    }
    sum
}

#[rite]  // Internal: inlines into caller's target_feature context
fn process_chunk(token: Desktop64, chunk: &[f32; 8]) -> f32 {
    let v = f32x8::from_array(token, *chunk);
    v.reduce_add()
}
```

Calling `#[arcane]` from a hot loop crosses the `#[target_feature]` boundary every iteration (4-6x slower depending on workload; see [benchmark data](PERFORMANCE.md)). `#[rite]` inlines into callers with matching features — no boundary.

### Enter `#[arcane]` once, use `#[rite]` inside

The cost isn't `summon()` (~1.3 ns cached) — it's the `#[target_feature]` boundary. Each `#[arcane]` call from non-SIMD code crosses a boundary that LLVM can't inline across. Even hoisting the token outside the loop doesn't help — you need the loop *inside* `#[arcane]` with `#[rite]` helpers.

### Concrete tokens for hot paths

This is subtle: generic bounds create LLVM optimization barriers.

```rust
// Generic bound prevents LLVM from inlining across the call
#[arcane]
fn process<T: HasX64V2>(token: T, data: &[f32]) -> f32 { ... }

// Concrete token: full inlining, single target_feature region
#[arcane]
fn process(token: X64V3Token, data: &[f32]) -> f32 { ... }
```

`#[target_feature]` changes LLVM's compilation target for that function. Generic callers and concrete callees have mismatched targets, preventing optimization across the boundary. Downcasting (V4 -> V3) is free. Dispatch once at the entry point.

### Memory operations via `safe_unaligned_simd`

The prelude re-exports `safe_unaligned_simd`, which takes references instead of raw pointers:

```rust
use archmage::prelude::*;

#[arcane]
fn load_and_square(token: Desktop64, data: &[f32; 8]) -> __m256 {
    let v = _mm256_loadu_ps(data);  // Takes &[f32; 8], not *const f32
    _mm256_mul_ps(v, v)
}
```

For high-level code, prefer magetypes (which uses `safe_unaligned_simd` internally).

## The Soundness Invariant

```
features_enabled_by_arcane(Token) ⊆ features_checked_by_summon(Token)
```

Verified by `cargo xtask validate`.

## Teaching Checklist

When explaining archmage:

1. Tokens are zero-sized proofs of CPU features
2. `summon()` does the runtime check, returns `Option<Token>`
3. `#[arcane]` generates `#[target_feature]` code
4. Inside `#[target_feature]`, most intrinsics are safe (Rust 1.85+)
5. `#[arcane]` at the boundary, `#[rite]` for everything else
6. Enter `#[arcane]` once, `#[rite]` for everything inside
7. Concrete tokens optimize better than trait bounds
8. `safe_unaligned_simd` (in prelude) for memory operations
9. magetypes provides high-level SIMD types

When showing examples:

1. Show the simple path first (magetypes + `#[arcane]`)
2. Explain what the macro generates (for understanding)
3. Memory ops use `safe_unaligned_simd`, not raw pointers
4. Include the `summon()` call in context
5. Show `#[rite]` for any function called from SIMD code

## Banned from Docs

These prelude aliases exist for convenience but must not appear in documentation or examples:

| Alias | Use instead |
|-------|-------------|
| `F32Vec`, `I32Vec`, etc. | `f32x8`, `f32x4`, `i32x8`, `i32x4` |
| `RecommendedToken` | `Desktop64`, `Arm64`, `Wasm128Token` |
| `LANES` (outside `#[magetypes]`) | Explicit: `8`, `4`, or width in type name |

These aliases pretend platforms are interchangeable. An 8-wide AVX2 algorithm is fundamentally different from a 4-wide NEON algorithm. Width-generic code belongs inside `#[magetypes]`, where `Token`, `f32xN`, and `LANES` are substitution placeholders that generate explicit implementations per platform.

## Cross-Architecture

All tokens exist on all architectures. On wrong arch, `summon()` returns `None`:

```rust
#[arcane]
fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // On ARM: generates unreachable!() stub
}

if let Some(token) = X64V3Token::summon() {
    x86_kernel(token, &data)  // Only runs on x86 with AVX2
}
```

### Eliminating Runtime Dispatch

| Platform | Compile Flag | Effect |
|----------|--------------|--------|
| x86-64 AVX2 | `-Ctarget-cpu=haswell` | `Desktop64::summon()` compiles away |
| x86-64 AVX-512 | `-Ctarget-cpu=skylake-avx512` | `Server64::summon()` compiles away |
| AArch64 | (default target) | `Arm64::summon()` always succeeds (NEON is baseline) |
| WASM | `--target wasm32-unknown-unknown -Ctarget-feature=+simd128` | `Wasm128Token::summon()` compiles away |

## Open Design Questions

1. **Should magetypes root export SSE2 types on x86-64?** SSE2 is baseline. Currently the root exports all widths. No change yet.

2. **Implicit token downcasting:** Should `impl From<X64V4Token> for X64V3Token` exist? Not implemented — pass concrete tokens and downcast manually.

3. **Implicit vector downcasting:** Should `f32x8` offer extract-low-half to `f32x4`? Not implemented.

## Missing Methods

| Method | Description | Workaround |
|--------|-------------|------------|
| `signum` | Returns -1, 0, or 1 | Comparison + blend |
| `tanh` / `tanh_lowp` | Hyperbolic tangent | `(exp(2x) - 1) / (exp(2x) + 1)` |
| `sin` / `cos` | Trigonometric | Not implemented |
| `and` / `or` / `xor` on floats | Bitwise ops on float vectors | Cast to integer, operate, cast back |

Note: `rcp` is `rcp_approx`, `rsqrt` is `rsqrt_approx` for fast approximations. Use `recip()` and `rsqrt()` for full precision.

## License

MIT OR Apache-2.0
