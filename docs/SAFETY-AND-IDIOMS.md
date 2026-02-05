# Archmage Safety Model & Idiomatic Usage

This is the authoritative reference for understanding and teaching archmage correctly. Read this before writing docs, examples, or explanations.

## Terminology: Get These Right

| Term | Definition | Example |
|------|------------|---------|
| **safe** | No `unsafe` keyword required | `let x = vec![1,2,3];` |
| **unsafe** | Requires `unsafe` keyword (Rust says "I can't verify this") | `unsafe { ptr.read() }` |
| **sound** | Cannot cause UB when used as documented | Most safe Rust code |
| **unsound** | CAN cause UB even when used correctly | A buggy `unsafe` impl |

**Key insight:** These are orthogonal properties.
- Safe code is usually sound, but not always (e.g., pre-1.0 `mem::uninitialized`)
- Unsafe code CAN be sound if it upholds all invariants correctly
- Archmage's `#[arcane]` generates unsafe code that IS sound (token proves features exist)

## The Core Safety Model

### 1. Tokens Are Proofs, Not Runtime Overhead

```rust
#[derive(Clone, Copy)]  // Zero-sized!
pub struct X64V3Token;  // Contains no data

impl SimdToken for X64V3Token {
    fn summon() -> Option<Self> {
        if /* runtime CPUID check */ {
            Some(Self)  // Token existence = proof features are available
        } else {
            None
        }
    }
}
```

**Token existence is the proof.** You cannot construct a token except through `summon()` (or the deprecated `forge_token_dangerously()`). If you have a token, the CPU features are available. Period.

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
        let v = _mm256_setzero_ps();  // Safe because we're in #[target_feature]
        // ...
    }
    // SAFETY: Token existence proves CPU support
    unsafe { __inner(data) }
}
```

The outer function `kernel` is **safe to call**. It contains an `unsafe` block, but that's an implementation detail. The token parameter proves the invariant that makes it sound.

### 3. Rust 1.85+ Changed Everything

As of Rust 1.85 (target_feature_11 stabilization), **value-based intrinsics are safe inside `#[target_feature]` functions**:

```rust
#[target_feature(enable = "avx2")]
fn example() {
    // ALL of these are SAFE — no `unsafe` needed:
    let a = _mm256_setzero_ps();
    let b = _mm256_add_ps(a, a);
    let c = _mm256_mul_ps(b, b);
    let d = _mm256_fmadd_ps(a, b, c);

    // ONLY memory operations need `unsafe`:
    let v = unsafe { _mm256_loadu_ps(ptr) };  // Raw pointer
    unsafe { _mm256_storeu_ps(ptr, v) };       // Raw pointer
}
```

**This is why archmage works.** The `unsafe` in the generated code is just for calling the `#[target_feature]` function from non-target-feature context. Inside that function, most intrinsics are safe.

## Two Crates, Two Philosophies

### Archmage: Minimal

Archmage provides **only** what's needed for safe SIMD dispatch:

- **Tokens** — Proof of CPU features (`X64V3Token`, `NeonToken`, etc.)
- **`#[arcane]`** — Generate `#[target_feature]` wrapper (called from non-SIMD code)
- **`#[rite]`** — Add `#[target_feature]` + `#[inline]` (called from `#[arcane]`/`#[rite]`)
- **`incant!`** — Runtime dispatch to suffixed functions
- **Detection** — `summon()`, `guaranteed()`, feature macros

That's it. No SIMD types, no operator overloading, no method chains.

### Magetypes: Maximal

Magetypes provides **everything** for convenient SIMD programming:

- **SIMD types** — `f32x8`, `i32x4`, `u8x16`, etc.
- **Operators** — `+`, `-`, `*`, `/`, `&`, `|`, `^`
- **Methods** — `.abs()`, `.sqrt()`, `.blend()`, `.reduce_add()`, etc.
- **Transcendentals** — `.log2()`, `.exp()`, `.sin()` (approximations)
- **Memory ops** — `.load()`, `.store()`, `.from_array()`, `.to_array()`
- **Cross-platform** — Same API on x86/ARM/WASM via polyfills

All magetypes operations are token-gated and safe inside `#[arcane]`.

## Idiomatic Patterns

### Pattern 1: `#[rite]` Inside, `#[arcane]` at the Boundary

**`#[rite]` is for any function called exclusively from `#[arcane]` or `#[rite]` code.** It adds `#[target_feature]` + `#[inline]` with zero wrapper overhead.

**`#[arcane]` is for the boundary** where you transition from non-SIMD to SIMD code (after `summon()`).

```rust
// Entry point (called after summon) — use #[arcane]
pub fn public_api(data: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        process_simd(token, data)
    } else {
        data.iter().sum()
    }
}

#[arcane]
fn process_simd(token: Desktop64, data: &[f32]) -> f32 {
    let mut sum = 0.0;
    for chunk in data.chunks_exact(8) {
        sum += process_chunk(token, chunk.try_into().unwrap());
    }
    sum
}

// Called from #[arcane] — use #[rite]
#[rite]
fn process_chunk(token: Desktop64, chunk: &[f32; 8]) -> f32 {
    let v = f32x8::from_array(token, *chunk);
    v.reduce_add()
}
```

**Why this matters:** `#[arcane]` generates a wrapper. Calling `#[arcane]` from `#[arcane]` pays wrapper overhead on each call. `#[rite]` inlines into the caller's `#[target_feature]` context.

### Pattern 2: Summon Once, Pass Everywhere

**Token hoisting is critical for performance.**

```rust
// WRONG: 42% performance regression
fn process_pair(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    if let Some(token) = Desktop64::summon() {  // Called millions of times!
        process_simd(token, a, b)
    } else {
        0.0
    }
}

fn process_all(pairs: &[([f32; 8], [f32; 8])]) -> f32 {
    pairs.iter().map(|(a, b)| process_pair(a, b)).sum()
}
```

```rust
// RIGHT: Zero overhead
fn process_all(pairs: &[([f32; 8], [f32; 8])]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        // Summon ONCE at entry
        process_all_simd(token, pairs)
    } else {
        process_all_scalar(pairs)
    }
}

#[arcane]
fn process_all_simd(token: Desktop64, pairs: &[([f32; 8], [f32; 8])]) -> f32 {
    // Token passed through — no repeated summon()
    pairs.iter().map(|(a, b)| process_simd(token, a, b)).sum()
}
```

### Pattern 3: Concrete Tokens for Hot Paths

**Generic bounds are optimization barriers.**

```rust
// BAD: Generic bound prevents LLVM inlining
#[arcane]
fn process_generic<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    inner_work(token, data)  // May not inline — T could be any type
}

// GOOD: Concrete token enables full inlining
#[arcane]
fn process_concrete(token: X64V3Token, data: &[f32]) -> f32 {
    inner_work(token, data)  // Fully inlinable — concrete type
}
```

Downcasting is free (V4 → V3). Use concrete tokens and let the compiler optimize.

### Pattern 4: Memory Operations via `safe_unaligned_simd`

The prelude re-exports `safe_unaligned_simd`. This is THE way to do memory operations:

```rust
use archmage::prelude::*;  // Includes safe_unaligned_simd

#[arcane]
fn load_and_square(token: Desktop64, data: &[f32; 8]) -> __m256 {
    // safe_unaligned_simd takes references, not raw pointers
    let v = _mm256_loadu_ps(data);  // Safe! From prelude
    _mm256_mul_ps(v, v)             // Arithmetic is safe inside #[arcane]
}
```

For high-level code, prefer magetypes (which uses `safe_unaligned_simd` internally):

```rust
#[arcane]
fn load_and_square(token: Desktop64, data: &[f32; 8]) -> f32x8 {
    let v = f32x8::from_array(token, *data);
    v * v
}
```

## Common Mistakes

### Mistake 1: Calling Intrinsics "Unsafe"

**WRONG:** "Intrinsics are unsafe, so you need `unsafe` blocks."

**RIGHT:** Value-based intrinsics are **safe inside `#[target_feature]`** (Rust 1.85+). Only memory operations (raw pointers) need `unsafe`.

### Mistake 2: Over-Explaining Safety

**WRONG:** Long explanations about why `#[arcane]` is sound, with disclaimers about unsafe code.

**RIGHT:** "The token proves CPU support. Inside `#[arcane]`, SIMD operations are safe."

### Mistake 3: Manual `#[target_feature]` in Examples

**WRONG:**
```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn inner(data: &[f32]) -> f32 { ... }

fn outer(token: X64V3Token, data: &[f32]) -> f32 {
    unsafe { inner(data) }
}
```

**RIGHT:**
```rust
#[arcane]
fn process(token: X64V3Token, data: &[f32]) -> f32 {
    // Just write the code
}
```

### Mistake 4: Using `#[arcane]` for Internal Functions

**WRONG:** `#[arcane]` on functions that are only called from other SIMD code.

**RIGHT:** `#[arcane]` at the boundary (after `summon()`), `#[rite]` for everything called from `#[arcane]`/`#[rite]`.

### Mistake 5: Forgetting Token in Dispatch

**WRONG:**
```rust
pub fn api(data: &[f32]) -> f32 {
    if Desktop64::summon().is_some() {
        process_simd(data)  // Where's the token?
    } else {
        process_scalar(data)
    }
}
```

**RIGHT:**
```rust
pub fn api(data: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        process_simd(token, data)  // Pass the token!
    } else {
        process_scalar(data)
    }
}
```

## The Soundness Invariant

For archmage to be sound:

```
features_enabled_by_arcane(Token) ⊆ features_checked_by_summon(Token)
```

The `#[arcane]` macro enables features based on the token type. `summon()` checks those same features at runtime. If they match, the system is sound.

This is verified by `cargo xtask validate` which checks that the macro's feature lists match the summon() implementations.

## Teaching Checklist

When explaining archmage:

1. ✅ Tokens are zero-sized proofs of CPU features
2. ✅ `summon()` does the runtime check, returns `Option<Token>`
3. ✅ `#[arcane]` generates `#[target_feature]` code
4. ✅ Inside `#[target_feature]`, most intrinsics are safe (Rust 1.85+)
5. ✅ `#[arcane]` at the boundary, `#[rite]` for anything called from SIMD code
6. ✅ Summon once at API boundary, pass token through
7. ✅ Concrete tokens enable better optimization than trait bounds
8. ✅ `safe_unaligned_simd` (in prelude) is how you do memory operations
9. ✅ magetypes provides high-level SIMD types that use tokens

When showing examples:

1. ✅ Show the simple path first (magetypes + `#[arcane]`)
2. ✅ Explain what the macro generates (for understanding)
3. ✅ Memory ops use `safe_unaligned_simd` from the prelude, not raw pointers
4. ✅ Include the `summon()` call in context
5. ✅ Show `#[rite]` for any function called from `#[arcane]`/`#[rite]`

## Banned from Code AND Docs

**NEVER use or document these prelude aliases:**

| ❌ Banned | ✅ Use instead |
|-----------|----------------|
| `F32Vec`, `I32Vec`, etc. | `f32x8`, `f32x4`, `i32x8`, `i32x4` |
| `RecommendedToken` | `Desktop64`, `Arm64`, `Simd128Token` |
| `LANES` (outside `#[magetypes]`) | Explicit: `8`, `4`, or width in type name |

**Why:** These aliases pretend platforms are interchangeable. They're not. An 8-wide AVX2 algorithm is fundamentally different from a 4-wide NEON algorithm. Hiding the width breeds bugs and confusion.

**The only place width-generic code belongs:** Inside `#[magetypes]` macro, where `Token`, `f32xN`, and `LANES` are substitution placeholders that generate explicit, separate implementations for each platform.

```rust
// ❌ WRONG: Hides what's actually happening
use magetypes::prelude::{F32Vec, RecommendedToken, LANES};

fn process(data: &[f32]) -> f32 {
    if let Some(token) = RecommendedToken::summon() {
        // What width? Who knows!
    }
}

// ✅ RIGHT: Explicit types, explicit dispatch
fn process(data: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        process_avx2(token, data)  // 8-wide, f32x8
    } else if let Some(token) = Arm64::summon() {
        process_neon(token, data)  // 4-wide, f32x4
    } else {
        process_scalar(data)
    }
}

// ✅ ALSO RIGHT: #[magetypes] for code generation
#[magetypes]
fn process_impl(token: Token, data: &[f32]) -> f32 {
    // Token, f32xN, LANES are placeholders here — generates separate functions
}

pub fn process(data: &[f32]) -> f32 {
    incant!(process_impl(data))
}
```

## Cross-Architecture Notes

All tokens exist on all architectures. On wrong arch, `summon()` returns `None`:

```rust
// This compiles on ARM:
#[arcane]
fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // On ARM: generates unreachable!() stub
}

// Dispatch works on all platforms:
if let Some(token) = X64V3Token::summon() {
    x86_kernel(token, &data)  // Only runs on x86 with AVX2
}
```

This enables single-crate cross-platform SIMD without `#[cfg]` everywhere.

### Eliminating Runtime Dispatch

| Platform | Compile Flag | Effect |
|----------|--------------|--------|
| x86-64 AVX2 | `-Ctarget-cpu=haswell` | `Desktop64::summon()` compiles away |
| x86-64 AVX-512 | `-Ctarget-cpu=skylake-avx512` | `Server64::summon()` compiles away |
| AArch64 | (default target) | `Arm64::summon()` always succeeds (NEON is baseline) |
| WASM | `--target wasm32-unknown-unknown -Ctarget-feature=+simd128` | `Simd128Token::summon()` compiles away |

**For ARM binaries:** NEON is mandatory on AArch64, so `Arm64::summon()` always returns `Some`. If you're distributing ARM binaries, there's no runtime dispatch needed for basic NEON—just use the token directly. The `summon()` call is still good practice for code clarity and cross-compilation safety.

**For x86 binaries with known deployment:** If you control the deployment environment and know AVX2 is available, compile with `-Ctarget-cpu=haswell` (or `native` for your machine). The `summon()` becomes a no-op that the compiler eliminates entirely.

## Open Design Questions

These are recorded for future consideration:

1. **Should magetypes root export SSE2 types on x86-64?** SSE2 is baseline on x86-64. Currently the root exports all widths (w128, w256, w512). Should w128 (SSE2) be the "default" with higher tiers in submodules?

2. **Submodule organization:** Current structure works because types are named by width (`f32x4`, `f32x8`). Submodules exist for explicit access:
   - `magetypes::simd::x86::w128::f32x4` — SSE
   - `magetypes::simd::x86::w256::f32x8` — AVX2
   - `magetypes::simd::arm::w128::f32x4` — NEON

   The root `magetypes::simd::*` re-exports all, which is fine since names don't conflict. **Recommendation:** Keep current structure, document submodule paths for explicit access.

3. **Implicit token downcasting:** Can you pass `X64V4Token` where `X64V3Token` is expected? Currently no implicit conversion. Should there be `impl From<X64V4Token> for X64V3Token`?

4. **Implicit vector type downcasting:** Can you use an `f32x8` (AVX2) in a context expecting `f32x4` (SSE) by extracting the low half? This needs explicit methods today.

## Missing Methods (should be added to magetypes)

These methods were expected but don't exist:

| Method | Description | Workaround |
|--------|-------------|------------|
| `signum` | Returns -1, 0, or 1 | Use comparison + blend |
| `tanh` / `tanh_lowp` | Hyperbolic tangent | Implement via `(exp(2x) - 1) / (exp(2x) + 1)` |
| `sin` / `cos` | Trigonometric functions | Not implemented |
| `and` / `or` / `xor` on floats | Bitwise ops | Cast to integer type, operate, cast back |

Note: `rcp` is `rcp_approx`, `rsqrt` is `rsqrt_approx` for fast approximations. Use `recip()` and `rsqrt()` for full precision.
