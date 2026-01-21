# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

## CRITICAL: Documentation Examples

**ALWAYS use `archmage::mem` for load/store in examples.** The entire point of this crate is to make SIMD safe. Never write examples with `unsafe { _mm256_loadu_ps(ptr) }` - that defeats the purpose.

```rust
// WRONG - bypasses the safety archmage provides
let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

// CORRECT - use the safe mem wrappers
let v = avx::_mm256_loadu_ps(token, data);
```

## Quick Start

```bash
cargo test                    # Run tests
cargo test --all-features     # Test with all integrations
cargo clippy --all-features   # Lint
cargo run -p xtask -- generate # Regenerate safe_unaligned_simd wrappers
```

## CRITICAL: Adding New Traits or Tokens

The `#[arcane]` macro in `archmage-macros` has hardcoded mappings from traits/tokens to CPU features. **These must stay in sync.**

When adding a new trait or token:
1. Add the trait/token to `src/tokens/`
2. **Update `archmage-macros/src/lib.rs`:**
   - `trait_to_features()` - for new marker traits
   - `token_to_features()` - for new token types
3. **Add a test to `tests/trait_token_sync.rs`** - compile-time verification

If `tests/trait_token_sync.rs` fails to compile, the macro mappings are out of sync. The compiler error tells you exactly which trait/token is unrecognized.

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
   - Tokens can only be created via `summon()` which checks CPUID
   - If you have a token, the CPU supports the features

## Generic Token Bounds

Use trait bounds to accept any compatible token:

```rust
#[arcane]
fn process(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
    // Works with Avx2Token, X64V3Token, X64V4Token, etc.
}

#[arcane]
fn fma_kernel<T: HasAvx2Fma>(token: T, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    // Requires AVX2 + FMA (use HasAvx2Fma, not separate traits)
}
```

**Recommended starting point:** `Desktop64` (alias for `X64V3Token`)

## Friendly Aliases

Use these intuitive names instead of memorizing microarchitecture levels:

| Alias | Target | What it means |
|-------|--------|---------------|
| `Desktop64` | x86_64 desktops | AVX2 + FMA (Haswell 2013+, Zen 1+) |
| `Server64` | x86_64 servers | + AVX-512 (Xeon 2017+, Zen 4+) |
| `Arm64` | AArch64 | NEON (all 64-bit ARM) |

**Why these names?**
- `Desktop64` - Universal on modern desktops. Intel removed AVX-512 from consumer chips (12th-14th gen), so this is the safe choice.
- `Server64` - AVX-512 is reliable on Xeon servers, Intel HEDT, and AMD Zen 4+.
- `Arm64` - NEON is baseline on all AArch64, always available.

```rust
use archmage::{Desktop64, SimdToken, arcane};

#[arcane]
fn process(token: Desktop64, data: &mut [f32; 8]) {
    // AVX2 + FMA intrinsics safe here
}

if let Some(token) = Desktop64::summon() {
    process(token, &mut data);
}
```

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

**Recommended (friendly aliases):**
- `Desktop64` = `X64V3Token` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+) **← Start here for x86**
- `Server64` = `X64V4Token` = `Avx512Token` - + AVX-512 F+BW+CD+DQ+VL (Xeon 2017+, Zen 4+)
- `Arm64` - NEON baseline **← Start here for ARM**

**x86 Feature Tokens:**
- `Sse42Token` (baseline) → `AvxToken` → `Avx2Token` → `Avx2FmaToken`
- `Avx512ModernToken` - Avx512 + VBMI2+VNNI+BF16 etc (Ice Lake/Zen 4)
- `Avx512Fp16Token` - Avx512 + FP16 (Sapphire Rapids+)

**ARM:**
- `NeonToken`, `NeonAesToken`, `NeonSha3Token`, `NeonFp16Token` = `Arm64`

## Marker Traits

Enable generic bounds:

```rust
fn requires_avx2(token: impl HasAvx2) { ... }
fn requires_fma(token: impl HasAvx2Fma) { ... }  // AVX2+FMA combined
fn requires_v3<T: HasX64V3>(token: T) { ... }
```

**Width traits:** `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`
**x86 Feature traits:** `HasSse42`, `HasAvx`, `HasAvx2`, `HasAvx2Fma`, `HasX64V3`, `HasAvx512`, `HasModernAvx512`
**Alias traits:** `HasDesktop64` = `HasX64V3`, `HasServer64` = `HasAvx512` (all equivalent)
**ARM traits:** `HasNeon`, `HasArmAes`, `HasArmSha3`, `HasArmFp16`, `HasArm64`

## Safe Memory Operations

With `safe_unaligned_simd` feature, the `mem` module provides reference-based load/store:

```rust
use archmage::{Desktop64, SimdToken, mem::avx};

if let Some(token) = Desktop64::summon() {
    let v = avx::_mm256_loadu_ps(token, &data);  // Safe! Reference, not pointer
    avx::_mm256_storeu_ps(token, &mut out, v);
}
```

## Methods with Self Receivers

Use `_self = Type` to enable self receivers. Use `_self` in the body instead of `self`:

```rust
trait SimdOps {
    fn double(&self, token: impl HasAvx2) -> Self;
}

impl SimdOps for [f32; 8] {
    #[arcane(_self = [f32; 8])]
    fn double(&self, _token: impl HasAvx2) -> Self {
        // Use _self instead of self
        let v = unsafe { _mm256_loadu_ps(_self.as_ptr()) };
        let doubled = _mm256_add_ps(v, v);
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
        out
    }
}
```

**All receiver types supported:** `self`, `&self`, `&mut self`

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
